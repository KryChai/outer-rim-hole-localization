"""业务数据 JSONL 留底：按接口分文件、按天滚动、超期自动清理。

写入约定：
- 每个接口一次调用，产生若干孔级/目标级行 + 一行 summary；
- ``request_id`` 由请求日志中间件经 ContextVar 注入，用于串联运行日志与业务行；
- 任何写入失败只记 warning，绝不影响接口响应。
"""

from __future__ import annotations

import json
import threading
from contextvars import ContextVar
from datetime import datetime, timedelta
from pathlib import Path
from uuid import uuid4

from app.core.config import settings
from app.core.logging import logger

_LOCK = threading.Lock()
_REQUEST_ID: ContextVar[str] = ContextVar("audit_request_id", default="")
_LAST_CLEANUP: dict[str, str] = {}

# fit_quality 中体积大且可由其他字段推导的明细，不落盘（接口响应中仍完整返回）
_FIT_QUALITY_SKIP = ("perturbations",)


def bind_request_id() -> str:
    """请求中间件调用：生成本次请求的 8 位短 id 并注入上下文。"""
    request_id = uuid4().hex[:8]
    _REQUEST_ID.set(request_id)
    return request_id


def _base_row(endpoint: str, conf) -> dict:
    return {
        "ts": datetime.now().isoformat(timespec="milliseconds"),
        "request_id": _REQUEST_ID.get(),
        "endpoint": endpoint,
        "conf": conf,
    }


def _append_rows(category: str, rows: list) -> None:
    if not settings.business_log_enabled or not rows:
        return
    try:
        directory = Path(settings.log_dir) / "business"
        with _LOCK:
            directory.mkdir(parents=True, exist_ok=True)
            path = directory / f"{category}_{datetime.now():%Y-%m-%d}.jsonl"
            with path.open("a", encoding="utf-8") as fh:
                for row in rows:
                    fh.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
        _cleanup_expired(directory)
    except Exception:
        logger.warning("业务日志写入失败（category=%s）", category, exc_info=True)


def _cleanup_expired(directory: Path) -> None:
    """每个目录每天最多清理一次，删除超过保留期的历史 JSONL。"""
    key = str(directory)
    today = f"{datetime.now():%Y-%m-%d}"
    if _LAST_CLEANUP.get(key) == today:
        return
    _LAST_CLEANUP[key] = today
    cutoff = datetime.now() - timedelta(days=settings.business_log_retention_days)
    for path in directory.glob("*.jsonl"):
        try:
            file_date = datetime.strptime(path.stem.rsplit("_", 1)[-1], "%Y-%m-%d")
        except ValueError:
            continue
        if file_date < cutoff:
            path.unlink()


def _shape_fields(hole: dict) -> dict:
    """圆/椭圆拟合的几何概要（完整推导参数见接口响应，此处只留统计常用字段）。"""
    circle = hole.get("circle")
    if circle:
        return {"radius_px": circle.get("radius"), "diameter_px": circle.get("diameter")}
    ellipse = hole.get("ellipse") or {}
    return {
        "semi_major_axis_px": ellipse.get("semi_major_axis_px"),
        "semi_minor_axis_px": ellipse.get("semi_minor_axis_px"),
        "axis_ratio": ellipse.get("axis_ratio"),
        "eccentricity": ellipse.get("eccentricity"),
        "rotation_angle_deg": ellipse.get("rotation_angle_deg"),
    }


def _refine_common(data: dict) -> dict:
    image_size = data.get("image_size") or {}
    return {
        "algorithm": data.get("algorithm"),
        "image_w": image_size.get("width"),
        "image_h": image_size.get("height"),
        "bbox_expansion_fraction": data.get("bbox_expansion_fraction"),
        "sigma_px": data.get("sigma_px"),
        "check_stability": data.get("check_stability"),
        "reason": data.get("reason", ""),
    }


def log_refine(endpoint: str, conf, data: dict) -> None:
    """circle/ellipse_refine 接口：每孔一行 + 一行 summary。"""
    common = _refine_common(data)
    rows = []
    for hole in data.get("holes", []):
        quality = dict(hole.get("fit_quality") or {})
        for key in _FIT_QUALITY_SKIP:
            quality.pop(key, None)
        center = hole.get("center") or {}
        rows.append(_base_row(endpoint, conf) | common | {
            "kind": "hole",
            "hole_idx": hole.get("detection_index"),
            "det_conf": hole.get("confidence"),
            "detection_center": hole.get("detection_center"),
            "detection_bbox": hole.get("detection_bbox"),
            "expanded_bbox": hole.get("expanded_bbox"),
            "roi_bbox": hole.get("roi_bbox"),
            "status": hole.get("status"),
            "center_x": center.get("x"),
            "center_y": center.get("y"),
            **_shape_fields(hole),
            "fit_quality": quality,
        })
    rows.append(_base_row(endpoint, conf) | common
                | {"kind": "summary", "summary": data.get("summary")})
    _append_rows("refine", rows)


def log_detect(endpoint: str, conf, data: dict) -> None:
    """单孔检测与整圆拟合接口：每孔一行 + 一行 summary（含整圆拟合质量）。"""
    common = {"msg": data.get("msg", "")}
    holes = data.get("all_holes")
    if holes is None:  # 单孔最高置信度接口
        best = data.get("max_conf_hole")
        holes = [best] if best else []
    rows = []
    for idx, hole in enumerate(holes):
        rows.append(_base_row(endpoint, conf) | common | {
            "kind": "hole",
            "hole_idx": idx,
            "det_conf": hole.get("conf"),
            "center_x": hole.get("center_x"),
            "center_y": hole.get("center_y"),
            "detection_bbox": {k: hole.get(k) for k in ("x1", "y1", "x2", "y2")},
        })
    rows.append(_base_row(endpoint, conf) | common | {
        "kind": "summary",
        "detected": len(rows),
        "fit_circle": data.get("fit_circle"),
        "fit_quality": data.get("fit_quality"),
    })
    _append_rows("detect", rows)


def log_tracking(endpoint: str, conf, session_id, data: dict) -> None:
    """孔位编号追踪与最近孔接口：每孔一行 + 一行 summary。"""
    common = {"msg": data.get("msg", "")}
    rows = []
    tracked = data.get("holes")
    if tracked is not None:  # 编号追踪接口
        for idx, hole in enumerate(tracked):
            rows.append(_base_row(endpoint, conf) | common | {
                "kind": "hole",
                "hole_idx": idx,
                "hole_id": hole.get("hole_id"),
                "det_conf": hole.get("conf"),
                "center_x": hole.get("center_x"),
                "center_y": hole.get("center_y"),
                "angle_deg": hole.get("angle_deg"),
                "missing": hole.get("missing"),
            })
    else:  # 最近孔接口
        nearest = data.get("hole")
        if nearest:
            rows.append(_base_row(endpoint, conf) | common | {
                "kind": "hole",
                "hole_idx": 0,
                "det_conf": nearest.get("conf"),
                "center_x": nearest.get("center_x"),
                "center_y": nearest.get("center_y"),
            })
    rows.append(_base_row(endpoint, conf) | common | {
        "kind": "summary",
        "session_id": session_id,
        "tracked": len(rows),
        "circle": data.get("circle"),
    })
    _append_rows("tracking", rows)


def log_judge(endpoint: str, conf, data: dict) -> None:
    """拧紧/对准判断接口：每目标一行 + 一行 summary。"""
    common = {"msg": data.get("msg", "")}
    rows = []
    for idx, det in enumerate(data.get("detections", [])):
        rows.append(_base_row(endpoint, conf) | common | {
            "kind": "target",
            "target_idx": idx,
            "class_id": det.get("class_id"),
            "class_name": det.get("class_name"),
            "det_conf": det.get("confidence"),
            "center_x": det.get("center_x"),
            "center_y": det.get("center_y"),
        })
    rows.append(_base_row(endpoint, conf) | common
                | {"kind": "summary", "detected": len(rows)})
    _append_rows("judge", rows)
