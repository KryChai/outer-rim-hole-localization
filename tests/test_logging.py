"""运行日志与业务 JSONL 留底单元测试（不依赖模型权重与网络）。"""

import json
import logging
import logging.handlers
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

from app.core import audit
from app.core.logging import setup_logging


def _fake_settings(tmp_path: Path, enabled: bool = True, retention: int = 30) -> SimpleNamespace:
    return SimpleNamespace(
        log_dir=str(tmp_path),
        business_log_enabled=enabled,
        business_log_retention_days=retention,
    )


def _read_rows(tmp_path: Path, category: str) -> list:
    path = tmp_path / "business" / f"{category}_{datetime.now():%Y-%m-%d}.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _refine_data() -> dict:
    hole_ok = {
        "detection_index": 0, "confidence": 0.9,
        "detection_center": {"x": 100.0, "y": 100.0},
        "detection_bbox": {"x1": 60.0, "y1": 60.0, "x2": 140.0, "y2": 140.0},
        "expanded_bbox": {"x1": 56.0, "y1": 56.0, "x2": 144.0, "y2": 144.0},
        "roi_bbox": {"x1": 56.0, "y1": 56.0, "x2": 144.0, "y2": 144.0},
        "status": "ok", "center": {"x": 100.2, "y": 99.8},
        "circle": {"center_x": 100.2, "center_y": 99.8, "radius": 41.5, "diameter": 83.0},
        "fit_quality": {"valid": True, "flags": [], "reason": "", "rmse_px": 0.3,
                        "scale_center_shift_px": 0.12,
                        "perturbations": [{"dx": -3, "shift_px": 0.2}]},
    }
    hole_review = {
        "detection_index": 1, "confidence": 0.7,
        "detection_center": {"x": 300.0, "y": 100.0},
        "detection_bbox": {"x1": 260.0, "y1": 60.0, "x2": 340.0, "y2": 140.0},
        "expanded_bbox": None, "roi_bbox": None,
        "status": "review", "center": {"x": 300.5, "y": 100.1},
        "ellipse": {"semi_major_axis_px": 42.0, "semi_minor_axis_px": 40.0,
                    "axis_ratio": 0.95, "eccentricity": 0.3, "rotation_angle_deg": 12.0},
        "fit_quality": {"valid": False, "flags": ["smoothing_sensitive"],
                        "reason": "smoothing_sensitive", "rmse_px": 1.4,
                        "scale_center_shift_px": None},
    }
    return {
        "algorithm": "test", "image_size": {"width": 640, "height": 480},
        "bbox_expansion_fraction": 0.1, "sigma_px": 1.4, "check_stability": True,
        "reason": "",
        "summary": {"detected": 2, "ok": 1, "review": 1, "failed": 0},
        "holes": [hole_ok, hole_review],
    }


def test_setup_logging_writes_rotating_file(tmp_path, monkeypatch):
    monkeypatch.setattr("app.core.logging.settings", SimpleNamespace(log_dir=str(tmp_path)))
    root = logging.getLogger()
    handlers_before = list(root.handlers)
    try:
        setup_logging()
        assert any(isinstance(h, logging.handlers.RotatingFileHandler) for h in root.handlers)
        logging.getLogger("yolo_service.test").info("hello-file-log")
        for handler in root.handlers:
            handler.flush()
        content = (tmp_path / "yolo_service.log").read_text(encoding="utf-8")
        assert "hello-file-log" in content
    finally:
        root.handlers[:] = handlers_before


def test_log_refine_rows_and_summary(tmp_path, monkeypatch):
    monkeypatch.setattr(audit, "settings", _fake_settings(tmp_path))
    audit.bind_request_id()
    audit.log_refine("usb_hole11_circle_refine", 0.5, _refine_data())

    rows = _read_rows(tmp_path, "refine")
    assert [row["kind"] for row in rows] == ["hole", "hole", "summary"]
    hole_ok, hole_review, summary = rows
    # 公共字段
    assert all(row["request_id"] for row in rows)
    assert all(row["endpoint"] == "usb_hole11_circle_refine" for row in rows)
    assert all(row["conf"] == 0.5 for row in rows)
    # 圆孔：几何概要 + fit_quality 全量（不含 perturbations 明细）
    assert hole_ok["status"] == "ok"
    assert hole_ok["radius_px"] == 41.5
    assert hole_ok["center_x"] == 100.2
    assert "perturbations" not in hole_ok["fit_quality"]
    assert hole_ok["fit_quality"]["scale_center_shift_px"] == 0.12
    # 椭圆孔：字段切换为椭圆概要
    assert hole_review["eccentricity"] == 0.3
    assert hole_review["fit_quality"]["scale_center_shift_px"] is None
    assert hole_review["fit_quality"]["flags"] == ["smoothing_sensitive"]
    # summary 行
    assert summary["summary"] == {"detected": 2, "ok": 1, "review": 1, "failed": 0}


def test_log_detect_circle_fit_shape(tmp_path, monkeypatch):
    monkeypatch.setattr(audit, "settings", _fake_settings(tmp_path))
    data = {
        "msg": "共检测到 3 个螺孔",
        "all_holes": [
            {"conf": 0.9, "center_x": 10.0, "center_y": 20.0, "x1": 0.0, "y1": 10.0,
             "x2": 20.0, "y2": 30.0},
            {"conf": 0.8, "center_x": 30.0, "center_y": 20.0, "x1": 20.0, "y1": 10.0,
             "x2": 40.0, "y2": 30.0},
        ],
        "fit_circle": {"circle_center_x": 20.0, "circle_center_y": 20.0, "radius": 10.0},
        "fit_quality": {"valid": True, "point_count": 2, "reason": "",
                        "residual_rms_px": 0.1, "residual_max_px": 0.2},
    }
    audit.log_detect("hole11_circle_fit", 0.5, data)
    rows = _read_rows(tmp_path, "detect")
    assert [row["kind"] for row in rows] == ["hole", "hole", "summary"]
    assert rows[0]["detection_bbox"] == {"x1": 0.0, "y1": 10.0, "x2": 20.0, "y2": 30.0}
    assert rows[-1]["fit_circle"]["radius"] == 10.0
    assert rows[-1]["detected"] == 2


def test_log_tracking_and_judge(tmp_path, monkeypatch):
    monkeypatch.setattr(audit, "settings", _fake_settings(tmp_path))
    audit.log_tracking("usb_hole11_track", 0.5, "s1", {
        "msg": "共追踪 1 个孔位",
        "holes": [{"hole_id": 1, "conf": 0.9, "center_x": 10.0, "center_y": 20.0,
                   "angle_deg": 30.0, "missing": False}],
        "circle": {"circle_center_x": 5.0, "circle_center_y": 5.0, "radius": 10.0},
    })
    track_rows = _read_rows(tmp_path, "tracking")
    assert track_rows[0]["hole_id"] == 1 and track_rows[0]["missing"] is False
    assert track_rows[1]["kind"] == "summary" and track_rows[1]["session_id"] == "s1"
    assert track_rows[1]["circle"]["radius"] == 10.0

    audit.log_judge("bolt7_judge", 0.5, {
        "msg": "共检测到 1 个目标",
        "detections": [{"class_id": 0, "class_name": "bolt", "confidence": 0.88,
                        "center_x": 15.0, "center_y": 25.0}],
    })
    judge_rows = _read_rows(tmp_path, "judge")
    assert judge_rows[0]["class_name"] == "bolt"
    assert judge_rows[1]["detected"] == 1


def test_business_log_disabled_writes_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr(audit, "settings", _fake_settings(tmp_path, enabled=False))
    audit.log_refine("usb_hole11_circle_refine", 0.5, _refine_data())
    assert not (tmp_path / "business").exists()


def test_write_failure_does_not_raise(tmp_path, monkeypatch):
    blocked = tmp_path / "not_a_dir"
    blocked.write_text("occupied", encoding="utf-8")
    monkeypatch.setattr(audit, "settings", _fake_settings(blocked))
    # 目录被同名文件占用时写入必然失败，但不应影响接口响应
    audit.log_refine("usb_hole11_circle_refine", 0.5, _refine_data())


def test_cleanup_expired_jsonl(tmp_path, monkeypatch):
    monkeypatch.setattr(audit, "settings", _fake_settings(tmp_path, retention=30))
    business = tmp_path / "business"
    business.mkdir(parents=True)
    old = business / "refine_2020-01-01.jsonl"
    fresh = business / "refine_2099-12-31.jsonl"
    old.write_text("{}\n", encoding="utf-8")
    fresh.write_text("{}\n", encoding="utf-8")
    audit._cleanup_expired(business)
    assert not old.exists()
    assert fresh.exists()
