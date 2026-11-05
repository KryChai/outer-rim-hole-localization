"""检测结果解析与业务计算逻辑。

将 YOLO 推理结果转换为接口需要的数据结构，包含螺孔中心提取、
最小二乘拟合圆以及通用目标解析。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def circle_fit_quality(points):
    """Centered least squares, finite/rank checks, normalized residual quality."""
    quality = {"valid": False, "point_count": len(points), "reason": ""}
    if len(points) < 3:
        quality["reason"] = "fewer_than_three_points"
        return (None, None, None), quality
    pts = np.asarray(points, dtype=np.float64)
    if pts.shape != (len(points), 2) or not np.isfinite(pts).all():
        quality["reason"] = "invalid_coordinates"
        return (None, None, None), quality
    mean = pts.mean(axis=0)
    xy = pts-mean
    scale = float(np.max(np.linalg.norm(xy, axis=1)))
    if scale < 1e-9:
        quality["reason"] = "degenerate_points"
        return (None, None, None), quality
    xy /= scale
    h = np.column_stack([2*xy, np.ones(len(pts))])
    coeff, _, rank, singular = np.linalg.lstsq(h, np.sum(xy*xy, axis=1), rcond=None)
    if rank < 3 or singular[-1] / singular[0] < 1e-8:
        quality["reason"] = "collinear_or_ill_conditioned"
        return (None, None, None), quality
    squared_radius = float(np.dot(coeff[:2], coeff[:2])+coeff[2])
    if squared_radius <= 0 or not np.isfinite(squared_radius):
        quality["reason"] = "invalid_radius"
        return (None, None, None), quality
    center = mean+scale*coeff[:2]
    radius = scale*np.sqrt(squared_radius)
    residual = np.linalg.norm(pts-center, axis=1)-radius
    quality.update(residual_rms_px=float(np.sqrt(np.mean(residual**2))),
                   residual_max_px=float(np.max(np.abs(residual))))
    if quality["residual_max_px"] > 0.10*radius:
        quality["reason"] = "excessive_residual"
        return (None, None, None), quality
    quality.update(valid=True, reason="")
    return (float(center[0]), float(center[1]), float(radius)), quality


def fit_circle(points: List[List[float]]) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """Backward-compatible circle API; corrected radius constant sign."""
    return circle_fit_quality(points)[0]


def parse_holes(result: Any, conf: float, class_id: int = 0) -> Tuple[List[Dict[str, float]], List[List[float]]]:
    """解析指定类别、置信度达标的螺孔中心，返回 (holes, point_set)。"""
    holes: List[Dict[str, float]] = []
    point_set: List[List[float]] = []

    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return holes, point_set

    for box in boxes:
        cls_id = int(box.cls.cpu().item())
        box_conf = float(box.conf.cpu().item())
        if cls_id != class_id or box_conf < conf:
            continue

        x1, y1, x2, y2 = box.xyxy.cpu().numpy()[0]
        center_x = float((x1 + x2) / 2)
        center_y = float((y1 + y2) / 2)
        holes.append({"conf": box_conf, "center_x": center_x, "center_y": center_y})
        point_set.append([center_x, center_y])

    return holes, point_set


def parse_detections(result: Any) -> List[Dict[str, Any]]:
    """解析全部检测目标（不按类别过滤，用于拧紧/对准判断）。"""
    detections: List[Dict[str, Any]] = []
    class_names = result.names

    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return detections

    for box in boxes:
        cls_id = int(box.cls.cpu().item())
        box_conf = float(box.conf.cpu().item())
        x1, y1, x2, y2 = box.xyxy.cpu().numpy()[0]
        detections.append(
            {
                "class_id": cls_id,
                "class_name": class_names.get(cls_id, str(cls_id)),
                "confidence": round(box_conf, 3),
                "center_x": float((x1 + x2) / 2),
                "center_y": float((y1 + y2) / 2),
            }
        )

    return detections


def build_best_hole_data(result: Any, conf: float) -> Dict[str, Any]:
    """构造「置信度最高单孔」接口返回数据。"""
    if result is None:
        return {"msg": "模型未返回检测结果", "max_conf_hole": None}

    holes, _ = parse_holes(result, conf)
    if not holes:
        return {"msg": "未检测到符合置信度的孔", "max_conf_hole": None}

    holes.sort(key=lambda item: item["conf"], reverse=True)
    return {"msg": "成功获取置信度最高孔洞坐标", "max_conf_hole": holes[0]}


def build_circle_fit_data(result: Any, conf: float, noun: str = "螺孔") -> Dict[str, Any]:
    """构造「全部螺孔 + 拟合圆」接口返回数据。"""
    if result is None:
        return {"msg": "模型未返回检测结果", "all_holes": [], "fit_circle": None}

    holes, point_set = parse_holes(result, conf)
    fit_result = None
    quality = {"valid": False, "point_count": len(point_set), "reason": "fewer_than_three_points"}
    if len(point_set) >= 3:
        (cx, cy, radius), quality = circle_fit_quality(point_set)
        if cx is not None:
            fit_result = {"circle_center_x": cx, "circle_center_y": cy, "radius": radius}

    return {
        "msg": f"共检测到 {len(holes)} 个符合阈值{noun}",
        "all_holes": holes,
        "fit_circle": fit_result,
        "fit_quality": quality,
    }


def build_judge_data(result: Any) -> Dict[str, Any]:
    """构造「拧紧/对准判断」接口返回数据。"""
    if result is None:
        return {"msg": "模型未返回检测结果", "detections": []}

    detections = parse_detections(result)
    return {"msg": f"共检测到 {len(detections)} 个目标", "detections": detections}
