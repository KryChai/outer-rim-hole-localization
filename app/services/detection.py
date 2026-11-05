"""检测结果解析与业务计算逻辑。

将 YOLO 推理结果转换为接口需要的数据结构，包含螺孔中心提取、
最小二乘拟合圆、透视纠正圆心以及通用目标解析。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from scipy.optimize import least_squares

# 单应精化残差门限（px）：等分布局下残差主要来自检测噪声（~1px 级），
# 超过 max(ELLIPSE_RESIDUAL_FLOOR_PX, 0.02 * 半长轴) 视为非等分，回退椭圆中心
ELLIPSE_RESIDUAL_FLOOR_PX = 2.0
ELLIPSE_RESIDUAL_SCALE = 0.02
# 椭圆拟合防线：最小扁平率（拦截共线退化）与点到椭圆的几何残差门限
ELLIPSE_MIN_ASPECT = 0.05
ELLIPSE_POINT_RESIDUAL_FLOOR_PX = 1.0
ELLIPSE_POINT_RESIDUAL_SCALE = 0.05


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
    """解析指定类别、置信度达标的螺孔，返回 (holes, point_set)。

    每个孔除中心点外，同时保留检测框左上/右下角坐标 (x1, y1, x2, y2)。
    """
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
        holes.append(
            {
                "conf": box_conf,
                "center_x": center_x,
                "center_y": center_y,
                "x1": float(x1),
                "y1": float(y1),
                "x2": float(x2),
                "y2": float(y2),
            }
        )
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


# ---------------------------------------------------------------------------
# 透视纠正圆心：椭圆拟合 + 等分约束单应精化
#
# 相机不正上方拍摄时，一圈孔的投影像是椭圆而非圆，圆心的像也不等于
# 椭圆中心。两级流水线：
#   1) 对全部孔中心做直接最小二乘椭圆拟合（Fitzgibbon），椭圆中心作为
#      仿射近似下的纠正圆心；
#   2) 利用 N 孔严格等分的先验，非线性拟合「标准等分圆 -> 图像」的单应
#      矩阵 H，纠正圆心 = H 作用于标准圆圆心 (0,0,1)，即 H 的第三列。
#      完全透视模型下精确，无需相机内参。
# 第 2 级失败或残差超过门限（布局非等分）时回退第 1 级结果。
# ---------------------------------------------------------------------------


def _conic_matrix(ellipse_params) -> np.ndarray:
    """由椭圆参数构造齐次二次曲线矩阵 Q（x^T Q x = 0 为椭圆）。"""
    cx, cy, semi_a, semi_b, angle_deg = ellipse_params
    t = np.deg2rad(angle_deg)
    R = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
    M = R @ np.diag([1.0 / semi_a**2, 1.0 / semi_b**2]) @ R.T
    c = np.array([cx, cy])
    lin = -(M @ c)
    const = float(c @ M @ c - 1.0)
    return np.array(
        [
            [M[0, 0], M[0, 1], lin[0]],
            [M[0, 1], M[1, 1], lin[1]],
            [lin[0], lin[1], const],
        ]
    )


def _ellipse_point_residuals(pts: np.ndarray, ellipse_params) -> np.ndarray:
    """各点到拟合椭圆的近似几何距离（一阶展开 |q(x)| / |grad q(x)|）。"""
    Q = _conic_matrix(ellipse_params)
    homog = np.column_stack([pts, np.ones(len(pts))])
    q = np.einsum("ij,jk,ik->i", homog, Q, homog)
    grad = 2.0 * (pts @ Q[:2, :2] + Q[:2, 2])
    grad_norm = np.linalg.norm(grad, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        dist = np.abs(q) / grad_norm
    return np.where(grad_norm > 1e-12, dist, np.inf)


def ellipse_fit_quality(points: List[List[float]]):
    """对孔中心点集做直接最小二乘椭圆拟合。

    返回 ((cx, cy, semi_axis_along_angle, semi_axis_perp, angle_deg), quality)，
    失败时第一项为 None。需要至少 5 个点；含扁平率与几何残差两道防线。

    注意 OpenCV 的 axes 顺序不保证长轴在前：semi_axis_along_angle 始终与
    angle 方向配对（供 H0/二次曲线构造使用），门限判断按长短轴排序。
    """
    quality: Dict[str, Any] = {"valid": False, "point_count": len(points), "reason": ""}
    if len(points) < 5:
        quality["reason"] = "fewer_than_five_points"
        return None, quality
    pts = np.asarray(points, dtype=np.float64)
    if pts.shape != (len(points), 2) or not np.isfinite(pts).all():
        quality["reason"] = "invalid_coordinates"
        return None, quality
    try:
        (cx, cy), (axis_a, axis_b), angle = cv2.fitEllipseDirect(
            pts.astype(np.float32).reshape(-1, 1, 2)
        )
    except Exception:
        quality["reason"] = "ellipse_fit_failed"
        return None, quality
    semi_a, semi_b = float(axis_a) / 2.0, float(axis_b) / 2.0
    params = (float(cx), float(cy), semi_a, semi_b, float(angle))
    if not np.isfinite(params).all() or semi_a <= 0 or semi_b <= 0:
        quality["reason"] = "invalid_ellipse"
        return None, quality
    semi_major, semi_minor = max(semi_a, semi_b), min(semi_a, semi_b)
    if semi_minor / semi_major < ELLIPSE_MIN_ASPECT:
        # 共线/近共线点会拟合出极端扁平的退化椭圆
        quality["reason"] = "degenerate_flat_ellipse"
        return None, quality
    residual = _ellipse_point_residuals(pts, params)
    residual_max = float(np.max(residual))
    quality["ellipse_residual_max_px"] = residual_max
    if not np.isfinite(residual_max) or residual_max > max(
        ELLIPSE_POINT_RESIDUAL_FLOOR_PX, ELLIPSE_POINT_RESIDUAL_SCALE * semi_major
    ):
        quality["reason"] = "excessive_ellipse_residual"
        return None, quality
    quality.update(valid=True)
    return params, quality


def _ellipse_to_unit_circle_h(ellipse_params) -> np.ndarray:
    """构造把单位圆映射到拟合椭圆的仿射矩阵 H0（用于初始化/归一化）。"""
    cx, cy, semi_a, semi_b, angle_deg = ellipse_params
    t = np.deg2rad(angle_deg)
    cos_t, sin_t = np.cos(t), np.sin(t)
    return np.array(
        [
            [semi_a * cos_t, -semi_b * sin_t, cx],
            [semi_a * sin_t, semi_b * cos_t, cy],
            [0.0, 0.0, 1.0],
        ]
    )


def _assign_slots(points, ellipse_params, n_total: int) -> Optional[List[int]]:
    """把检测到的孔分配到等分布局的 slot 序号（与输入点顺序对齐）。

    先用 H0 逆变换把点归一化到单位圆（该空间中角度近似均匀），再按
    角度间隙定位缺孔。支持「无缺孔」与「恰好缺一孔」两种情形；孔数
    不符或无法辨识缺孔位置时返回 None。
    """
    pts = np.asarray(points, dtype=np.float64)
    m = len(pts)
    if m not in (n_total, n_total - 1):
        return None
    try:
        Hinv = np.linalg.inv(_ellipse_to_unit_circle_h(ellipse_params))
    except np.linalg.LinAlgError:
        return None
    homog = np.column_stack([pts, np.ones(m)])
    u = (Hinv @ homog.T).T
    u = u[:, :2] / u[:, 2:3]
    alpha = np.arctan2(u[:, 1], u[:, 0])
    order = np.argsort(alpha)
    alpha_sorted = alpha[order]
    step = 2 * np.pi / n_total
    if m == n_total:
        slots_sorted = list(range(n_total))
    else:
        gaps = np.diff(np.concatenate([alpha_sorted, alpha_sorted[:1] + 2 * np.pi]))
        big = int(np.argmax(gaps))
        if gaps[big] < 1.6 * step:
            return None  # 无明显缺孔间隙，无法确定 slot
        # α 升序下：位置 0..big -> slot 0..big，缺孔 = big+1，其后位置 -> big+2..
        slots_sorted = list(range(0, big + 1)) + list(range(big + 2, n_total))
    slots: List[int] = [0] * m
    for j, idx in enumerate(order):
        slots[idx] = slots_sorted[j]
    return slots


def homography_center_fit(points, slots, n_total: int, ellipse_params):
    """等分约束下拟合「标准等分圆 -> 图像」的单应矩阵，返回纠正圆心。

    拟合参数为 H（9 元素，按范数归一消尺度）与标准圆相位 phi；镜像
    （顺/逆时针）两个方向分别拟合，取残差最小者。标准圆圆心 (0,0,1)
    的像即 H 的第三列。

    返回 (center_xy, residual_rms_px)，失败时 (None, None)。
    """
    pts = np.asarray(points, dtype=np.float64)
    m = len(pts)
    slots_arr = np.asarray(slots, dtype=np.float64)
    H0 = _ellipse_to_unit_circle_h(ellipse_params)
    try:
        H0_inv = np.linalg.inv(H0)
    except np.linalg.LinAlgError:
        return None, None

    best = None
    for mirror in (False, True):
        # 按绕椭圆中心极角排序；镜像方向取反序（单应保持循环序，方向未知）
        alpha_raw = np.arctan2(pts[:, 1] - ellipse_params[1], pts[:, 0] - ellipse_params[0])
        order = np.argsort(alpha_raw if not mirror else -alpha_raw)
        p_sorted = pts[order]
        s_sorted = slots_arr[order]

        homog = np.column_stack([p_sorted, np.ones(m)])
        u = (H0_inv @ homog.T).T
        u = u[:, :2] / u[:, 2:3]
        k0 = s_sorted[0]
        phi0 = float(np.arctan2(u[0, 1], u[0, 0]) - k0 * 2 * np.pi / n_total)

        def unpack(x: np.ndarray):
            H = x[:9].reshape(3, 3)
            H = H / np.linalg.norm(H)
            return H, float(x[9])

        def resid(x: np.ndarray) -> np.ndarray:
            H, phi = unpack(x)
            ang = phi + s_sorted * (2 * np.pi / n_total)
            pk = np.column_stack([np.cos(ang), np.sin(ang), np.ones(m)])
            proj = (H @ pk.T).T
            proj = proj[:, :2] / proj[:, 2:3]
            return (proj - p_sorted).ravel()

        try:
            sol = least_squares(
                resid,
                np.concatenate([H0.ravel(), [phi0]]),
                method="lm",
                xtol=1e-12,
                ftol=1e-12,
                max_nfev=400,
            )
        except Exception:
            continue
        if best is None or sol.cost < best.cost:
            best = sol

    if best is None:
        return None, None
    H, _ = unpack(best.x)
    if abs(H[2, 2]) < 1e-12:
        return None, None
    center_x = float(H[0, 2] / H[2, 2])
    center_y = float(H[1, 2] / H[2, 2])
    if not np.isfinite([center_x, center_y]).all():
        return None, None
    residual_rms = float(np.sqrt(2 * best.cost / (2 * m)))
    return (center_x, center_y), residual_rms


def corrected_circle_center(points: List[List[float]], n_total: int) -> Dict[str, Any]:
    """两级流水线求透视纠正圆心，返回接口所需的结构化数据。

    返回字段：center（纠正圆心 + 实际采用方法）、ellipse（拟合椭圆）、
    correction（单应相对椭圆中心的修正量）、quality（诊断信息）。
    method 取值：homography（单应精化）/ ellipse（仿射级回退）/ none。
    """
    point_count = len(points)
    out: Dict[str, Any] = {
        "center": None,
        "method": "none",
        "ellipse": None,
        "correction": None,
        "quality": {
            "valid": False,
            "point_count": point_count,
            "expected_count": n_total,
            "reason": "fewer_than_five_points",
        },
    }

    ellipse_params, ellipse_quality = ellipse_fit_quality(points)
    if ellipse_params is None:
        out["quality"].update(ellipse_quality)
        return out

    cx, cy, semi_a, semi_b, angle_deg = ellipse_params
    semi_major, semi_minor = max(semi_a, semi_b), min(semi_a, semi_b)
    # 长轴在 angle 的垂直方向时，展示用角度 +90°（对 180° 取模）
    angle_display = (angle_deg + (90.0 if semi_b > semi_a else 0.0)) % 180.0
    out["ellipse"] = {
        "center_x": cx,
        "center_y": cy,
        "semi_major": semi_major,
        "semi_minor": semi_minor,
        "angle_deg": angle_display,
    }
    out["center"] = {"center_x": cx, "center_y": cy, "method": "ellipse"}
    out["method"] = "ellipse"
    quality = out["quality"]
    quality.update(ellipse_quality)
    quality["reason"] = "ellipse_only"

    slots = _assign_slots(points, ellipse_params, n_total)
    if slots is None:
        quality["reason"] = "slot_assignment_failed"
        return out

    gate = max(ELLIPSE_RESIDUAL_FLOOR_PX, ELLIPSE_RESIDUAL_SCALE * semi_major)
    center, residual_rms = homography_center_fit(points, slots, n_total, ellipse_params)
    quality["homography_residual_rms_px"] = residual_rms
    quality["residual_gate_px"] = gate
    if center is None:
        quality["reason"] = "homography_fit_failed"
        return out
    if residual_rms > gate:
        # 残差超标：布局不符合等分假设，保留椭圆中心
        quality["reason"] = "homography_residual_exceeded"
        return out

    out["center"] = {"center_x": center[0], "center_y": center[1], "method": "homography"}
    out["method"] = "homography"
    out["correction"] = {
        "dx": center[0] - cx,
        "dy": center[1] - cy,
    }
    quality.update(valid=True, reason="")
    return out


def build_center_fit_data(
    result: Any, conf: float, n_total: int, noun: str = "螺孔"
) -> Dict[str, Any]:
    """构造「全部螺孔 + 透视纠正圆心」接口返回数据。"""
    if result is None:
        return {
            "msg": "模型未返回检测结果",
            "all_holes": [],
            "center": None,
            "ellipse": None,
            "correction": None,
            "fit_quality": {"valid": False, "point_count": 0, "reason": "no_result"},
        }

    holes, point_set = parse_holes(result, conf)
    if not point_set:
        return {
            "msg": f"未检测到符合置信度的{noun}",
            "all_holes": holes,
            "center": None,
            "ellipse": None,
            "correction": None,
            "fit_quality": {
                "valid": False,
                "point_count": 0,
                "expected_count": n_total,
                "reason": "no_holes",
            },
        }

    fit = corrected_circle_center(point_set, n_total)
    return {
        "msg": f"共检测到 {len(holes)} 个符合阈值{noun}",
        "all_holes": holes,
        "center": fit["center"],
        "ellipse": fit["ellipse"],
        "correction": fit["correction"],
        "fit_quality": fit["quality"],
    }


def build_judge_data(result: Any) -> Dict[str, Any]:
    """构造「拧紧/对准判断」接口返回数据。"""
    if result is None:
        return {"msg": "模型未返回检测结果", "detections": []}

    detections = parse_detections(result)
    return {"msg": f"共检测到 {len(detections)} 个目标", "detections": detections}
