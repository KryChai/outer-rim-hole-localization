"""把外圈椭圆拟合接到 YOLO 检测框，并生成前端坐标系的完整参数。"""
import numpy as np
import cv2

from app.services.circle_refinement_core import expanded_box
from app.services.ellipse_refinement_core import (
    ellipse_outline, ellipse_params_to_conic, fit_ellipse_roi,
)

ALGORITHM = "polar_radial_algebraic_ellipse_v1"
EXPANSION = .10
SIGMA = 1.4


def bbox_dict(box):
    return dict(zip(("x1", "y1", "x2", "y2"), map(float, box)))


def fit_ellipse_box(image, box):
    """仅 1.05x ROI 内像素参与拟合，图像不缩放。"""
    box = np.asarray(box, dtype=float)
    if box.shape != (4,) or not np.isfinite(box).all() or min(box[2:]-box[:2]) < 16:
        raise ValueError("invalid_or_too_small_bbox")
    h, w = image.shape[:2]
    if box[2] <= 0 or box[3] <= 0 or box[0] >= w or box[1] >= h:
        raise ValueError("bbox_outside_image")
    x0, y0, xend, yend = expanded_box(box, image.shape, EXPANSION)
    if min(xend-x0, yend-y0) < 16:
        raise ValueError("roi_too_small")
    clipped = bool(box[0] < 1 or box[1] < 1 or box[2] > w-1 or box[3] > h-1)
    fit = fit_ellipse_roi(image[y0:yend, x0:xend], box-[x0, y0, x0, y0], clipped, SIGMA)
    return fit, np.array([x0, y0]), clipped


def eccentricity(a, b):
    return float(np.sqrt(max(0., 1.-(b/a)**2)))


def ellipse_parameters(cx, cy, a, b, phi):
    """返回全部可通过拟合参数推导的椭圆几何量，均为原图 px 坐标。"""
    eccentricity_value = eccentricity(a, b)
    linear = float(np.sqrt(max(0., a*a-b*b)))
    cos_phi, sin_phi = np.cos(phi), np.sin(phi)
    foci = [{"x": float(cx+sign*linear*cos_phi), "y": float(cy+sign*linear*sin_phi)}
            for sign in (1, -1)]
    perimeter = float(np.pi*(3*(a+b)-np.sqrt((3*a+b)*(a+3*b)))
                      if min(a, b) > 0 else 0.)
    conic = ellipse_params_to_conic(cx, cy, a, b, phi)
    return {
        "center_x": float(cx), "center_y": float(cy),
        "semi_major_axis_px": float(a), "semi_minor_axis_px": float(b),
        "major_axis_length_px": float(2*a), "minor_axis_length_px": float(2*b),
        "axis_ratio": float(b/a), "eccentricity": eccentricity_value,
        "linear_eccentricity_px": linear,
        "focal_parameter_px": float(b*b/a),
        "flattening": float(1.-b/a),
        "rotation_angle_deg": float(np.rad2deg(phi) % 180.),
        "rotation_angle_rad": float(phi),
        "foci_px": foci,
        "major_axis_endpoints_px": [
            {"x": float(cx+sign*a*cos_phi), "y": float(cy+sign*a*sin_phi)}
            for sign in (1, -1)],
        "minor_axis_endpoints_px": [
            {"x": float(cx-sign*b*sin_phi), "y": float(cy+sign*b*cos_phi)}
            for sign in (1, -1)],
        "area_px2": float(np.pi*a*b),
        "perimeter_px": perimeter,
        "equation": {
            "general_form": "A*x^2+B*x*y+C*y^2+D*x+E*y+F=0",
            "A": float(conic[0]), "B": float(conic[1]), "C": float(conic[2]),
            "D": float(conic[3]), "E": float(conic[4]), "F": float(conic[5]),
            "canonical_form": "(u/a)^2+(v/b)^2=1",
            "standard_form": "(u/a)^2+(v/b)^2=1, "
                             "u=(x-h)*cos(theta)+(y-k)*sin(theta), "
                             "v=-(x-h)*sin(theta)+(y-k)*cos(theta)",
            "h": float(cx), "k": float(cy),
            "a": float(a), "b": float(b),
            "theta_deg": float(np.rad2deg(phi) % 180.),
            "theta_rad": float(phi),
        },
    }


def refine_hole(image, hole, index, check_stability=True):
    box = np.array([hole[k] for k in ("x1", "y1", "x2", "y2")], dtype=float)
    out = {"detection_index": index, "confidence": hole["conf"],
           "detection_center": {"x": hole["center_x"], "y": hole["center_y"]},
           "detection_bbox": bbox_dict(box), "expanded_bbox": None, "roi_bbox": None,
           "status": "failed", "center": None, "ellipse": None, "visualization": None,
           "fit_quality": {"valid": False, "flags": [], "reason": "",
                           "stability_checked": False}}
    try:
        if not np.isfinite(box).all():
            raise ValueError("invalid_bbox")
        padding = (box[2:]-box[:2])*EXPANSION/2
        out["expanded_bbox"] = bbox_dict(box+np.r_[-padding, padding])
        out["roi_bbox"] = bbox_dict(expanded_box(box, image.shape, EXPANSION))
        fit, offset, clipped = fit_ellipse_box(image, box)
        cx, cy, a, b, phi = fit["ellipse"]
        cx, cy = float(cx+offset[0]), float(cy+offset[1])
        points = fit["points"][fit["inliers"]].astype(float)+offset
        if not np.isfinite(np.r_[cx, cy, a, b, phi, fit["rmse"], points.ravel()]).all():
            raise ValueError("nonfinite_fit")

        flags = []
        if clipped:
            flags.append("image_boundary_truncated")
        if fit["coverage"] < .85 or fit["inlier_fraction"] < .65:
            flags.append("incomplete_edge_support")
        if fit["rmse"] > 2:
            flags.append("noisy_edge")
        if b < 3 or a/b > 5:
            flags.append("degenerate_ellipse")
        if (fit["scale_center_shift"] is None or fit["scale_center_shift"] > 1
                or (fit["scale_axis_ratio_change"] is not None
                    and fit["scale_axis_ratio_change"] > .03)):
            flags.append("smoothing_sensitive")
        relative_rmse = fit["rmse"]/a
        if relative_rmse > .03:
            flags.append("relative_rmse_high")

        perturbations, shifts, ratios = [], [], []
        if check_stability:
            h, w = image.shape[:2]
            for dx, dy in ((-3, 0), (3, 0), (0, -3), (0, 3)):
                shifted = box+[dx, dy, dx, dy]
                shifted[[0, 2]] = np.clip(shifted[[0, 2]], 0, w)
                shifted[[1, 3]] = np.clip(shifted[[1, 3]], 0, h)
                item = {"dx": dx, "dy": dy, "center_shift_px": None,
                        "axis_ratio_change": None, "eccentricity_change": None,
                        "reason": ""}
                try:
                    other, origin, _ = fit_ellipse_box(image, shifted)
                    ocx, ocy, oa, ob, _ = other["ellipse"]
                    item["center_shift_px"] = float(np.linalg.norm(
                        other["ellipse"][:2]+origin-[cx, cy]))
                    item["axis_ratio_change"] = float(abs(ob/oa-b/a))
                    item["eccentricity_change"] = float(abs(eccentricity(oa, ob)
                                                              - eccentricity(a, b)))
                    shifts.append(item["center_shift_px"])
                    ratios.append(item["axis_ratio_change"])
                except (ValueError, RuntimeError, cv2.error, np.linalg.LinAlgError) as exc:
                    item["reason"] = str(exc)
                perturbations.append(item)
            if len(shifts) < 4:
                flags.append("bbox_perturbation_fit_failed")
            elif max(shifts) > 1 or max(ratios) > .03:
                flags.append("bbox_sensitive")
        else:
            flags.append("stability_not_checked")

        params = ellipse_parameters(cx, cy, a, b, phi)
        outline = ellipse_outline(cx, cy, a, b, phi)
        out.update(status="review" if flags else "ok", center={"x": cx, "y": cy},
                   ellipse=params,
                   visualization={"outline_points_px": outline.tolist(),
                                  "edge_points_px": points.tolist()},
                   fit_quality={"valid": not flags, "flags": flags,
                                "reason": flags[0] if flags else "",
                                "rmse_px": float(fit["rmse"]),
                                "relative_rmse_major_axis": float(relative_rmse),
                                "angular_coverage": float(fit["coverage"]),
                                "inlier_fraction": float(fit["inlier_fraction"]),
                                "edge_point_count": len(points),
                                "semi_major_axis_px": float(a),
                                "semi_minor_axis_px": float(b),
                                "axis_ratio": float(b/a),
                                "eccentricity": float(params["eccentricity"]),
                                "rotation_angle_deg": float(params["rotation_angle_deg"]),
                                "scale_center_shift_px": fit["scale_center_shift"],
                                "scale_axis_ratio_change": fit["scale_axis_ratio_change"],
                                "scale_eccentricity_change": fit["scale_eccentricity_change"],
                                "candidate_count": 1,
                                "stability_checked": check_stability,
                                "bbox_jitter_max_center_shift_px": max(shifts) if len(shifts) == 4 else None,
                                "bbox_jitter_max_axis_ratio_change": max(ratios) if len(ratios) == 4 else None,
                                "perturbations": perturbations})
    except (ValueError, RuntimeError, cv2.error, np.linalg.LinAlgError) as exc:
        out["fit_quality"].update(reason=str(exc), flags=["fit_failed"])
    return out


def build_ellipse_refinement_data(image, holes, check_stability=True, no_result=False):
    results = [refine_hole(image, hole, i, check_stability) for i, hole in enumerate(holes)]
    return {"algorithm": ALGORITHM, "center_definition": "fitted_outer_ellipse_center",
            "coordinate_frame": "original_image_pixels", "units": "px",
            "image_size": {"width": image.shape[1], "height": image.shape[0]},
            "bbox_expansion_fraction": EXPANSION, "sigma_px": SIGMA,
            "check_stability": check_stability,
            "fit_parameters": {"radial_samples": 720, "radial_step_px": .5,
                               "initial_band_fraction": .14,
                               "refine_band_fraction": .03,
                               "final_band_fraction": .02,
                               "inlier_mad_scale": 2.7,
                               "fit_method": "fitzgibbon_direct_least_squares"},
            "reason": "no_result" if no_result else "no_holes" if not holes else "",
            "summary": {"detected": len(results), **{s: sum(r["status"] == s for r in results)
                        for s in ("ok", "review", "failed")}}, "holes": results}
