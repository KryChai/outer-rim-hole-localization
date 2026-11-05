"""Adapt the offline outer-circle method to detected boxes and frontend geometry."""
import numpy as np
import cv2

from app.services.circle_refinement_core import expanded_box, fit_roi

ALGORITHM = "polar_dp_soft_l1_circle_v1"
EXPANSION = .05
SIGMA = 1.4


def bbox_dict(box):
    return dict(zip(("x1", "y1", "x2", "y2"), map(float, box)))


def fit_box(image, box):
    """Only pixels in the 1.05x ROI enter the fitter; never resize the image."""
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
    fit = fit_roi(image[y0:yend, x0:xend], box-[x0, y0, x0, y0], clipped, SIGMA)
    return fit, np.array([x0, y0]), clipped


def refine_hole(image, hole, index, check_stability=True):
    box = np.array([hole[k] for k in ("x1", "y1", "x2", "y2")], dtype=float)
    out = {"detection_index": index, "confidence": hole["conf"],
           "detection_center": {"x": hole["center_x"], "y": hole["center_y"]},
           "detection_bbox": bbox_dict(box), "expanded_bbox": None, "roi_bbox": None,
           "status": "failed", "center": None, "circle": None, "visualization": None,
           "fit_quality": {"valid": False, "flags": [], "reason": "",
                           "stability_checked": False}}
    try:
        if not np.isfinite(box).all():
            raise ValueError("invalid_bbox")
        padding = (box[2:]-box[:2])*EXPANSION/2
        out["expanded_bbox"] = bbox_dict(box+np.r_[-padding, padding])
        out["roi_bbox"] = bbox_dict(expanded_box(box, image.shape, EXPANSION))
        fit, offset, clipped = fit_box(image, box)
        cx, cy = fit["circle"][:2]+offset
        radius = float(fit["circle"][2])
        points = fit["points"][fit["inliers"]]+offset
        if not np.isfinite(np.r_[cx, cy, radius, fit["rmse"], points.ravel()]).all():
            raise ValueError("nonfinite_fit")
        cx, cy = float(cx), float(cy)
        flags = []
        if clipped:
            flags.append("image_boundary_truncated")
        if fit["coverage"] < .85 or fit["inlier_fraction"] < .65:
            flags.append("incomplete_edge_support")
        if fit["rmse"] > 2:
            flags.append("noncircular_or_noisy_edge")
        if fit["scale_center_shift"] is None or fit["scale_center_shift"] > 1:
            flags.append("smoothing_sensitive")
        if fit["candidate_center_spread"] > 1:
            flags.append("ambiguous_concentric_edge")
        if fit["ellipse_axis_ratio"] is not None and fit["ellipse_axis_ratio"] < .95:
            flags.append("elliptical_projection")
        relative_rmse = fit["rmse"]/radius
        if relative_rmse > .03:
            flags.append("relative_radial_rmse_high")

        perturbations, changes = [], []
        if check_stability:
            h, w = image.shape[:2]
            for dx, dy in ((-3, 0), (3, 0), (0, -3), (0, 3)):
                shifted = box+[dx, dy, dx, dy]
                shifted[[0, 2]] = np.clip(shifted[[0, 2]], 0, w)
                shifted[[1, 3]] = np.clip(shifted[[1, 3]], 0, h)
                item = {"dx": dx, "dy": dy, "center_shift_px": None, "reason": ""}
                try:
                    other, origin, _ = fit_box(image, shifted)
                    change = float(np.linalg.norm(other["circle"][:2]+origin-[cx, cy]))
                    item["center_shift_px"] = change
                    changes.append(change)
                except (ValueError, RuntimeError, cv2.error, np.linalg.LinAlgError) as exc:
                    item["reason"] = str(exc)
                perturbations.append(item)
            if len(changes) < 4:
                flags.append("bbox_perturbation_fit_failed")
            elif max(changes) > 1:
                flags.append("bbox_sensitive")
        else:
            flags.append("stability_not_checked")

        angles = np.linspace(0, 2*np.pi, 181)
        outline = np.c_[cx+radius*np.cos(angles), cy+radius*np.sin(angles)]
        out.update(status="review" if flags else "ok", center={"x": cx, "y": cy},
                   circle={"center_x": cx, "center_y": cy, "radius": radius,
                           "diameter": 2*radius,
                           "equation": {"standard_form": "(x-h)^2+(y-k)^2=r_squared",
                                        "h": cx, "k": cy, "r_squared": radius**2,
                                        "general_form": "x^2+y^2+D*x+E*y+F=0",
                                        "D": -2*cx, "E": -2*cy,
                                        "F": cx**2+cy**2-radius**2}},
                   visualization={"outline_points_px": outline.tolist(),
                                  "edge_points_px": points.tolist()},
                   fit_quality={"valid": not flags, "flags": flags,
                                "reason": flags[0] if flags else "",
                                "rmse_px": float(fit["rmse"]),
                                "relative_rmse_radius": float(relative_rmse),
                                "angular_coverage": float(fit["coverage"]),
                                "inlier_fraction": float(fit["inlier_fraction"]),
                                "edge_point_count": len(points),
                                "scale_center_shift_px": fit["scale_center_shift"],
                                "candidate_center_spread_px": fit["candidate_center_spread"],
                                "ellipse_axis_ratio": fit["ellipse_axis_ratio"],
                                "ellipse_center_shift_px": fit["ellipse_center_shift"],
                                "candidate_count": fit["candidates"],
                                "stability_checked": check_stability,
                                "bbox_jitter_max_center_shift_px": max(changes) if len(changes) == 4 else None,
                                "perturbations": perturbations})
    except (ValueError, RuntimeError, cv2.error, np.linalg.LinAlgError) as exc:
        out["fit_quality"].update(reason=str(exc), flags=["fit_failed"])
    return out


def build_circle_refinement_data(image, holes, check_stability=True, no_result=False):
    results = [refine_hole(image, hole, i, check_stability) for i, hole in enumerate(holes)]
    return {"algorithm": ALGORITHM, "center_definition": "fitted_outer_circle_center",
            "coordinate_frame": "original_image_pixels", "units": "px",
            "image_size": {"width": image.shape[1], "height": image.shape[0]},
            "bbox_expansion_fraction": EXPANSION, "sigma_px": SIGMA,
            "check_stability": check_stability,
            "reason": "no_result" if no_result else "no_holes" if not holes else "",
            "summary": {"detected": len(results), **{s: sum(r["status"] == s for r in results)
                        for s in ("ok", "review", "failed")}}, "holes": results}
