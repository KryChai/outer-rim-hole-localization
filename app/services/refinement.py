"""Single-hole, calibrated coaxial-circle refinement. No metric size is assumed.

The two physical boundaries must be parallel, coaxial circles at different depths.
The larger image contour is the configured reference (outer/front mouth). A dark
shadow or a thread is not a second circular boundary. Invalid/ambiguous fits never
fall back to a box center. All public coordinates are in the original image.
"""
from dataclasses import dataclass
from hashlib import sha256
import json
import math
import os
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares
from app.services.caliper import BOUNDARY_METHOD, BOUNDARY_REVISION, detect_boundaries


ALGORITHM = "calibrated_coaxial_v1"


@dataclass(frozen=True)
class Calibration:
    matrix: np.ndarray
    distortion: np.ndarray
    image_size: tuple
    identifier: str

    @classmethod
    def from_dict(cls, data):
        k = np.asarray(data["camera_matrix"], dtype=float)
        d = np.asarray(data["dist_coeffs"], dtype=float).reshape(-1)
        size = tuple(data["image_size"])
        if (k.shape != (3, 3) or not np.isfinite(k).all()
                or k[0, 0] <= 0 or k[1, 1] <= 0
                or not np.allclose(k[2], [0, 0, 1])
                or abs(k[0, 1]) > 1e-9 or abs(k[1, 0]) > 1e-9):
            raise ValueError("camera_matrix must be a finite OpenCV pinhole matrix")
        if np.linalg.cond(k) > 1e8:
            raise ValueError("camera_matrix is ill-conditioned")
        if len(d) not in (4, 5, 8, 12, 14) or not np.isfinite(d).all():
            raise ValueError("dist_coeffs must contain 4/5/8/12/14 finite values")
        if len(size) != 2 or any(isinstance(v, bool) or not isinstance(v, int) or v <= 0 for v in size):
            raise ValueError("image_size must be [width, height] positive integers")
        identity = sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()[:16]
        return cls(k, d, size, identity)


def load_calibration(path=None):
    path = path or os.environ.get("CAMERA_CALIBRATION_FILE")
    if not path:
        raise ValueError("CAMERA_CALIBRATION_FILE is required for coaxial refinement")
    try:
        return Calibration.from_dict(json.loads(Path(path).read_text(encoding="utf-8-sig")))
    except (OSError, KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid camera calibration: {exc}") from exc


def conic(ellipse):
    (cx, cy), (diam_a, diam_b), degrees = ellipse
    angle = math.radians(degrees)
    rotation = np.array([[math.cos(angle), -math.sin(angle)],
                         [math.sin(angle), math.cos(angle)]])
    matrix = rotation @ np.diag([4 / diam_a**2, 4 / diam_b**2]) @ rotation.T
    center = np.array([cx, cy])
    q = np.empty((3, 3))
    q[:2, :2] = matrix
    q[:2, 2] = q[2, :2] = -matrix @ center
    q[2, 2] = center @ matrix @ center - 1
    return q


def geometric_residual(points, q):
    homog = np.column_stack([points, np.ones(len(points))])
    values = np.einsum("ij,jk,ik->i", homog, q, homog)
    gradient = 2 * (points @ q[:2, :2] + q[:2, 2])
    return np.abs(values) / np.maximum(np.linalg.norm(gradient, axis=1), 1e-12)


def ellipse_points(ellipse, count=96):
    center, diameters, degrees = ellipse
    angles = np.linspace(0, 2 * np.pi, count, endpoint=False)
    a = math.radians(degrees)
    rotation = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
    return np.column_stack([np.cos(angles), np.sin(angles)]) @ np.diag(np.array(diameters)/2) @ rotation.T + center


def distort_points(points, calibration):
    """Undistorted pixel coordinates -> original distorted pixel coordinates."""
    rays = np.column_stack([points, np.ones(len(points))]) @ np.linalg.inv(calibration.matrix).T
    projected, _ = cv2.projectPoints(rays, np.zeros(3), np.zeros(3),
                                    calibration.matrix, calibration.distortion)
    return projected.reshape(-1, 2)


def normal_candidates(q, k):
    cone = k.T @ q @ k
    values, vectors = np.linalg.eigh(cone)
    if not values[0] < 0 < values[1] <= values[2]:
        raise ValueError("invalid elliptic cone")
    low, mid, high = values
    a = math.sqrt(max(0, (high - mid) / (high - low)))
    b = math.sqrt(max(0, (mid - low) / (high - low)))
    normals = []
    for sign in (-1, 1):
        n = a * vectors[:, 2] + sign * b * vectors[:, 0]
        n *= 1 if n[2] >= 0 else -1
        normals.append(n / np.linalg.norm(n))
    return normals


def recover_coaxial(outer_q, inner_q, calibration):
    """Recover projected centers from a common normal; absolute scale is absent."""
    k = calibration.matrix
    cones = [k.T @ q @ k for q in (outer_q, inner_q)]
    candidates = []
    for first in normal_candidates(outer_q, k):
        for second in normal_candidates(inner_q, k):
            if float(first @ second) < math.cos(math.radians(6)):
                continue
            start = first + second
            start /= np.linalg.norm(start)

            def circularity(x):
                n = np.array([x[0], x[1], 1.0])
                n /= np.linalg.norm(n)
                a = np.cross(n, [0, 1, 0])
                a /= np.linalg.norm(a)
                basis = np.column_stack([a, np.cross(n, a)])
                residual = []
                for cone in cones:
                    m = basis.T @ cone @ basis
                    scale = max(abs(np.trace(m)), 1e-15)
                    residual.extend([(m[0, 0] - m[1, 1])/scale, 2*m[0, 1]/scale])
                return residual

            if start[2] < .15:
                continue  # grazing views are unsuitable for this near-field mode
            solution = least_squares(circularity, start[:2]/start[2], method="trf",
                                     max_nfev=60, bounds=(-6, 6))
            if not solution.success or np.linalg.norm(solution.fun) > .035:
                continue
            n = np.r_[solution.x, 1.0]
            n /= np.linalg.norm(n)
            # Pole of the support plane's vanishing line gives the real circle
            # center's image, not the fitted ellipse center. Scale cancels.
            centers = []
            rays = []
            for cone in cones:
                ray = np.linalg.solve(cone, n)
                ray /= float(ray @ n)
                if not np.isfinite(ray).all() or ray[2] <= 0:
                    break
                rays.append(ray)
                pixel = k @ ray
                centers.append(pixel[:2]/pixel[2])
            if len(rays) != 2:
                continue
            transverse = [r - n for r in rays]  # both normalized to n.C = 1
            norms = [np.linalg.norm(t) for t in transverse]
            axis_error = 0.0
            if min(norms) > .002:
                cosine = float(transverse[0] @ transverse[1] / np.prod(norms))
                axis_error = math.acos(np.clip(cosine, -1, 1))
                if axis_error > math.radians(4):
                    continue
                depth_ratio = float(transverse[0] @ transverse[1] / norms[1]**2)
                if depth_ratio < .995:
                    continue  # configured outer reference must be the front rim
            score = float(np.linalg.norm(solution.fun) + axis_error)
            candidates.append((score, centers, n))
    if not candidates:
        raise ValueError("coaxial_geometry_inconsistent")
    candidates.sort(key=lambda c: c[0])
    best = candidates[0]
    for candidate in candidates[1:]:
        if np.linalg.norm(candidate[1][0]-best[1][0]) > .75:
            raise ValueError("ambiguous_pose")
    return best[1], best[2], best[0]


def _subpixel_edges(gray, points, ellipse):
    """Quadratic peak interpolation of the normal intensity gradient (not corners)."""
    q = conic(ellipse)
    normals = points @ q[:2, :2] + q[:2, 2]
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-9)
    samples = points[:, None, :] + normals[:, None, :] * np.arange(-3, 4)[None, :, None]
    intensity = cv2.remap(gray, samples[:, :, 0].astype(np.float32),
                          samples[:, :, 1].astype(np.float32), cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REFLECT101).astype(float)
    gradient = (intensity[:, 2:] - intensity[:, :-2]) / 2
    peak = np.argmax(np.abs(gradient), axis=1)
    selected = (peak > 0) & (peak < 4)
    indices = np.where(selected)[0]
    if len(indices) < 24:
        return points
    p = peak[indices]
    scores = np.abs(gradient[indices])
    left, middle, right = scores[np.arange(len(p)), p-1], scores[np.arange(len(p)), p], scores[np.arange(len(p)), p+1]
    denominator = left - 2*middle + right
    delta = np.clip(.5*(left-right)/np.minimum(denominator, -1e-6), -.5, .5)
    offset = p - 2 + delta
    return points[indices] + normals[indices]*offset[:, None]


def extract_ellipses_legacy(image, bbox, calibration):
    height, width = image.shape[:2]
    x1, y1, x2, y2 = map(float, bbox)
    if not np.isfinite(bbox).all() or x2 <= x1 or y2 <= y1:
        raise ValueError("invalid_bbox")
    if x1 <= 1 or y1 <= 1 or x2 >= width-1 or y2 >= height-1:
        raise ValueError("hole_at_image_border")
    padding = .15*max(x2-x1, y2-y1)
    left, top = max(0, int(x1-padding)), max(0, int(y1-padding))
    right, bottom = min(width, int(x2+padding+1)), min(height, int(y2+padding+1))
    roi = image[top:bottom, left:right]
    if min(roi.shape[:2]) < 16:
        raise ValueError("roi_too_small")
    gray = cv2.GaussianBlur(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY), (5, 5), .8)
    gradient = cv2.magnitude(cv2.Sobel(gray, cv2.CV_32F, 1, 0), cv2.Sobel(gray, cv2.CV_32F, 0, 1))
    high = max(25, float(np.percentile(gradient, 85)))
    edges = cv2.Canny(gray, .4*high, high)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    candidates = []
    for contour in contours:
        raw = contour.reshape(-1, 2).astype(float)
        if len(raw) < 30:
            continue
        raw = raw[::max(1, len(raw)//240)]
        try:
            initial = cv2.fitEllipse(raw.astype(np.float32))
            if min(initial[1]) <= 0:
                continue
            local = _subpixel_edges(gray, raw, initial)
        except (ValueError, cv2.error, FloatingPointError):
            continue
        raw = local + [left, top]
        pts = cv2.undistortPoints(raw.reshape(-1, 1, 2), calibration.matrix,
                                  calibration.distortion, P=calibration.matrix).reshape(-1, 2)
        try:
            ellipse = cv2.fitEllipseDirect(pts.astype(np.float32))
            for _ in range(2):
                residual = geometric_residual(pts, conic(ellipse))
                keep = residual <= max(.7, 2.5*float(np.median(residual)))
                if keep.sum() < 24:
                    break
                pts, raw = pts[keep], raw[keep]
                ellipse = cv2.fitEllipseDirect(pts.astype(np.float32))
            center, axes, angle = ellipse
            if not np.isfinite([*center, *axes, angle]).all() or min(axes) < 8:
                continue
            if min(axes)/max(axes) < .25 or max(axes) > 1.5*max(x2-x1, y2-y1):
                continue
            q = conic(ellipse)
            rms = float(np.sqrt(np.mean(geometric_residual(pts, q)**2)))
            if rms > 1.2:
                continue
            angle_r = math.radians(angle)
            rotation = np.array([[math.cos(angle_r), -math.sin(angle_r)],
                                 [math.sin(angle_r), math.cos(angle_r)]])
            unit = (pts-center) @ rotation / (np.array(axes)/2)
            angles = np.arctan2(unit[:, 1], unit[:, 0])
            bins = np.clip(np.floor((angles+np.pi)/(2*np.pi)*36).astype(int), 0, 35)
            coverage = len(np.unique(bins)) / 36
            if coverage < .75:
                continue
            if any(np.linalg.norm(np.array(center)-c["ellipse"][0]) < 1.5
                   and np.max(np.abs(np.sort(axes)-np.sort(c["ellipse"][1]))) < 3
                   for c in candidates):
                continue
            candidates.append({"ellipse": ellipse, "q": q, "rms": rms,
                               "coverage": coverage, "edges": raw})
        except (ValueError, cv2.error, np.linalg.LinAlgError):
            continue
    return sorted(candidates, key=lambda c: np.prod(c["ellipse"][1]), reverse=True)[:6]


def extract_ellipses(image, bbox, calibration):
    return detect_boundaries(image, bbox, calibration)[0]


def refine_hole(image, hole, calibration):
    out = {**hole, "detection_center": {"center_x": hole["center_x"], "center_y": hole["center_y"]},
           "refined_center": None, "fit_quality": {"valid": False, "reason": "no_pair"},
           "geometry": None, "boundary_diagnostics": None}
    try:
        if image.shape[1::-1] != calibration.image_size:
            raise ValueError("calibration_resolution_mismatch")
        bbox = [hole[k] for k in ("x1", "y1", "x2", "y2")]
        candidates, diagnostics = detect_boundaries(image, bbox, calibration)
        out["boundary_diagnostics"] = diagnostics
        diagnostics['pair_checks'] = []
        if candidates:
            raw_outline = distort_points(ellipse_points(candidates[0]["ellipse"]), calibration)
            span_ratio = np.ptp(raw_outline, axis=0)/[bbox[2]-bbox[0], bbox[3]-bbox[1]]
            diagnostics['outer_bbox_span_ratio'] = span_ratio.tolist()
            if np.any(span_ratio < .78) or np.any(span_ratio > 1.25):
                raise ValueError('outer_reference_not_observed')
        fits = []
        # Never silently change the reference mouth to a smaller/deeper ring
        # merely because that pair happens to have a lower residual.
        for i, outer in enumerate(candidates[:1]):
            for inner_index, inner in enumerate(candidates[i+1:], start=i+1):
                check = {'inner_candidate': inner_index}
                diagnostics['pair_checks'].append(check)
                if max(inner["ellipse"][1]) > .9*max(outer["ellipse"][1]):
                    check['reason'] = 'boundary_sizes_not_separated'
                    continue
                inner_points = ellipse_points(inner["ellipse"])
                homog = np.column_stack([inner_points, np.ones(len(inner_points))])
                if np.mean(np.einsum('ij,jk,ik->i', homog, outer["q"], homog) <= .1) < .9:
                    check['reason'] = 'inner_not_nested'
                    continue
                try:
                    centers, axis, score = recover_coaxial(outer["q"], inner["q"], calibration)
                except ValueError as exc:
                    check['reason'] = str(exc)
                    continue
                check.update(reason='geometry_pass', geometry_score=float(score))
                fits.append((score + .01*(outer["rms"]+inner["rms"]), centers, axis, outer, inner))
        if not fits:
            raise ValueError("no_reliable_coaxial_pair")
        fits.sort(key=lambda f: f[0])
        best = fits[0]
        if any(np.linalg.norm(f[1][0]-best[1][0]) > .75 for f in fits[1:]):
            raise ValueError("ambiguous_boundary_pair")
        _, centers, axis, outer, inner = best
        original_centers = distort_points(np.asarray(centers), calibration)
        center = original_centers[0]
        if not (hole["x1"] <= center[0] <= hole["x2"] and hole["y1"] <= center[1] <= hole["y2"]):
            raise ValueError("center_outside_detection_box")
        out["refined_center"] = {"center_x": float(center[0]), "center_y": float(center[1]),
                                 "method": ALGORITHM, "reference": "outer_front"}
        out["fit_quality"] = {"valid": True, "reason": "", "residual_rms_px": max(outer["rms"], inner["rms"]),
                              "angular_coverage_deg": min(outer["coverage"], inner["coverage"])*360,
                              "geometry_residual": best[0], "metric_scale_known": False,
                              "boundary_method": BOUNDARY_METHOD,
                              "caliper_support_fraction": min(outer['support_fraction'], inner['support_fraction'])}
        out["geometry"] = {"axis_camera": axis.tolist(), "inner_center_px": original_centers[1].tolist(),
                           "outlines_px": [distort_points(ellipse_points(c["ellipse"]), calibration).tolist()
                                           for c in (outer, inner)],
                           "edge_points_px": [c["edges"].tolist() for c in (outer, inner)]}
    except (ValueError, KeyError, cv2.error, np.linalg.LinAlgError) as exc:
        out["fit_quality"]["reason"] = str(exc)
    return out


def build_refinement_data(image, holes, calibration, target, max_distance=80):
    """Select by the coarse observation, never by fitting success or confidence.

    Reject an ambiguous nearest box instead of jumping to a different good fit.
    target and gate are in original pixels. No cross-request state is stored.
    """
    distances = [math.hypot(h["center_x"]-target[0], h["center_y"]-target[1]) for h in holes]
    order = np.argsort(distances)
    selected = int(order[0]) if len(order) and distances[order[0]] <= max_distance else None
    reason = "target_not_found" if selected is None else ""
    if len(order) > 1 and distances[order[1]] - distances[order[0]] < 5:
        selected, reason = None, "ambiguous_target"
    results = [refine_hole(image, h, calibration) for h in holes]
    target_hole = results[selected] if selected is not None else None
    return {"algorithm": ALGORITHM, "boundary_method": BOUNDARY_METHOD, "boundary_revision": BOUNDARY_REVISION, "coordinate_frame": "original_image_pixels",
            "calibration_id": calibration.identifier, "image_size": list(calibration.image_size),
            "metric_scale_known": False, "holes": results, "target_hole": target_hole,
            "reason": reason}
