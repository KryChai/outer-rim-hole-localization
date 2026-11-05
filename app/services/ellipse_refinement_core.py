"""极坐标径向梯度 + 直接最小二乘的螺孔外圈椭圆拟合。

初值来自外圈圆的极坐标动态规划结果，随后沿椭圆轮廓做亚像素径向采样，
用 Fitzgibbon 直接最小二乘法迭代拟合椭圆，并以 Sampson 距离剔除离群点。
"""
import cv2
import numpy as np

from app.services.circle_refinement_core import make_gradients, remap, trace_outer_rim


def ellipse_radial_radius(theta, a, b, phi):
    """椭圆中心到轮廓的极径 r(theta)，theta 为原图坐标系方向角。"""
    psi = theta-phi
    return a*b/np.sqrt((b*np.cos(psi))**2+(a*np.sin(psi))**2)


def ellipse_outline(cx, cy, a, b, phi, count=181):
    theta = np.linspace(0, 2*np.pi, count)
    rr = ellipse_radial_radius(theta, a, b, phi)
    return np.column_stack((cx+rr*np.cos(theta), cy+rr*np.sin(theta)))


def ellipse_params_to_conic(cx, cy, a, b, phi):
    """(x,y) 原图坐标下的圆锥曲线系数：A x^2+B xy+C y^2+D x+E y+F=0。"""
    c, s = np.cos(phi), np.sin(phi)
    A = b*b*c*c+a*a*s*s
    B = 2*(b*b-a*a)*s*c
    C = b*b*s*s+a*a*c*c
    D = -2*A*cx-B*cy
    E = -B*cx-2*C*cy
    F = A*cx*cx+B*cx*cy+C*cy*cy-a*a*b*b
    return np.array([A, B, C, D, E, F], dtype=float)


def ellipse_sampson_distance(conic, points):
    """用 Sampson 距离近似点到椭圆的几何距离，单位 px。"""
    A, B, C, D, E, F = conic
    x, y = points[:, 0], points[:, 1]
    value = A*x*x+B*x*y+C*y*y+D*x+E*y+F
    gx = 2*A*x+B*y+D
    gy = B*x+2*C*y+E
    grad = np.hypot(gx, gy)
    return np.abs(value)/np.maximum(grad, 1e-9)


def _conic_to_params(conic, mean, scale):
    """归一化坐标下的圆锥曲线系数 -> 原图 (cx,cy,a,b,phi)，a>=b。"""
    A, B, C, D, E, F = conic
    matrix = np.array([[2*A, B], [B, 2*C]], dtype=float)
    if abs(np.linalg.det(matrix)) < 1e-12:
        raise ValueError("singular_conic")
    cx, cy = np.linalg.solve(matrix, [-D, -E])
    quadratic = A*cx*cx+B*cx*cy+C*cy*cy
    constant = F-quadratic
    eigen_values, eigen_vectors = np.linalg.eigh(np.array([[A, B/2], [B/2, C]]))
    axes_sq = -constant/eigen_values
    if not np.isfinite(axes_sq).all() or (axes_sq <= 0).any():
        raise ValueError("conic_is_not_ellipse")
    semi = np.sqrt(axes_sq)
    major = int(np.argmax(semi))
    a, b = semi[major], semi[1-major]
    direction = eigen_vectors[:, major]
    phi = float(np.arctan2(direction[1], direction[0])) % np.pi
    cx, cy = cx*scale+mean[0], cy*scale+mean[1]
    a, b = a*scale, b*scale
    if not np.isfinite([cx, cy, a, b, phi]).all() or a <= 0 or b <= 0:
        raise ValueError("degenerate_ellipse")
    return np.array([cx, cy, a, b, phi], dtype=float)


def fit_ellipse_points(points):
    """Fitzgibbon 直接最小二乘椭圆拟合；点坐标先中心化/尺度归一化。"""
    pts = np.asarray(points, dtype=float)
    if len(pts) < 5 or not np.isfinite(pts).all():
        raise ValueError("ellipse_fit_needs_5_finite_points")
    mean = pts.mean(axis=0)
    scale = float(np.abs(pts-mean).max())
    if scale < 1e-6:
        raise ValueError("ellipse_fit_degenerate_points")
    x = (pts[:, 0]-mean[0])/scale
    y = (pts[:, 1]-mean[1])/scale
    d1 = np.column_stack((x*x, x*y, y*y))
    d2 = np.column_stack((x, y, np.ones_like(x)))
    s1, s2, s3 = d1.T@d1, d1.T@d2, d2.T@d2
    try:
        t = -np.linalg.solve(s3, s2.T)
    except np.linalg.LinAlgError as exc:
        raise ValueError("ellipse_fit_singular_system") from exc
    m = s1+s2@t
    # Halir-Flusser 约束矩阵形式，避免 6x6 广义特征值问题。
    m = np.array([m[2]/2, -m[1], m[0]/2])
    values, vectors = np.linalg.eig(m)
    if np.iscomplexobj(values) or np.iscomplexobj(vectors):
        if max(np.max(np.abs(np.imag(values))), np.max(np.abs(np.imag(vectors)))) > 1e-9:
            raise ValueError("ellipse_fit_complex_eigen_solution")
        values, vectors = np.real(values), np.real(vectors)
    if not np.isfinite(values).all() or not np.isfinite(vectors).all():
        raise ValueError("ellipse_fit_eigen_failed")
    valid = 4*vectors[0]*vectors[2]-vectors[1]**2 > 0
    if not valid.any():
        raise ValueError("ellipse_fit_no_ellipse_eigenvector")
    a1 = vectors[:, int(np.argmax(np.where(valid, 1, -1)))]
    conic = np.r_[a1, t@a1].astype(float)
    return _conic_to_params(conic, mean, scale)


def ellipse_radial_points(gx, gy, ellipse, band, angles=720):
    """沿椭圆轮廓每个方向取一个径向梯度峰，做抛物线亚像素插值。"""
    theta = np.arange(angles)*2*np.pi/angles
    co, si = np.cos(theta)[:, None], np.sin(theta)[:, None]
    offsets = np.arange(-band, band+.25, .5)
    r0 = ellipse_radial_radius(theta, ellipse[2], ellipse[3], ellipse[4])
    rr = r0[:, None]+offsets[None, :]
    xx = ellipse[0]+co*rr
    yy = ellipse[1]+si*rr
    vx, vy = remap(gx, xx, yy), remap(gy, xx, yy)
    radial = np.abs(vx*co+vy*si)
    strength = radial*radial/(np.hypot(vx, vy)+1e-5)
    valid = (xx >= 3) & (yy >= 3) & (xx < gx.shape[1]-3) & (yy < gx.shape[0]-3)
    strength[~valid] = 0
    score = strength*np.exp(-.5*(offsets/(band*.65))**2)[None, :]
    peak = np.argmax(score, axis=1)
    ii = np.arange(angles)
    lo, hi = np.maximum(peak-1, 0), np.minimum(peak+1, len(offsets)-1)
    a, b, c = strength[ii, lo], strength[ii, peak], strength[ii, hi]
    denom = a-2*b+c
    delta = np.clip(.5*(a-c)/np.where(np.abs(denom) > 1e-6, denom, -1e-6), -1, 1)
    radii = r0+offsets[peak]+.5*delta
    points = np.column_stack((ellipse[0]+np.cos(theta)*radii,
                              ellipse[1]+np.sin(theta)*radii))
    good = (peak > 0) & (peak < len(offsets)-1) & (b > 1.0)
    return points, b, good, theta


def _valid_ellipse(ellipse, center, d):
    if not np.isfinite(ellipse).all():
        return False
    cx, cy, a, b, _ = ellipse
    if a <= 0 or b <= 0 or a < b:
        return False
    if np.linalg.norm(ellipse[:2]-center) > .22*d:
        return False
    if not .36*d < a < .58*d or b < 3:
        return False
    return True


def _angle_change(old, new):
    delta = (new-old+np.pi/2) % np.pi-np.pi/2
    return abs(float(delta))


def _filter_strength(good, strength):
    valid = good.copy()
    if valid.any():
        valid &= strength >= max(1.0, np.percentile(strength[valid], 20)*.45)
    return valid


def refine_ellipse_from(gx, gy, initial, d, center, iterations=8):
    """以 initial 为起点迭代拟合；失败返回 None。"""
    ellipse = np.asarray(initial, dtype=float)
    for it in range(iterations):
        band = max(4.0, d*(.14 if it == 0 else .03 if it == 1 else .02))
        pts, strength, good, _ = ellipse_radial_points(gx, gy, ellipse, band)
        valid = _filter_strength(good, strength)
        if valid.sum() < 5:
            return None
        try:
            updated = fit_ellipse_points(pts[valid])
        except (ValueError, np.linalg.LinAlgError):
            return None
        if not _valid_ellipse(updated, center, d):
            return None
        change = max(float(np.linalg.norm(updated[:2]-ellipse[:2])),
                     abs(float(updated[2]-ellipse[2])), abs(float(updated[3]-ellipse[3])),
                     _angle_change(ellipse[4], updated[4])*min(updated[2], updated[3]))
        ellipse = updated
        if it > 0 and change < .02:
            break
    pts, strength, good, theta = ellipse_radial_points(gx, gy, ellipse, max(4.0, d*.02))
    residual = ellipse_sampson_distance(ellipse_params_to_conic(*ellipse), pts)
    valid = _filter_strength(good, strength)
    if not valid.any():
        return None
    median = float(np.median(residual[valid]))
    mad = 1.4826*float(np.median(np.abs(residual[valid]-median)))
    threshold = min(max(1.6, 2.7*mad), max(3., .018*d))
    inliers = valid & (np.abs(residual) < threshold)
    if inliers.sum() < 5:
        return None
    try:
        ellipse = fit_ellipse_points(pts[inliers])
    except (ValueError, np.linalg.LinAlgError):
        return None
    residual = ellipse_sampson_distance(ellipse_params_to_conic(*ellipse), pts)
    inliers &= np.abs(residual) < threshold
    if inliers.sum() < 5:
        return None
    rmse = float(np.sqrt(np.mean(residual[inliers]**2)))
    sectors = np.floor((theta[inliers] % (2*np.pi))/(2*np.pi)*72).astype(int)
    coverage = len(np.unique(sectors))/72.
    edge_strength = float(np.median(strength[inliers]))
    return {"ellipse": ellipse, "points": pts, "inliers": inliers, "rmse": rmse,
            "coverage": coverage, "edge_strength": edge_strength,
            "inlier_fraction": float(inliers.mean()), "threshold": threshold}


def fit_ellipse_roi(roi, box_local, image_clipped=False, sigma=1.4):
    gray, gx, gy = make_gradients(roi, sigma)
    x1, y1, x2, y2 = box_local
    bw, bh = x2-x1, y2-y1
    d = max(bw, bh)
    center = np.array([(x1+x2)/2, (y1+y2)/2])
    if image_clipped and bw < .85*bh:
        center[0] = x2-d/2 if x1 < 1 else x1+d/2
    if image_clipped and bh < .85*bw:
        center[1] = y2-d/2 if y1 < 1 else y1+d/2
    seed = trace_outer_rim(gx, gy, center, d, outer_bias=4)
    initial = (np.r_[seed[:2], seed[2], seed[2], 0.] if seed is not None
               else np.r_[center, d*.49, d*.49, 0.])
    best = refine_ellipse_from(gx, gy, initial, d, center)
    if best is None:
        raise RuntimeError("No supported outer-rim ellipse found")
    # 平滑尺度敏感性检查：只报告差异，不作为置信区间。
    _, gx2, gy2 = make_gradients(roi, sigma*1.5)
    other = refine_ellipse_from(gx2, gy2, best["ellipse"], d, center)
    if other is not None:
        best["scale_center_shift"] = float(np.linalg.norm(other["ellipse"][:2]-best["ellipse"][:2]))
        best["scale_axis_ratio_change"] = float(abs(other["ellipse"][3]/other["ellipse"][2]
                                                    - best["ellipse"][3]/best["ellipse"][2]))
        best["scale_eccentricity_change"] = float(abs(_eccentricity(*other["ellipse"][2:])
                                                       - _eccentricity(*best["ellipse"][2:])))
    else:
        best["scale_center_shift"] = None
        best["scale_axis_ratio_change"] = None
        best["scale_eccentricity_change"] = None
    return best


def _eccentricity(a, b, _phi):
    return float(np.sqrt(max(0., 1.-(b/a)**2)))
