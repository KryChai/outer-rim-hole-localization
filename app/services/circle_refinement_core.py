"""Polar DP, radial subpixel edges and Soft-L1 geometric circle fitting.

Ported from the local 圆拟合精定位 project's fit_hole_circles_fast.py.
The vectorized DP and all fitting parameters are preserved. YOLO boxes
provide only the ROI and initial scale; no weights or file I/O live here.
"""
import cv2
import numpy as np
from scipy.optimize import least_squares


def expanded_box(box, shape, expand):
    """expand=.05 means total width/height *1.05, i.e. 2.5% per side."""
    x1, y1, x2, y2 = box
    dx, dy = (x2-x1)*expand/2, (y2-y1)*expand/2
    h, w = shape[:2]
    return [max(0, int(np.floor(x1-dx))), max(0, int(np.floor(y1-dy))),
            min(w, int(np.ceil(x2+dx))), min(h, int(np.ceil(y2+dy)))]


def remap(im, x, y):
    return cv2.remap(im, x.astype(np.float32), y.astype(np.float32),
                     cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def make_gradients(roi, sigma):
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY).astype(np.float32)
    smooth = cv2.GaussianBlur(gray, (0, 0), sigma)
    gx = cv2.Sobel(smooth, cv2.CV_32F, 1, 0, ksize=3, scale=1/8)
    gy = cv2.Sobel(smooth, cv2.CV_32F, 0, 1, ksize=3, scale=1/8)
    return gray, gx, gy


def radial_points(gx, gy, circle, band, angles=720):
    """One radially aligned edge per ray; parabolic subpixel peak interpolation."""
    theta = np.arange(angles)*2*np.pi/angles
    co, si = np.cos(theta)[:, None], np.sin(theta)[:, None]
    offsets = np.arange(-band, band+0.25, .5)
    rr = circle[2]+offsets[None, :]
    xx = circle[0]+co*rr
    yy = circle[1]+si*rr
    vx, vy = remap(gx, xx, yy), remap(gy, xx, yy)
    # Squared alignment suppresses machining marks tangent to the desired rim.
    radial = np.abs(vx*co+vy*si)
    mag = np.hypot(vx, vy)
    strength = radial*radial/(mag+1e-5)
    valid = (xx >= 3) & (yy >= 3) & (xx < gx.shape[1]-3) & (yy < gx.shape[0]-3)
    strength[~valid] = 0
    # A soft proximity prior limits jumps between adjacent concentric boundaries.
    score = strength*np.exp(-.5*(offsets/(band*.65))**2)[None, :]
    peak = np.argmax(score, axis=1)
    ii = np.arange(angles)
    # Interpolate the unpenalized gradient, so the prior does not shift the edge.
    lo, hi = np.maximum(peak-1, 0), np.minimum(peak+1, len(offsets)-1)
    a, b, c = strength[ii, lo], strength[ii, peak], strength[ii, hi]
    denom = a-2*b+c
    delta = np.clip(.5*(a-c)/np.where(np.abs(denom)>1e-6, denom, -1e-6), -1, 1)
    radii = circle[2]+offsets[peak]+.5*delta
    points = np.column_stack((circle[0]+np.cos(theta)*radii,
                              circle[1]+np.sin(theta)*radii))
    good = (peak > 0) & (peak < len(offsets)-1) & (b > 1.0)
    return points, b, good, theta


def geometric_fit(points, initial, f_scale=1.5):
    def residual(p):
        return np.hypot(points[:, 0]-p[0], points[:, 1]-p[1])-p[2]
    result = least_squares(residual, initial, loss='soft_l1', f_scale=f_scale,
                           max_nfev=100, xtol=1e-9, ftol=1e-9, gtol=1e-9)
    if not result.success or not np.isfinite(result.x).all() or result.x[2] <= 0:
        raise ValueError('circle_fit_not_converged')
    return result.x


def trace_outer_rim(gx, gy, center, d, polarity=0, outer_bias=0):
    """Polar dynamic programming: require a continuous outer contour.

    Three revolutions remove the arbitrary seam/start radius. The last full
    revolution supplies the path. Missing image samples have neutral reward;
    they never supply points for the fit.
    """
    n = 720
    theta = np.arange(n)*2*np.pi/n
    co, si = np.cos(theta)[:, None], np.sin(theta)[:, None]
    radii = np.arange(d*.405, d*.565, .5)
    xx, yy = center[0]+co*radii, center[1]+si*radii
    vx, vy = remap(gx, xx, yy), remap(gy, xx, yy)
    projection = vx*co+vy*si
    radial = np.maximum(projection*polarity, 0) if polarity else np.abs(projection)
    strength = radial*radial/(np.hypot(vx, vy)+1e-5)
    valid = (xx >= 3) & (yy >= 3) & (xx < gx.shape[1]-3) & (yy < gx.shape[0]-3)
    # Log response keeps a specular highlight from dominating an entire arc.
    reward = np.log1p(strength)/np.log(8.)
    reward -= .45*((radii[None, :]/d-.49)/.08)**2
    reward += outer_bias*(radii[None, :]/d-.49)
    reward[~valid] = 0
    m = len(radii)
    prev = np.zeros(m)
    back = np.zeros((n*3, m), np.int16)
    # Half a pixel per angular step is sufficient for bbox-centre errors and
    # weak ellipticity while discouraging switches between concentric rims.
    transitions = np.arange(-3, 4)
    idx = np.arange(m)
    source_raw = idx[None, :]+transitions[:, None]
    source = np.clip(source_raw, 0, m-1)
    forbidden = (source_raw < 0) | (source_raw >= m)
    penalties = .16*transitions[:, None]**2
    for t in range(n*3):
        choices = prev[source]-penalties
        choices[forbidden] = -1e6
        which = np.argmax(choices, axis=0)
        back[t] = idx+transitions[which]
        prev = choices[which, idx]+reward[t%n]
        prev -= np.max(prev)
    path = np.zeros(n*3, int)
    state = int(np.argmax(prev))
    for t in range(n*3-1, -1, -1):
        path[t] = state
        state = int(back[t, state])
    peaks = path[-n:]
    ii = np.arange(n)
    lo, hi = np.maximum(peaks-1, 0), np.minimum(peaks+1, m-1)
    a, b, c = strength[ii, lo], strength[ii, peaks], strength[ii, hi]
    denom = a-2*b+c
    delta = np.clip(.5*(a-c)/np.where(np.abs(denom)>1e-6, denom, -1e-6), -1, 1)
    rr = radii[peaks]+.5*delta
    points = np.column_stack((center[0]+np.cos(theta)*rr, center[1]+np.sin(theta)*rr))
    good = valid[ii, peaks] & (b > 1.) & (peaks > 0) & (peaks < m-1)
    if good.sum() < 80:
        return None
    try:
        circle = geometric_fit(points[good], np.r_[center, d*.49])
    except (ValueError, np.linalg.LinAlgError):
        return None
    return circle


def refine(gx, gy, initial, d, iterations=12):
    circle = np.array(initial, dtype=float)
    for it in range(iterations):
        band = max(4.0, d*(.035 if it == 0 else .022))
        pts, strength, valid, theta = radial_points(gx, gy, circle, band)
        if valid.any():
            valid &= strength >= max(1.0, np.percentile(strength[valid], 20)*.45)
        if valid.sum() < 40:
            return None
        updated = geometric_fit(pts[valid], circle)
        if abs(updated[2]-d*.49) > .13*d or np.linalg.norm(updated[:2]-initial[:2]) > .20*d:
            return None
        change = np.linalg.norm(updated-circle)
        circle = updated
        if change < .015:
            break
    pts, strength, valid, theta = radial_points(gx, gy, circle, max(4.0, d*.02))
    residual = np.hypot(pts[:, 0]-circle[0], pts[:, 1]-circle[1])-circle[2]
    # The narrow final band rejects competing inner rims and large burrs.
    median = np.median(residual[valid]) if valid.any() else 0
    mad = 1.4826*np.median(np.abs(residual[valid]-median)) if valid.any() else 99
    threshold = min(max(1.6, 2.7*mad), max(3., .018*d))
    inliers = valid & (np.abs(residual) < threshold)
    if inliers.sum() < 40:
        return None
    circle = geometric_fit(pts[inliers], circle, f_scale=1.)
    residual = np.hypot(pts[:, 0]-circle[0], pts[:, 1]-circle[1])-circle[2]
    inliers &= np.abs(residual) < threshold
    if inliers.sum() < 40:
        return None
    rmse = float(np.sqrt(np.mean(residual[inliers]**2)))
    sectors = np.floor((theta[inliers]%(2*np.pi))/(2*np.pi)*72).astype(int)
    coverage = len(np.unique(sectors))/72.
    edge_strength = float(np.median(strength[inliers]))
    return {'circle': circle, 'points': pts, 'inliers': inliers, 'rmse': rmse,
            'coverage': coverage, 'edge_strength': edge_strength,
            'inlier_fraction': float(inliers.mean()), 'threshold': threshold}


def fit_roi(roi, box_local, image_clipped=False, sigma=1.4):
    gray, gx, gy = make_gradients(roi, sigma)
    x1, y1, x2, y2 = box_local
    bw, bh = x2-x1, y2-y1
    d = max(bw, bh)
    center = np.array([(x1+x2)/2, (y1+y2)/2])
    if image_clipped and bw < .85*bh:
        center[0] = x2-d/2 if x1 < 1 else x1+d/2
    if image_clipped and bh < .85*bw:
        center[1] = y2-d/2 if y1 < 1 else y1+d/2
    seeds = [np.r_[center, d*r] for r in (.465, .49, .515)]
    path_seed = trace_outer_rim(gx, gy, center, d, outer_bias=4)
    if path_seed is not None:
        seeds.append(path_seed)
    if image_clipped:
        path_seed2 = trace_outer_rim(gx, gy, center, d, outer_bias=8)
        if path_seed2 is not None:
            seeds.append(path_seed2)
    # Hough is used only to initialize, never as the precision measurement.
    hc = cv2.HoughCircles(np.uint8(np.clip(cv2.GaussianBlur(gray, (0, 0), 1.5), 0, 255)),
                          cv2.HOUGH_GRADIENT, dp=1, minDist=max(8, d*.06),
                          param1=70, param2=20, minRadius=int(d*.425), maxRadius=int(d*.555))
    if hc is not None:
        for c in hc[0]:
            if np.linalg.norm(c[:2]-center) < .18*d:
                seeds.append(c.astype(float))
    # Rank by angular support, edge strength and a weak outer-box scale prior.
    fits = []
    for seed in seeds:
        try:
            f = refine(gx, gy, seed, d)
        except (ValueError, RuntimeError, cv2.error, np.linalg.LinAlgError):
            continue
        if f is None:
            continue
        c = f['circle']
        if np.linalg.norm(c[:2]-center) > .20*d or not .415*d < c[2] < .57*d:
            continue
        f['score'] = (f['coverage'] + .25*f['inlier_fraction']
                      + .08*np.log1p(f['edge_strength']) - .16*f['rmse']
                      - .7*abs(c[2]/d-.49) - .3*np.linalg.norm(c[:2]-center)/d)
        fits.append(f)
    if not fits:
        raise RuntimeError('No supported outer-rim circle found')
    # A single attractive inner rim must not beat the outer contour that many
    # starts converge to. This is a basin-support heuristic, not independent
    # statistical evidence or a confidence probability.
    for f in fits:
        neighbors = sum(np.linalg.norm(f['circle']-g['circle']) < max(1.75, .01*d) for g in fits)
        f['selection_score'] = f['score']+.055*np.log1p(neighbors)
    best = max(fits, key=lambda f: f['selection_score'])
    best['candidate_details'] = [{'circle': f['circle'].tolist(), 'score': f['score'],
                                   'rmse': f['rmse'], 'coverage': f['coverage'],
                                   'inlier_fraction': f['inlier_fraction']} for f in fits]
    near_best = [f for f in fits if f['selection_score'] > best['selection_score']-.06]
    best['candidate_center_spread'] = max(float(np.linalg.norm(f['circle'][:2]-best['circle'][:2]))
                                           for f in near_best)
    # Smoothing-scale rerun: sensitivity, NOT a confidence interval.
    _, gx2, gy2 = make_gradients(roi, sigma*1.5)
    try:
        other = refine(gx2, gy2, best['circle'], d)
    except (ValueError, RuntimeError, cv2.error, np.linalg.LinAlgError):
        other = None
    best['scale_center_shift'] = (float(np.linalg.norm(other['circle'][:2]-best['circle'][:2]))
                                  if other is not None else None)
    best['candidates'] = len(fits)
    pts = best['points'][best['inliers']].astype(np.float32)
    best['ellipse_axis_ratio'] = None
    best['ellipse_center_shift'] = None
    if len(pts) >= 5:
        ec, axes, angle = cv2.fitEllipse(pts.reshape(-1, 1, 2))
        best['ellipse_axis_ratio'] = float(min(axes)/max(axes))
        best['ellipse_center_shift'] = float(np.linalg.norm(np.array(ec)-best['circle'][:2]))
    return best
