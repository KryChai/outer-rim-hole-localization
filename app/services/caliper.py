"""Normal calipers, multi-peak RANSAC and robust orthogonal ellipse refinement.

No robot, YOLO, camera or metric-size dependency. Raw image points remain paired
with their undistorted fit points. A rejected fit never becomes a box center.
"""
from dataclasses import asdict, dataclass
import math
import time

import cv2
import numpy as np
from scipy.optimize import least_squares

BOUNDARY_METHOD = 'normal_caliper_ransac_v1'
BOUNDARY_REVISION = 2


@dataclass(frozen=True)
class CaliperConfig:
    calipers: int = 96
    tangent_half_width: int = 2
    min_gradient: float = 3.0
    max_peaks: int = 6
    ransac_trials: int = 480
    inlier_distance_px: float = 1.5
    min_support: float = .65
    min_coverage_deg: float = 270.0
    max_rms_px: float = 1.2
    seed: int = 19

    def validate(self):
        values = asdict(self)
        integer_keys = ('calipers', 'tangent_half_width', 'max_peaks', 'ransac_trials', 'seed')
        if (not all(isinstance(values[k], int) and not isinstance(values[k], bool) for k in integer_keys)
                or not all(math.isfinite(v) for v in values.values()) or self.seed < 0):
            raise ValueError('invalid_caliper_configuration')
        if not (48 <= self.calipers <= 240 and 0 <= self.tangent_half_width <= 5
                and 1 <= self.max_peaks <= 8 and 64 <= self.ransac_trials <= 2000
                and 0 < self.min_gradient <= 50 and .25 <= self.inlier_distance_px <= 3
                and .5 <= self.min_support <= 1 and 180 <= self.min_coverage_deg <= 360
                and 0 < self.max_rms_px <= 1.2):
            raise ValueError('invalid_caliper_configuration')


def ellipse_points(ellipse, count=96):
    center, axes, degrees = ellipse
    a = math.radians(degrees)
    rotation = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
    t = np.linspace(0, 2*np.pi, count, endpoint=False)
    return (np.c_[np.cos(t), np.sin(t)]*(np.array(axes)/2)) @ rotation.T + center


def conic(ellipse):
    center, axes, degrees = ellipse
    a = math.radians(degrees)
    rotation = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
    matrix = rotation @ np.diag(4/np.array(axes)**2) @ rotation.T
    center = np.array(center)
    return np.block([[matrix, (-matrix@center)[:, None]],
                     [(-center@matrix)[None, :], np.array([[center@matrix@center-1]])]])


def approximate_distance(points, ellipse):
    q = conic(ellipse)
    f = np.einsum('ij,jk,ik->i', np.c_[points, np.ones(len(points))], q,
                  np.c_[points, np.ones(len(points))])
    g = 2*(points@q[:2, :2]+q[:2, 2])
    return np.abs(f)/np.maximum(np.linalg.norm(g, axis=1), 1e-9)


def orthogonal_residual(points, params):
    """Signed point-to-ellipse distance; Newton solve is local to near-edge data."""
    cx, cy, log_a, log_b, angle = params
    a, b = np.exp([log_a, log_b])
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    local = (points-[cx, cy])@rotation
    x, y = local.T
    t = np.arctan2(a*y, b*x)
    for _ in range(10):
        s, c = np.sin(t), np.cos(t)
        f = (b*b-a*a)*s*c+a*x*s-b*y*c
        df = (b*b-a*a)*(c*c-s*s)+a*x*c+b*y*s
        t -= np.clip(f/np.where(np.abs(df) > 1e-9, df, 1e-9), -.3, .3)
    nearest = np.c_[a*np.cos(t), b*np.sin(t)]
    sign = np.where((x/a)**2+(y/b)**2 >= 1, 1., -1.)
    return np.linalg.norm(local-nearest, axis=1)*sign


def undistort(points, calibration):
    if calibration is None: return points.copy()
    return cv2.undistortPoints(points.reshape(-1, 1, 2), calibration.matrix,
                              calibration.distortion, P=calibration.matrix).reshape(-1, 2)


def distort(points, calibration):
    if calibration is None: return points.copy()
    rays = np.c_[points, np.ones(len(points))]@np.linalg.inv(calibration.matrix).T
    return cv2.projectPoints(rays, np.zeros(3), np.zeros(3), calibration.matrix,
                            calibration.distortion)[0].reshape(-1, 2)


def normal_samples(gray, ellipse, offsets, config, calibration=None, invalid_mask=None):
    """Small tangent strips, signed peaks, subpixel interpolation; no forced peak."""
    points = ellipse_points(ellipse, config.calipers)
    q = conic(ellipse)
    normals = points@q[:2, :2]+q[:2, 2]
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    tangent = np.c_[-normals[:, 1], normals[:, 0]]
    tangent_offsets = np.arange(-config.tangent_half_width, config.tangent_half_width+1)
    samples = (points[:, None, None, :]+normals[:, None, None, :]*offsets[None, :, None, None]
               +tangent[:, None, None, :]*tangent_offsets[None, None, :, None])
    raw = distort(samples.reshape(-1, 2), calibration).reshape(samples.shape)
    valid = ((raw[..., 0] >= 1) & (raw[..., 0] < gray.shape[1]-2)
             & (raw[..., 1] >= 1) & (raw[..., 1] < gray.shape[0]-2)).all(axis=2)
    # Keep one map row per caliper. Flattening caliper*offset into map rows
    # exceeds OpenCV's signed-short remap limit for ordinary ~330px holes.
    coords = raw.reshape(config.calipers, -1, 2)
    values = cv2.remap(gray, coords[..., 0].astype('float32'), coords[..., 1].astype('float32'),
                        cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT).reshape(raw.shape[:-1])
    if invalid_mask is not None:
        masks = cv2.remap(invalid_mask.astype('uint8'), coords[..., 0].astype('float32'),
                         coords[..., 1].astype('float32'), cv2.INTER_NEAREST,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=1).reshape(raw.shape[:-1])
        valid &= ~(masks > 0).any(axis=2)
    profiles = values.mean(axis=2)
    profiles = cv2.GaussianBlur(profiles, (5, 1), .6)
    step = offsets[1]-offsets[0]
    gradient = (profiles[:, 2:]-profiles[:, :-2])/(2*step)
    abs_g = np.abs(gradient)
    records = []
    for group in range(config.calipers):
        peaks = np.where((abs_g[group, 1:-1] >= abs_g[group, :-2])
                         & (abs_g[group, 1:-1] > abs_g[group, 2:])
                         & (abs_g[group, 1:-1] >= config.min_gradient))[0]+1
        peaks = sorted(peaks, key=lambda i: -abs_g[group, i])
        used = []
        for peak in peaks:
            if not valid[group, peak:peak+3].all(): continue
            if any(abs(peak-i)*step < 2 for i in used): continue
            left, mid, right = abs_g[group, peak-1:peak+2]
            delta = np.clip(.5*(left-right)/min(left-2*mid+right, -1e-6), -.5, .5)
            distance = offsets[peak+1]+step*delta
            fit_point = points[group]+normals[group]*distance
            raw_point = distort(fit_point[None], calibration)[0]
            records.append((raw_point, fit_point, group, float(gradient[group, peak])))
            used.append(peak)
            if len(used) >= config.max_peaks: break
    if not records:
        return {'raw': np.empty((0, 2)), 'fit': np.empty((0, 2)), 'group': np.empty(0, int), 'strength': np.empty(0)}
    return {'raw': np.array([r[0] for r in records]), 'fit': np.array([r[1] for r in records]),
            'group': np.array([r[2] for r in records]), 'strength': np.array([r[3] for r in records])}


def valid_ellipse(ellipse, bbox):
    center, axes, angle = ellipse
    x1, y1, x2, y2 = bbox
    return (np.isfinite([*center, *axes, angle]).all() and min(axes) >= 8
            and min(axes)/max(axes) >= .25 and max(axes) <= 1.35*max(x2-x1, y2-y1)
            and x1 <= center[0] <= x2 and y1 <= center[1] <= y2)


def select_support(samples, ellipse, config, threshold=None):
    distance = approximate_distance(samples['fit'], ellipse)
    order = np.argsort(distance, kind='stable')
    indices = []
    groups = set()
    for i in order:
        if distance[i] > (threshold or config.inlier_distance_px): break
        group = int(samples['group'][i])
        if group not in groups:
            groups.add(group); indices.append(int(i))
    return np.array(indices, dtype=int)


def same_ellipse(first, second, tolerance=2.):
    return (np.linalg.norm(np.array(first[0])-second[0]) < tolerance
            and np.max(np.abs(np.sort(first[1])-np.sort(second[1]))) < 2*tolerance
            and np.mean(approximate_distance(ellipse_points(first), second)) < tolerance)


def hypotheses(samples, bbox, config):
    if len(samples['fit']) < 24: return []
    center = np.array([(bbox[0]+bbox[2])/2, (bbox[1]+bbox[3])/2])
    radius = np.array([bbox[2]-bbox[0], bbox[3]-bbox[1]])/2
    rho = np.linalg.norm((samples['fit']-center)/radius, axis=1)
    pools = [np.where(np.abs(rho-r) < .13)[0] for r in np.arange(.3, 1.21, .1)]
    rng = np.random.default_rng(config.seed)
    candidates = []
    for trial in range(config.ransac_trials):
        pool = pools[trial % len(pools)]
        if len(pool) < 12: continue
        chosen = rng.choice(pool, 5, replace=False)
        if len(np.unique(samples['group'][chosen]//(config.calipers//8))) < 4: continue
        try:
            ellipse = cv2.fitEllipseDirect(samples['fit'][chosen].astype('float32'))
            if not valid_ellipse(ellipse, bbox): continue
            support = select_support(samples, ellipse, config, threshold=2.)
            if len(support) < .4*config.calipers: continue
            ellipse = cv2.fitEllipseDirect(samples['fit'][support].astype('float32'))
            if not valid_ellipse(ellipse, bbox): continue
            support = select_support(samples, ellipse, config)
            score = len(support)
            if score < .45*config.calipers: continue
            duplicate = next((i for i, item in enumerate(candidates) if same_ellipse(ellipse, item[1])), None)
            if duplicate is None: candidates.append((score, ellipse))
            elif score > candidates[duplicate][0]: candidates[duplicate] = (score, ellipse)
        except (cv2.error, ValueError, FloatingPointError):
            continue
    # Do not spend the entire hypothesis budget on near-identical outer rings.
    # Preserve scale diversity so that eccentric smaller/deeper rims can survive.
    selected, bins = [], {}
    scale = max(bbox[2]-bbox[0], bbox[3]-bbox[1])
    for _, ellipse in sorted(candidates, key=lambda c: -c[0]):
        bucket = int(max(ellipse[1])/scale/.15)
        if bins.get(bucket, 0) >= 3: continue
        selected.append(ellipse); bins[bucket] = bins.get(bucket, 0)+1
        if len(selected) >= 18: break
    return selected


def dark_region_seeds(gray, bbox, calibration):
    """Only initialization: final edges are always remeasured on raw intensity.

    A deep mouth can be strongly eccentric relative to the detection box. Region
    seeds remove that center assumption without treating a threshold mask as a
    measured physical boundary or using it to bypass the quality gates.
    """
    x1, y1, x2, y2 = bbox
    scale = max(x2-x1, y2-y1)
    padding = .15*scale
    left, top = max(0, int(x1-padding)), max(0, int(y1-padding))
    right, bottom = min(gray.shape[1], int(x2+padding+1)), min(gray.shape[0], int(y2+padding+1))
    roi = gray[top:bottom, left:right]
    central = gray[int(y1):int(y2)+1, int(x1):int(x2)+1]
    seeds = []
    for threshold in np.unique(np.percentile(central, [10, 20, 30, 40])):
        binary = (roi <= threshold).astype('uint8')*255
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for contour in contours:
            if len(contour) < 24 or cv2.contourArea(contour) < .008*scale**2: continue
            raw = contour.reshape(-1, 2).astype(float)
            if (np.min(raw[:, 0]) <= 1 or np.min(raw[:, 1]) <= 1
                    or np.max(raw[:, 0]) >= roi.shape[1]-2 or np.max(raw[:, 1]) >= roi.shape[0]-2): continue
            raw = raw[::max(1, len(raw)//240)]+[left, top]
            points = undistort(raw, calibration)
            try:
                ellipse = cv2.fitEllipseDirect(points.astype('float32'))
                if not valid_ellipse(ellipse, bbox) or min(ellipse[1]) < .12*scale: continue
                rms = float(np.sqrt(np.mean(approximate_distance(points, ellipse)**2)))
                if rms > max(3., .02*scale): continue
                if not any(same_ellipse(ellipse, s, 2.) for s in seeds): seeds.append(ellipse)
            except (ValueError, cv2.error, FloatingPointError): continue
    return sorted(seeds, key=lambda e: np.prod(e[1]))[:12]


def refine_candidate(gray, ellipse, bbox, config, calibration, invalid_mask):
    scale = max(bbox[2]-bbox[0], bbox[3]-bbox[1])
    samples = None
    for _ in range(2):
        length = min(10., max(3., .04*scale))
        samples = normal_samples(gray, ellipse, np.arange(-length, length+.25, .5), config,
                                 calibration, invalid_mask)
        selected = select_support(samples, ellipse, config, threshold=min(3., length/2))
        if len(selected) < .5*config.calipers: return None, 'insufficient_caliper_support'
        points = samples['fit'][selected]
        start = np.r_[ellipse[0], np.log(np.array(ellipse[1])/2), math.radians(ellipse[2])]
        bound = max(4., .08*scale)
        lower = np.r_[np.array(ellipse[0])-bound, np.log([4., 4.]), start[4]-np.pi/2]
        upper = np.r_[np.array(ellipse[0])+bound, np.log([scale, scale]), start[4]+np.pi/2]
        solution = least_squares(lambda x: orthogonal_residual(points, x), start, bounds=(lower, upper),
                                 loss='huber', f_scale=.7, max_nfev=35)
        if not solution.success: return None, 'robust_refinement_not_converged'
        p = solution.x
        ellipse = (tuple(p[:2]), tuple(2*np.exp(p[2:4])), float(np.degrees(p[4]) % 180))
        if not valid_ellipse(ellipse, bbox): return None, 'invalid_refined_ellipse'
    selected = select_support(samples, ellipse, config)
    if len(selected) < config.min_support*config.calipers: return None, 'insufficient_caliper_support'
    points = samples['fit'][selected]
    params = np.r_[ellipse[0], np.log(np.array(ellipse[1])/2), math.radians(ellipse[2])]
    residual = orthogonal_residual(points, params)
    rms = float(np.sqrt(np.mean(residual**2)))
    angle = math.radians(ellipse[2])
    rotation = np.array([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])
    unit = (points-ellipse[0])@rotation/(np.array(ellipse[1])/2)
    bins = np.floor((np.arctan2(unit[:, 1], unit[:, 0])+np.pi)/(2*np.pi)*36).astype(int).clip(0, 35)
    coverage = len(np.unique(bins))/36
    if rms > config.max_rms_px: return None, 'excessive_orthogonal_residual'
    if coverage*360 < config.min_coverage_deg: return None, 'insufficient_angular_coverage'
    return {'ellipse': ellipse, 'q': conic(ellipse), 'rms': rms, 'coverage': coverage,
            'edges': samples['raw'][selected], 'support_fraction': len(selected)/config.calipers,
            'signed_strength': samples['strength'][selected].tolist(),
            'candidate_points_px': samples['raw'].tolist(), 'selected_indices': selected.tolist()}, ''


def detect_boundaries(image, bbox, calibration=None, config=None, invalid_mask=None, outer_only=False):
    config = config or CaliperConfig(); config.validate()
    start = time.perf_counter()
    bbox = np.asarray(bbox, dtype=float)
    if bbox.shape != (4,) or not np.isfinite(bbox).all() or min(bbox[2:]-bbox[:2]) < 16:
        raise ValueError('invalid_bbox')
    if bbox[0] <= 1 or bbox[1] <= 1 or bbox[2] >= image.shape[1]-1 or bbox[3] >= image.shape[0]-1:
        raise ValueError('hole_at_image_border')
    if calibration is not None and image.shape[1::-1] != calibration.image_size:
        raise ValueError('calibration_resolution_mismatch')
    if invalid_mask is not None and invalid_mask.shape != image.shape[:2]: raise ValueError('invalid_mask_size')
    gray = cv2.GaussianBlur(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), (5, 5), .8)
    center = tuple((bbox[:2]+bbox[2:])/2)
    seed = (center, tuple((bbox[2:]-bbox[:2])*.96), 0.)
    if calibration is not None:
        seed = cv2.fitEllipseDirect(undistort(ellipse_points(seed), calibration).astype('float32'))
    scale = max(bbox[2:]-bbox[:2])
    scan_min, scan_max = (-.18*scale, .12*scale) if outer_only else (-.36*scale, .17*scale)
    samples = normal_samples(gray, seed, np.arange(scan_min, scan_max+.25, .5), config,
                             calibration, invalid_mask)
    ransac_initial = hypotheses(samples, bbox, config)
    region_initial = dark_region_seeds(gray, bbox, calibration)
    initial = list(region_initial)
    for ellipse in ransac_initial:
        if not any(same_ellipse(ellipse, other, 1.5) for other in initial): initial.append(ellipse)
    if outer_only:
        # A single outer rim is sufficient. Never require an inner dark region.
        initial = [seed]+[e for e in initial if np.all(
            np.ptp(ellipse_points(e), axis=0)/(bbox[2:]-bbox[:2]) >= .72)]
    candidates, rejected = [], []
    for ellipse in initial:
        try:
            candidate, reason = refine_candidate(gray, ellipse, bbox, config, calibration, invalid_mask)
            if candidate is None: rejected.append({'ellipse': ellipse, 'reason': reason}); continue
            if not any(same_ellipse(candidate['ellipse'], c['ellipse'], .8) for c in candidates): candidates.append(candidate)
        except (ValueError, cv2.error, FloatingPointError, np.linalg.LinAlgError) as exc:
            rejected.append({'ellipse': ellipse, 'reason': str(exc)})
    candidates.sort(key=lambda c: np.prod(c['ellipse'][1]), reverse=True)
    report = {'boundary_method': BOUNDARY_METHOD, 'config': asdict(config),
              'detection_target': 'outer_rim' if outer_only else 'all_boundaries',
              'sample_count': len(samples['raw']), 'hypothesis_count': len(initial),
              'initialization_revision': BOUNDARY_REVISION, 'region_seed_count': len(region_initial),
              'ransac_seed_count': len(ransac_initial),
              'candidate_count': len(candidates), 'rejected': rejected,
              'stage': ('candidates_ready' if candidates else 'all_candidates_rejected' if initial
                        else 'no_edges' if len(samples['raw']) == 0 else 'no_ransac_model'),
              'initial_candidate_points_px': samples['raw'].tolist(),
              'initial_caliper_groups': samples['group'].tolist(),
              'initial_signed_strength': samples['strength'].tolist(),
              'accepted_candidates': [{k: (v.tolist() if isinstance(v, np.ndarray) else v)
                                       for k, v in c.items() if k != 'q'} for c in candidates],
              'elapsed_ms': (time.perf_counter()-start)*1000,
              'coordinate_frame': 'original_image_pixels', 'calibration_used': calibration is not None}
    return candidates, report
