"""Largest visible outer hole rim in original pixels. No inner rim or K needed.

The returned point is the fitted image ellipse center, NOT a perspective-
corrected spatial circle center. The detection box scopes the physical hole.
"""
import cv2
import numpy as np

from app.services.caliper import BOUNDARY_METHOD, BOUNDARY_REVISION, detect_boundaries, ellipse_points

ALGORITHM = 'outer_rim_ellipse_v1'
CENTER_DEFINITION = 'fitted_outer_ellipse_center'
REFERENCE = 'outer_rim'


def select_outer(candidates, bbox):
    bbox = np.asarray(bbox, dtype=float)
    size = bbox[2:]-bbox[:2]
    tolerance = .03*max(size)
    accepted = []
    checks = []
    for index, candidate in enumerate(candidates):
        curve = ellipse_points(candidate['ellipse'], 180)
        span = np.ptp(curve, axis=0)/size
        inside = np.mean(np.all((curve >= bbox[:2]-tolerance) & (curve <= bbox[2:]+tolerance), axis=1))
        reason = ''
        if np.any(span < .78): reason = 'too_small_for_outer_rim'
        elif inside < .95: reason = 'outer_outline_outside_target_box'
        checks.append({'candidate_index': index, 'span_ratio': span.tolist(),
                       'fraction_inside_box': float(inside), 'reason': reason})
        if not reason: accepted.append((float(np.prod(candidate['ellipse'][1])), index, candidate))
    if not accepted: return None, checks, 'outer_rim_not_observed'
    accepted.sort(key=lambda item: -item[0])
    # Select maximum area among qualified outer outlines; RMS cannot switch the
    # reference to a smaller/deeper rim. Keep all alternatives in the audit.
    area, index, candidate = accepted[0]
    for check in checks: check['selected'] = check['candidate_index'] == index
    return (index, candidate), checks, ''


def refine_outer_hole(image, hole):
    out = {**hole, 'detection_center': {k: hole[k] for k in ('center_x', 'center_y')},
           'refined_center': None, 'fit_quality': {'valid': False, 'reason': ''},
           'geometry': None, 'boundary_diagnostics': None}
    try:
        bbox = [hole[k] for k in ('x1', 'y1', 'x2', 'y2')]
        candidates, diagnostics = detect_boundaries(image, bbox, outer_only=True)
        out['boundary_diagnostics'] = diagnostics
        selected, checks, reason = select_outer(candidates, bbox)
        diagnostics['outer_selection_checks'] = checks
        diagnostics['selection_policy'] = 'largest_qualified_ellipse_area'
        if selected is None: raise ValueError(reason)
        index, rim = selected
        center = rim['ellipse'][0]
        out['refined_center'] = {'center_x': float(center[0]), 'center_y': float(center[1]),
                                'method': ALGORITHM, 'reference': REFERENCE,
                                'center_definition': CENTER_DEFINITION}
        out['fit_quality'] = {'valid': True, 'reason': '', 'residual_rms_px': rim['rms'],
                             'angular_coverage_deg': 360*rim['coverage'],
                             'caliper_support_fraction': rim['support_fraction'],
                             'boundary_method': BOUNDARY_METHOD, 'metric_scale_known': False}
        out['geometry'] = {'outlines_px': [ellipse_points(rim['ellipse'], 180).tolist()],
                           'edge_points_px': [rim['edges'].tolist()],
                           'ellipse': {'center_px': list(center), 'diameters_px': list(rim['ellipse'][1]),
                                       'angle_deg': float(rim['ellipse'][2])},
                           'selected_candidate_index': index}
    except (ValueError, KeyError, cv2.error, np.linalg.LinAlgError) as exc:
        out['fit_quality']['reason'] = str(exc)
    return out


def build_outer_data(image, holes, target, max_distance=80):
    distances = [np.linalg.norm(np.array([h['center_x'], h['center_y']])-target) for h in holes]
    order = np.argsort(distances)
    selected = int(order[0]) if len(order) and distances[order[0]] <= max_distance else None
    reason = 'target_not_found' if selected is None else ''
    if len(order) > 1 and distances[order[1]]-distances[order[0]] < 5:
        selected, reason = None, 'ambiguous_target'
    results = [refine_outer_hole(image, h) for h in holes]
    return {'algorithm': ALGORITHM, 'boundary_method': BOUNDARY_METHOD, 'boundary_revision': BOUNDARY_REVISION,
            'reference': REFERENCE, 'center_definition': CENTER_DEFINITION,
            'coordinate_frame': 'original_image_pixels', 'image_size': list(image.shape[1::-1]),
            'calibration_id': None, 'calibration_required': False, 'metric_scale_known': False,
            'holes': results, 'target_hole': results[selected] if selected is not None else None, 'reason': reason}
