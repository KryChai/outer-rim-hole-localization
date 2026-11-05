"""Near-field single-hole refinement, separate from legacy whole-ring fitting."""
import asyncio
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import threading
from uuid import uuid4
from typing import Literal

from fastapi import APIRouter, Depends, Query
import numpy as np

from app.api.deps import get_model_manager, parse_image
from app.core.exceptions import ServiceError
from app.core.logging import logger
from app.schemas.detection import ApiResponse
from app.services.detection import parse_holes
from app.services.refinement import ALGORITHM, build_refinement_data, load_calibration
from app.services.caliper import BOUNDARY_METHOD, BOUNDARY_REVISION
from app.services import outer_rim

router = APIRouter(prefix="/api", tags=["近景单孔精定位"])
_audit_lock = threading.Lock()


@router.get("/refinement/status")
async def refinement_status(mode: Literal['outer_rim', 'coaxial'] = Query('outer_rim')):
    if mode == 'outer_rim':
        return {'code': 200, 'algorithm': outer_rim.ALGORITHM, 'boundary_method': BOUNDARY_METHOD,
                'boundary_revision': BOUNDARY_REVISION, 'ready': True,
                'reference': outer_rim.REFERENCE, 'center_definition': outer_rim.CENTER_DEFINITION,
                'calibration_id': None, 'calibration_required': False, 'image_size': None,
                'image_size_policy': 'from_upload', 'coordinate_frame': 'original_image_pixels',
                'metric_scale_known': False}
    try:
        calibration = load_calibration()
    except ValueError as exc:
        raise ServiceError(str(exc), status_code=503) from exc
    return {"code": 200, "algorithm": ALGORITHM, "boundary_method": BOUNDARY_METHOD, "boundary_revision": BOUNDARY_REVISION, "ready": True,
            "calibration_id": calibration.identifier, "image_size": list(calibration.image_size),
            "coordinate_frame": "original_image_pixels", "reference": "outer_front",
            "metric_scale_known": False}


def _fit_and_audit(image, holes, calibration, target, max_distance, model, mode='outer_rim'):
    data = (outer_rim.build_outer_data(image, holes, target, max_distance) if mode == 'outer_rim'
            else build_refinement_data(image, holes, calibration, target, max_distance))
    data["request_id"] = uuid4().hex
    data["requested_target"] = list(target)
    data["model"] = model
    # Fitted edges/outlines and every center are preserved, not just console text.
    audit_path = os.environ.get("REFINEMENT_LOG_DIR")
    if not audit_path:
        data["audit_saved"] = False
        return data
    log_dir = Path(audit_path)
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        record = {"timestamp": datetime.now(timezone.utc).isoformat(), **data,
                  "calibration_snapshot": None if calibration is None else {"camera_matrix": calibration.matrix.tolist(),
                     "dist_coeffs": calibration.distortion.tolist(), "image_size": list(calibration.image_size)}}
        with _audit_lock, (log_dir / "refinement.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
            stream.flush()
        data["audit_saved"] = True
    except (OSError, ValueError):
        logger.exception("精定位日志保存失败")
        data["audit_saved"] = False
    return data


async def _run(model, image, conf, manager, target_u, target_v, max_distance, mode='outer_rim'):
    if not all(math.isfinite(v) for v in (target_u, target_v, max_distance)):
        raise ServiceError("Target coordinates and gate must be finite", 400)
    calibration = None
    if mode == 'coaxial':
        try:
            calibration = load_calibration()
        except ValueError as exc:
            raise ServiceError(str(exc), 503) from exc
        if image.shape[1::-1] != calibration.image_size:
            raise ServiceError("Image size does not match camera calibration", 400)
    if not (0 <= target_u < image.shape[1] and 0 <= target_v < image.shape[0]):
        raise ServiceError("Target outside image", 400)
    result = await manager.predict_single_async(model, image, conf)
    holes, _ = parse_holes(result, conf) if result is not None else ([], [])
    data = await asyncio.to_thread(_fit_and_audit, image, holes, calibration,
                                   (target_u, target_v), max_distance, model, mode)
    return ApiResponse(task=model + "_refine_center", data=data)


@router.post("/usb_hole7_refine_center", response_model=ApiResponse,
             summary="7 型孔：默认拟合检测框内最外圈中心")
async def hole7_refine(image: np.ndarray = Depends(parse_image),
                       conf: float = Query(.5, ge=0, le=1),
                       target_u: float = Query(..., ge=0), target_v: float = Query(..., ge=0),
                       max_distance: float = Query(80, gt=0, le=500),
                       mode: Literal['outer_rim', 'coaxial'] = Query('outer_rim'),
                       manager=Depends(get_model_manager)):
    return await _run("usb_hole7", image, conf, manager, target_u, target_v, max_distance, mode)


@router.post("/usb_hole11_refine_center", response_model=ApiResponse,
             summary="11 型孔：默认拟合检测框内最外圈中心")
async def hole11_refine(image: np.ndarray = Depends(parse_image),
                        conf: float = Query(.5, ge=0, le=1),
                        target_u: float = Query(..., ge=0), target_v: float = Query(..., ge=0),
                        max_distance: float = Query(80, gt=0, le=500),
                        mode: Literal['outer_rim', 'coaxial'] = Query('outer_rim'),
                        manager=Depends(get_model_manager)):
    return await _run("usb_hole11", image, conf, manager, target_u, target_v, max_distance, mode)
