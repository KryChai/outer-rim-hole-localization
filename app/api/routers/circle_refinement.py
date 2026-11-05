"""Per-detection outer circle fitting, distinct from whole-pattern circle_fit."""
import asyncio

import numpy as np
from fastapi import APIRouter, Depends, Query

from app.api.deps import get_model_manager, parse_image
from app.schemas.detection import ApiResponse
from app.schemas.circle_refinement import CircleRefinementResponse
from app.services.detection import parse_holes
from app.services.circle_refinement import build_circle_refinement_data

router = APIRouter(prefix="/api", tags=["螺孔外圈圆拟合精定位"])


async def _run(manager, model, image, conf, check_stability):
    result = await manager.predict_single_async(model, image, conf)
    holes, _ = parse_holes(result, conf) if result is not None else ([], [])
    data = await asyncio.to_thread(build_circle_refinement_data, image, holes,
                                   check_stability, result is None)
    return ApiResponse(task=model+"_circle_refine", data=data)


@router.post("/usb_hole7_circle_refine", response_model=CircleRefinementResponse,
             summary="7 型：每个检测框扩大 5% 后拟合孔口外圈圆")
async def usb_hole7_circle_refine(
    image: np.ndarray = Depends(parse_image),
    conf: float = Query(.5, ge=0, le=1),
    check_stability: bool = Query(True, description="四方向 3px 框扰动质量检查；关闭时标为 review"),
    manager=Depends(get_model_manager),
):
    return await _run(manager, "usb_hole7", image, conf, check_stability)


@router.post("/usb_hole11_circle_refine", response_model=CircleRefinementResponse,
             summary="11 型：每个检测框扩大 5% 后拟合孔口外圈圆")
async def usb_hole11_circle_refine(
    image: np.ndarray = Depends(parse_image),
    conf: float = Query(.5, ge=0, le=1),
    check_stability: bool = Query(True, description="四方向 3px 框扰动质量检查；关闭时标为 review"),
    manager=Depends(get_model_manager),
):
    return await _run(manager, "usb_hole11", image, conf, check_stability)
