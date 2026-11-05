"""螺丝拧紧 / 螺孔对准判断接口。"""

from __future__ import annotations

import numpy as np
from fastapi import APIRouter, Depends, Query

from app.api.deps import get_model_manager, parse_image
from app.schemas.detection import ApiResponse
from app.services import detection as detection_service
from app.services.model_manager import ModelManager

router = APIRouter(prefix="/api", tags=["拧紧/对准判断"])


async def _run_judge(manager: ModelManager, key: str, image: np.ndarray, conf: float) -> dict:
    result = await manager.predict_single_async(key, image, conf)
    return detection_service.build_judge_data(result)


@router.post(
    "/bolt7_judge",
    summary="bolt7_judge：检测7号螺丝是否拧紧",
    response_model=ApiResponse,
)
async def bolt7_judge(
    image: np.ndarray = Depends(parse_image),
    conf: float = Query(0.5, ge=0.0, le=1.0, description="检测置信度阈值 0~1"),
    manager: ModelManager = Depends(get_model_manager),
) -> ApiResponse:
    data = await _run_judge(manager, "bolt7_judge", image, conf)
    return ApiResponse(task="bolt7_judge", data=data)


@router.post(
    "/bolt11_judge",
    summary="bolt11_judge：检测11号螺丝是否拧紧",
    response_model=ApiResponse,
)
async def bolt11_judge(
    image: np.ndarray = Depends(parse_image),
    conf: float = Query(0.5, ge=0.0, le=1.0, description="检测置信度阈值 0~1"),
    manager: ModelManager = Depends(get_model_manager),
) -> ApiResponse:
    data = await _run_judge(manager, "bolt11_judge", image, conf)
    return ApiResponse(task="bolt11_judge", data=data)


@router.post(
    "/hole7_judge",
    summary="hole7_judge：检测7号螺孔是否对准",
    response_model=ApiResponse,
)
async def hole7_judge(
    image: np.ndarray = Depends(parse_image),
    conf: float = Query(0.5, ge=0.0, le=1.0, description="检测置信度阈值 0~1"),
    manager: ModelManager = Depends(get_model_manager),
) -> ApiResponse:
    data = await _run_judge(manager, "hole7_judge", image, conf)
    return ApiResponse(task="hole7_judge", data=data)


@router.post(
    "/hole11_judge",
    summary="hole11_judge：检测11号螺孔是否对准",
    response_model=ApiResponse,
)
async def hole11_judge(
    image: np.ndarray = Depends(parse_image),
    conf: float = Query(0.5, ge=0.0, le=1.0, description="检测置信度阈值 0~1"),
    manager: ModelManager = Depends(get_model_manager),
) -> ApiResponse:
    data = await _run_judge(manager, "hole11_judge", image, conf)
    return ApiResponse(task="hole11_judge", data=data)
