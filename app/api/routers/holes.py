"""单孔最高置信度检测接口。"""

from __future__ import annotations

import numpy as np
from fastapi import APIRouter, Depends, Query

from app.api.deps import get_model_manager, parse_image
from app.core import audit
from app.schemas.detection import ApiResponse
from app.services import detection as detection_service
from app.services.model_manager import ModelManager

router = APIRouter(prefix="/api", tags=["孔位检测"])


@router.post(
    "/usb_hole7_detect",
    summary="根据图片绝对路径获取置信度最高7号孔检测框中心点坐标",
    response_model=ApiResponse,
)
async def usb_hole7_detect(
    image: np.ndarray = Depends(parse_image),
    conf: float = Query(0.5, ge=0.0, le=1.0, description="检测置信度阈值 0~1"),
    manager: ModelManager = Depends(get_model_manager),
) -> ApiResponse:
    result = await manager.predict_single_async("usb_hole7", image, conf)
    data = detection_service.build_best_hole_data(result, conf)
    audit.log_detect("detect_best_hole7", conf, data)
    return ApiResponse(task="detect_best_hole7", data=data)


@router.post(
    "/usb_hole11_detect",
    summary="根据图片绝对路径获取置信度最高11号孔检测框中心点坐标",
    response_model=ApiResponse,
)
async def usb_hole11_detect(
    image: np.ndarray = Depends(parse_image),
    conf: float = Query(0.5, ge=0.0, le=1.0, description="检测置信度阈值 0~1"),
    manager: ModelManager = Depends(get_model_manager),
) -> ApiResponse:
    result = await manager.predict_single_async("usb_hole11", image, conf)
    data = detection_service.build_best_hole_data(result, conf)
    audit.log_detect("detect_best_hole11", conf, data)
    return ApiResponse(task="detect_best_hole11", data=data)
