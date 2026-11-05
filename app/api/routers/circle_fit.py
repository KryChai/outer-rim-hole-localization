"""全部螺孔检测框中心 + 最小二乘拟合圆接口。"""

from __future__ import annotations

import numpy as np
from fastapi import APIRouter, Depends, Query

from app.api.deps import get_model_manager, parse_image
from app.core import audit
from app.schemas.detection import ApiResponse
from app.services import detection as detection_service
from app.services.model_manager import ModelManager

router = APIRouter(prefix="/api", tags=["拟合圆检测"])


async def _run_circle_fit(
    manager: ModelManager,
    key: str,
    image: np.ndarray,
    conf: float,
    task: str,
    noun: str = "螺孔",
) -> dict:
    result = await manager.predict_single_async(key, image, conf)
    data = detection_service.build_circle_fit_data(result, conf, noun=noun)
    audit.log_detect(task, conf, data)
    return data


@router.post(
    "/usb_hole7_circle_fit",
    summary="usb_hole7：输出所有螺孔检测框中心 + 最小二乘拟合整圆圆心",
    response_model=ApiResponse,
)
async def usb_hole7_circle_fit(
    image: np.ndarray = Depends(parse_image),
    conf: float = Query(0.5, ge=0.0, le=1.0, description="检测置信度阈值 0~1"),
    manager: ModelManager = Depends(get_model_manager),
) -> ApiResponse:
    data = await _run_circle_fit(manager, "usb_hole7", image, conf, "hole7_circle_fit")
    return ApiResponse(task="hole7_circle_fit", data=data)


@router.post(
    "/hole7_mid_on_base_circle_fit",
    summary="hole7_mid_on_base：输出所有螺孔检测框中心 + 最小二乘拟合整圆圆心",
    response_model=ApiResponse,
)
async def hole7_mid_on_base_circle_fit(
    image: np.ndarray = Depends(parse_image),
    conf: float = Query(0.5, ge=0.0, le=1.0, description="检测置信度阈值 0~1"),
    manager: ModelManager = Depends(get_model_manager),
) -> ApiResponse:
    data = await _run_circle_fit(manager, "hole7_mid_on_base", image, conf,
                                 "hole7_mid_on_base_circle_fit")
    return ApiResponse(task="hole7_mid_on_base_circle_fit", data=data)


@router.post(
    "/usb_hole11_circle_fit",
    summary="usb_hole11：输出所有螺孔检测框中心 + 最小二乘拟合整圆圆心",
    response_model=ApiResponse,
)
async def usb_hole11_circle_fit(
    image: np.ndarray = Depends(parse_image),
    conf: float = Query(0.5, ge=0.0, le=1.0, description="检测置信度阈值 0~1"),
    manager: ModelManager = Depends(get_model_manager),
) -> ApiResponse:
    data = await _run_circle_fit(manager, "usb_hole11", image, conf, "hole11_circle_fit")
    return ApiResponse(task="hole11_circle_fit", data=data)


@router.post(
    "/hat_on_table_hole7_circle_fit",
    summary="hat_on_table_hole7：输出所有螺孔检测框中心 + 最小二乘拟合整圆圆心",
    response_model=ApiResponse,
)
async def hat_on_table_hole7_circle_fit(
    image: np.ndarray = Depends(parse_image),
    conf: float = Query(0.5, ge=0.0, le=1.0, description="检测置信度阈值 0~1"),
    manager: ModelManager = Depends(get_model_manager),
) -> ApiResponse:
    data = await _run_circle_fit(manager, "hat_on_table_hole7", image, conf, "hat7_circle_fit")
    return ApiResponse(task="hat7_circle_fit", data=data)


@router.post(
    "/mid_on_table_hole11_circle_fit",
    summary="mid_on_table_hole11：输出所有螺孔检测框中心 + 最小二乘拟合整圆圆心",
    response_model=ApiResponse,
)
async def mid_on_table_hole11_circle_fit(
    image: np.ndarray = Depends(parse_image),
    conf: float = Query(0.5, ge=0.0, le=1.0, description="检测置信度阈值 0~1"),
    manager: ModelManager = Depends(get_model_manager),
) -> ApiResponse:
    data = await _run_circle_fit(manager, "mid_on_table_hole11", image, conf, "mid11_circle_fit")
    return ApiResponse(task="mid11_circle_fit", data=data)


@router.post(
    "/mid_on_table_hole7_circle_fit",
    summary="mid_on_table_hole7：输出所有螺孔检测框中心 + 最小二乘拟合整圆圆心",
    response_model=ApiResponse,
)
async def mid_on_table_hole7_circle_fit(
    image: np.ndarray = Depends(parse_image),
    conf: float = Query(0.5, ge=0.0, le=1.0, description="检测置信度阈值 0~1"),
    manager: ModelManager = Depends(get_model_manager),
) -> ApiResponse:
    data = await _run_circle_fit(manager, "mid_on_table_hole7", image, conf, "mid7_circle_fit")
    return ApiResponse(task="mid7_circle_fit", data=data)


@router.post(
    "/base_hole_circle_fit",
    summary="base_hole：输出所有基座螺孔检测框中心 + 最小二乘拟合整圆圆心",
    response_model=ApiResponse,
)
async def base_hole_circle_fit(
    image: np.ndarray = Depends(parse_image),
    conf: float = Query(0.5, ge=0.0, le=1.0, description="检测置信度阈值 0~1"),
    manager: ModelManager = Depends(get_model_manager),
) -> ApiResponse:
    data = await _run_circle_fit(manager, "base_hole", image, conf,
                                 "base_hole_circle_fit", noun="基座螺孔")
    return ApiResponse(task="base_hole_circle_fit", data=data)
