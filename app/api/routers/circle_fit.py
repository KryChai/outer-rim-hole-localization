"""全部螺孔中心 + 最小二乘拟合圆 / 透视纠正圆心接口。"""

from __future__ import annotations

import numpy as np
from fastapi import APIRouter, Depends, Query

from app.api.deps import get_model_manager, parse_image
from app.schemas.detection import ApiResponse
from app.services import detection as detection_service
from app.services.model_manager import ModelManager

router = APIRouter(prefix="/api", tags=["拟合圆检测"])


async def _run_circle_fit(
    manager: ModelManager,
    key: str,
    image: np.ndarray,
    conf: float,
    noun: str = "螺孔",
) -> dict:
    result = await manager.predict_single_async(key, image, conf)
    return detection_service.build_circle_fit_data(result, conf, noun=noun)


async def _run_center_fit(
    manager: ModelManager,
    key: str,
    image: np.ndarray,
    conf: float,
    n_total: int,
) -> dict:
    result = await manager.predict_single_async(key, image, conf)
    return detection_service.build_center_fit_data(result, conf, n_total)


@router.post(
    "/usb_hole7_circle_fit",
    summary="usb_hole7：输出所有螺孔中心 + 最小二乘拟合整圆圆心",
    response_model=ApiResponse,
)
async def usb_hole7_circle_fit(
    image: np.ndarray = Depends(parse_image),
    conf: float = Query(0.5, ge=0.0, le=1.0, description="检测置信度阈值 0~1"),
    manager: ModelManager = Depends(get_model_manager),
) -> ApiResponse:
    data = await _run_circle_fit(manager, "usb_hole7", image, conf)
    return ApiResponse(task="hole7_circle_fit", data=data)


@router.post(
    "/hole7_mid_on_base_circle_fit",
    summary="hole7_mid_on_base：输出所有螺孔中心 + 最小二乘拟合整圆圆心",
    response_model=ApiResponse,
)
async def hole7_mid_on_base_circle_fit(
    image: np.ndarray = Depends(parse_image),
    conf: float = Query(0.5, ge=0.0, le=1.0, description="检测置信度阈值 0~1"),
    manager: ModelManager = Depends(get_model_manager),
) -> ApiResponse:
    data = await _run_circle_fit(manager, "hole7_mid_on_base", image, conf)
    return ApiResponse(task="hole7_mid_on_base_circle_fit", data=data)


@router.post(
    "/usb_hole11_circle_fit",
    summary="usb_hole11：输出所有螺孔中心 + 最小二乘拟合整圆圆心",
    response_model=ApiResponse,
)
async def usb_hole11_circle_fit(
    image: np.ndarray = Depends(parse_image),
    conf: float = Query(0.5, ge=0.0, le=1.0, description="检测置信度阈值 0~1"),
    manager: ModelManager = Depends(get_model_manager),
) -> ApiResponse:
    data = await _run_circle_fit(manager, "usb_hole11", image, conf)
    return ApiResponse(task="hole11_circle_fit", data=data)


@router.post(
    "/usb_hole7_center",
    summary="旧孔圈接口：7 型工件的 12 孔圈，近景请用 refine_center",
    deprecated=True,
    response_model=ApiResponse,
)
async def usb_hole7_center(
    image: np.ndarray = Depends(parse_image),
    conf: float = Query(0.5, ge=0.0, le=1.0, description="检测置信度阈值 0~1"),
    manager: ModelManager = Depends(get_model_manager),
) -> ApiResponse:
    data = await _run_center_fit(manager, "usb_hole7", image, conf, n_total=12)
    return ApiResponse(task="hole7_center", data=data)


@router.post(
    "/usb_hole11_center",
    summary="旧孔圈接口：11 型工件的 8 孔圈，近景请用 refine_center",
    deprecated=True,
    response_model=ApiResponse,
)
async def usb_hole11_center(
    image: np.ndarray = Depends(parse_image),
    conf: float = Query(0.5, ge=0.0, le=1.0, description="检测置信度阈值 0~1"),
    manager: ModelManager = Depends(get_model_manager),
) -> ApiResponse:
    data = await _run_center_fit(manager, "usb_hole11", image, conf, n_total=8)
    return ApiResponse(task="hole11_center", data=data)


@router.post(
    "/hat_on_table_hole7_circle_fit",
    summary="hat_on_table_hole7：输出所有螺孔中心 + 最小二乘拟合整圆圆心",
    response_model=ApiResponse,
)
async def hat_on_table_hole7_circle_fit(
    image: np.ndarray = Depends(parse_image),
    conf: float = Query(0.5, ge=0.0, le=1.0, description="检测置信度阈值 0~1"),
    manager: ModelManager = Depends(get_model_manager),
) -> ApiResponse:
    data = await _run_circle_fit(manager, "hat_on_table_hole7", image, conf)
    return ApiResponse(task="hat7_circle_fit", data=data)


@router.post(
    "/mid_on_table_hole11_circle_fit",
    summary="mid_on_table_hole11：输出所有螺孔中心 + 最小二乘拟合整圆圆心",
    response_model=ApiResponse,
)
async def mid_on_table_hole11_circle_fit(
    image: np.ndarray = Depends(parse_image),
    conf: float = Query(0.5, ge=0.0, le=1.0, description="检测置信度阈值 0~1"),
    manager: ModelManager = Depends(get_model_manager),
) -> ApiResponse:
    data = await _run_circle_fit(manager, "mid_on_table_hole11", image, conf)
    return ApiResponse(task="mid11_circle_fit", data=data)


@router.post(
    "/mid_on_table_hole7_circle_fit",
    summary="mid_on_table_hole7：输出所有螺孔中心 + 最小二乘拟合整圆圆心",
    response_model=ApiResponse,
)
async def mid_on_table_hole7_circle_fit(
    image: np.ndarray = Depends(parse_image),
    conf: float = Query(0.5, ge=0.0, le=1.0, description="检测置信度阈值 0~1"),
    manager: ModelManager = Depends(get_model_manager),
) -> ApiResponse:
    data = await _run_circle_fit(manager, "mid_on_table_hole7", image, conf)
    return ApiResponse(task="mid7_circle_fit", data=data)


@router.post(
    "/base_hole_circle_fit",
    summary="base_hole：输出所有基座螺孔中心 + 最小二乘拟合整圆圆心",
    response_model=ApiResponse,
)
async def base_hole_circle_fit(
    image: np.ndarray = Depends(parse_image),
    conf: float = Query(0.5, ge=0.0, le=1.0, description="检测置信度阈值 0~1"),
    manager: ModelManager = Depends(get_model_manager),
) -> ApiResponse:
    data = await _run_circle_fit(manager, "base_hole", image, conf, noun="基座螺孔")
    return ApiResponse(task="base_hole_circle_fit", data=data)
