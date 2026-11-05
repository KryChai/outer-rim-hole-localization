"""目标孔追踪接口：只返回检测结果中离图像中心最近的检测框中心。

与 ``tracking``（输出一圈孔的稳定编号）不同，这里是无状态接口：
每帧独立取离图像中心最近的检测框中心，不维护任何跨帧状态。
"""

from __future__ import annotations

import numpy as np
from fastapi import APIRouter, Depends, Query

from app.api.deps import get_model_manager, parse_image
from app.core import audit
from app.schemas.detection import ApiResponse
from app.services import detection as detection_service
from app.services.model_manager import ModelManager

router = APIRouter(prefix="/api", tags=["目标孔追踪"])


async def _run_nearest_hole(
    manager: ModelManager,
    key: str,
    image: np.ndarray,
    conf: float,
    task: str,
) -> dict:
    result = await manager.predict_single_async(key, image, conf)
    if result is None:
        return {"msg": "模型未返回检测结果", "hole": None}

    holes, _ = detection_service.parse_holes(result, conf)
    if not holes:
        return {"msg": "未检测到孔", "hole": None}

    img_cx, img_cy = image.shape[1] / 2.0, image.shape[0] / 2.0
    nearest = min(
        holes,
        key=lambda h: (h["center_x"] - img_cx) ** 2 + (h["center_y"] - img_cy) ** 2,
    )
    data = {
        "msg": "返回离图像中心最近的检测框中心",
        "hole": {
            "center_x": float(nearest["center_x"]),
            "center_y": float(nearest["center_y"]),
            "conf": round(float(nearest["conf"]), 3),
        },
    }
    audit.log_tracking(task, conf, None, data)
    return data


@router.post(
    "/usb_hole7_nearest_hole",
    summary="返回 usb_hole7 检测结果中离图像中心最近的检测框中心",
    response_model=ApiResponse,
)
async def usb_hole7_nearest_hole(
    image: np.ndarray = Depends(parse_image),
    conf: float = Query(0.5, ge=0.0, le=1.0, description="检测置信度阈值 0~1"),
    manager: ModelManager = Depends(get_model_manager),
) -> ApiResponse:
    data = await _run_nearest_hole(manager, "usb_hole7", image, conf, "usb_hole7_nearest_hole")
    return ApiResponse(task="usb_hole7_nearest_hole", data=data)


@router.post(
    "/usb_hole11_nearest_hole",
    summary="返回 usb_hole11 检测结果中离图像中心最近的检测框中心",
    response_model=ApiResponse,
)
async def usb_hole11_nearest_hole(
    image: np.ndarray = Depends(parse_image),
    conf: float = Query(0.5, ge=0.0, le=1.0, description="检测置信度阈值 0~1"),
    manager: ModelManager = Depends(get_model_manager),
) -> ApiResponse:
    data = await _run_nearest_hole(manager, "usb_hole11", image, conf, "usb_hole11_nearest_hole")
    return ApiResponse(task="usb_hole11_nearest_hole", data=data)
