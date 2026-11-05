"""孔位编号追踪接口。"""

from __future__ import annotations

import numpy as np
from fastapi import APIRouter, Depends, Query

from app.api.deps import get_model_manager, parse_image
from app.schemas.detection import ApiResponse
from app.services import detection as detection_service
from app.services.model_manager import ModelManager
from app.services.tracking import tracker_registry

router = APIRouter(prefix="/api", tags=["孔位编号追踪"])


async def _run_track(
    manager: ModelManager,
    key: str,
    image: np.ndarray,
    conf: float,
    session_id: str,
) -> dict:
    result = await manager.predict_single_async(key, image, conf)
    if result is None:
        return {"msg": "模型未返回检测结果", "holes": [], "circle": None}

    holes, point_set = detection_service.parse_holes(result, conf)
    if not holes:
        return {"msg": "未检测到孔", "holes": [], "circle": None}

    tracker = tracker_registry.get(key, session_id)

    circle = None
    if len(point_set) >= 3:
        center_x, center_y, radius = detection_service.fit_circle(point_set)
        if center_x is not None:
            circle = {
                "circle_center_x": center_x,
                "circle_center_y": center_y,
                "radius": radius,
            }

    if circle is not None:
        tracked_holes = tracker.update(holes, circle["circle_center_x"], circle["circle_center_y"])
        msg = f"共追踪 {len(tracked_holes)} 个孔位"
    else:
        # 孔数不足，无法拟合圆：只追踪离图像中心最近的孔并返回其编号
        tracked_holes = tracker.update_nearest(holes, image.shape[1], image.shape[0])
        msg = "孔数不足，仅追踪离图像中心最近的孔位"

    return {"msg": msg, "holes": tracked_holes, "circle": circle}


@router.post(
    "/usb_hole7_track",
    summary="检测一圈孔并输出稳定的孔位编号追踪",
    response_model=ApiResponse,
)
async def usb_hole7_track(
    image: np.ndarray = Depends(parse_image),
    conf: float = Query(0.5, ge=0.0, le=1.0, description="检测置信度阈值 0~1"),
    session_id: str = Query("default", description="工件/任务标识，不同工件使用不同 id"),
    manager: ModelManager = Depends(get_model_manager),
) -> ApiResponse:
    data = await _run_track(manager, "usb_hole7", image, conf, session_id)
    return ApiResponse(task="usb_hole7_track", data=data)


@router.post(
    "/usb_hole7_track/reset",
    summary="重置指定工件的孔位追踪状态（换新工件时调用）",
    response_model=ApiResponse,
)
async def usb_hole7_track_reset(
    session_id: str = Query("default", description="工件/任务标识"),
) -> ApiResponse:
    tracker_registry.reset("usb_hole7", session_id)
    return ApiResponse(
        task="usb_hole7_track_reset",
        data={"msg": "追踪状态已重置"},
    )


@router.post(
    "/usb_hole11_track",
    summary="检测一圈孔并输出稳定的孔位编号追踪",
    response_model=ApiResponse,
)
async def usb_hole11_track(
    image: np.ndarray = Depends(parse_image),
    conf: float = Query(0.5, ge=0.0, le=1.0, description="检测置信度阈值 0~1"),
    session_id: str = Query("default", description="工件/任务标识，不同工件使用不同 id"),
    manager: ModelManager = Depends(get_model_manager),
) -> ApiResponse:
    data = await _run_track(manager, "usb_hole11", image, conf, session_id)
    return ApiResponse(task="usb_hole11_track", data=data)


@router.post(
    "/usb_hole11_track/reset",
    summary="重置指定工件的孔位追踪状态（换新工件时调用）",
    response_model=ApiResponse,
)
async def usb_hole11_track_reset(
    session_id: str = Query("default", description="工件/任务标识"),
) -> ApiResponse:
    tracker_registry.reset("usb_hole11", session_id)
    return ApiResponse(
        task="usb_hole11_track_reset",
        data={"msg": "追踪状态已重置"},
    )
