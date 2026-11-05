"""FastAPI 依赖注入。"""

from __future__ import annotations

import numpy as np
from fastapi import File, Request, UploadFile

from app.core.exceptions import ServiceError
from app.services.model_manager import ModelManager
from app.utils.image import decode_image


async def parse_image(file: UploadFile = File(..., description="上传图片")) -> np.ndarray:
    """读取并解码上传图片，返回 BGR 图像。"""
    data = await file.read()
    try:
        return decode_image(data)
    except ValueError as exc:
        raise ServiceError(str(exc), status_code=400) from exc


def get_model_manager(request: Request) -> ModelManager:
    """从应用状态中获取模型管理器（由 lifespan 初始化）。"""
    return request.app.state.model_manager
