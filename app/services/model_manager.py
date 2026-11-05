"""YOLO 模型生命周期管理与推理。"""

from __future__ import annotations

import asyncio
import os

import numpy as np

from app.core.config import MODEL_REGISTRY
from app.core.exceptions import ServiceError
from app.core.logging import logger


class ModelManager:
    """集中加载与复用显式启用的 YOLO 模型。

    模型在应用启动时（lifespan）一次性加载，并通过 :meth:`predict` 提供
    统一的推理入口；推理为 CPU/GPU 阻塞操作，因此额外提供异步封装。
    """

    def __init__(self) -> None:
        self._models: dict[str, YOLO] = {}

    def load(self, key: str) -> None:
        spec = MODEL_REGISTRY[key]
        path = spec.path
        if not path.exists():
            raise FileNotFoundError(f"模型文件不存在：{path}")
        from ultralytics import YOLO

        self._models[key] = YOLO(str(path))
        logger.info("模型加载成功：%s -> %s", key, path.name)

    def load_all(self) -> None:
        keys = os.getenv("MODEL_KEYS", "").split(",")
        for key in dict.fromkeys(key.strip() for key in keys if key.strip()):
            self.load(key)

    def predict(self, key: str, image: np.ndarray, conf: float):
        if key not in self._models:
            raise ServiceError(f"Model is not configured: {key}", 503)
        return self._models[key].predict(source=image, conf=conf)

    def predict_single(self, key: str, image: np.ndarray, conf: float):
        """返回首个（也是唯一一个）推理结果，无结果时返回 None。"""
        results = self.predict(key, image, conf)
        return results[0] if results else None

    async def predict_single_async(self, key: str, image: np.ndarray, conf: float):
        return await asyncio.to_thread(self.predict_single, key, image, conf)
