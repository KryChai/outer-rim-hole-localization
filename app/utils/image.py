"""图像处理工具函数。"""

from __future__ import annotations

import cv2
import numpy as np


def decode_image(data: bytes) -> np.ndarray:
    """将上传的图片字节流解码为 BGR 图像。

    解码失败时抛出 :class:`ValueError`，由上层转换为统一错误响应。
    """
    array = np.frombuffer(data, np.uint8)
    image = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("无法解析图片")
    return image
