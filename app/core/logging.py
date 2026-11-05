"""统一日志配置。"""

from __future__ import annotations

import logging
import sys

logger = logging.getLogger("yolo_service")


def setup_logging(level: int = logging.INFO) -> None:
    """初始化根日志器，输出到标准输出。"""
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
