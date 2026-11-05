"""统一日志配置：stdout 与按大小轮转的文件双输出。"""

from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

from app.core.config import settings

logger = logging.getLogger("yolo_service")

LOG_FILE_MAX_BYTES = 10 * 1024 * 1024  # 单文件 10MB
LOG_FILE_BACKUP_COUNT = 5  # 最多保留 5 个历史文件


def setup_logging(level: int = logging.INFO) -> None:
    """初始化根日志器，同时输出到标准输出与轮转文件。"""
    log_dir = Path(settings.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    file_handler = RotatingFileHandler(
        log_dir / "yolo_service.log",
        maxBytes=LOG_FILE_MAX_BYTES,
        backupCount=LOG_FILE_BACKUP_COUNT,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)

    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()
    root.addHandler(stream_handler)
    root.addHandler(file_handler)
