"""应用配置与模型注册表。

所有可配置项集中在 :class:`Settings`，模型文件统一由 :class:`ModelSpec`
描述，避免路径散落在业务代码中。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

# 项目根目录：app/core/config.py -> app -> 项目根目录
BASE_DIR = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = Path(os.getenv("MODELS_DIR", str(BASE_DIR / "models")))


@dataclass(frozen=True)
class Settings:
    app_name: str = "outer-rim-hole-localization"
    version: str = "3.0"
    host: str = os.getenv("HOST", "127.0.0.1")
    port: int = int(os.getenv("PORT", "8001"))
    default_conf: float = 0.5
    models_dir: Path = MODELS_DIR


settings = Settings()


@dataclass(frozen=True)
class ModelSpec:
    """单个 YOLO 模型的元信息。"""

    key: str
    filename: str

    @property
    def path(self) -> Path:
        return settings.models_dir / self.filename


# 模型注册表：key 为业务名称，filename 为权重文件名
MODEL_SPECS: tuple[ModelSpec, ...] = (
    ModelSpec("usb_hole7", "usb_hole7_detect.pt"),
    ModelSpec("usb_hole11", "usb_hole11_detect.pt"),
    ModelSpec("hat_on_table_hole7", "hat_on_table_hole7_detect.pt"),
    ModelSpec("mid_on_table_hole11", "mid_on_table_hole11_detect.pt"),
    ModelSpec("mid_on_table_hole7", "mid_on_table_hole7_detect.pt"),
    ModelSpec("hole7_mid_on_base", "hole7_mid_on_base.pt"),
    ModelSpec("base_hole", "base_hole_detect.pt"),
    ModelSpec("bolt7_judge", "bolt7_judge.pt"),
    ModelSpec("bolt11_judge", "bolt11_judge.pt"),
    ModelSpec("hole7_judge", "hole7_judge.pt"),
    ModelSpec("hole11_judge", "hole11_judge.pt"),
)

MODEL_REGISTRY: dict[str, ModelSpec] = {spec.key: spec for spec in MODEL_SPECS}
