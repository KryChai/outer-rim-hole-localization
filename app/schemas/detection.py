"""接口请求/响应数据模型。"""

from __future__ import annotations

from typing import Any, List, Optional

from pydantic import BaseModel, Field


class Hole(BaseModel):
    """单个螺孔检测结果。"""

    conf: float = Field(..., description="置信度")
    center_x: float = Field(..., description="检测框中心 X 坐标")
    center_y: float = Field(..., description="检测框中心 Y 坐标")
    x1: float = Field(..., description="检测框左上角 X")
    y1: float = Field(..., description="检测框左上角 Y")
    x2: float = Field(..., description="检测框右下角 X")
    y2: float = Field(..., description="检测框右下角 Y")


class CircleFit(BaseModel):
    """最小二乘拟合圆结果。"""

    circle_center_x: float
    circle_center_y: float
    radius: float


class Detection(BaseModel):
    """通用目标检测结果（用于拧紧/对准判断）。"""

    class_id: int
    class_name: str
    confidence: float
    center_x: float
    center_y: float


class ApiResponse(BaseModel):
    """统一响应外壳。"""

    code: int = 200
    task: str
    data: Any = None


class HealthResponse(BaseModel):
    status: str = "ok"
