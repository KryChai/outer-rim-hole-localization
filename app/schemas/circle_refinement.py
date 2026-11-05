"""Typed visualization contract, exposed in /docs and /openapi.json."""
from typing import Literal

from pydantic import BaseModel, Field


class Point(BaseModel):
    x: float
    y: float


class BoundingBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float


class CircleEquation(BaseModel):
    standard_form: str
    h: float
    k: float
    r_squared: float
    general_form: str
    D: float
    E: float
    F: float


class Circle(BaseModel):
    center_x: float
    center_y: float
    radius: float
    diameter: float
    equation: CircleEquation


class Visualization(BaseModel):
    outline_points_px: list[tuple[float, float]]
    edge_points_px: list[tuple[float, float]]


class Perturbation(BaseModel):
    dx: int
    dy: int
    center_shift_px: float | None
    reason: str


class FitQuality(BaseModel):
    valid: bool
    flags: list[str]
    reason: str
    stability_checked: bool
    rmse_px: float | None = None
    relative_rmse_radius: float | None = None
    angular_coverage: float | None = None
    inlier_fraction: float | None = None
    edge_point_count: int | None = None
    scale_center_shift_px: float | None = None
    candidate_center_spread_px: float | None = None
    ellipse_axis_ratio: float | None = None
    ellipse_center_shift_px: float | None = None
    candidate_count: int | None = None
    bbox_jitter_max_center_shift_px: float | None = None
    perturbations: list[Perturbation] = Field(default_factory=list)


class RefinedHole(BaseModel):
    detection_index: int
    confidence: float
    detection_center: Point
    detection_bbox: BoundingBox
    expanded_bbox: BoundingBox | None
    roi_bbox: BoundingBox | None
    status: Literal["ok", "review", "failed"]
    center: Point | None
    circle: Circle | None
    visualization: Visualization | None
    fit_quality: FitQuality


class ImageSize(BaseModel):
    width: int
    height: int


class Summary(BaseModel):
    detected: int
    ok: int
    review: int
    failed: int


class CircleRefinementData(BaseModel):
    algorithm: str
    center_definition: str
    coordinate_frame: str
    units: str
    image_size: ImageSize
    bbox_expansion_fraction: float
    sigma_px: float
    check_stability: bool
    reason: str
    summary: Summary
    holes: list[RefinedHole]


class CircleRefinementResponse(BaseModel):
    code: int = 200
    task: str
    data: CircleRefinementData
