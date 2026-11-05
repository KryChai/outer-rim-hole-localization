"""椭圆精定位接口的类型契约，暴露在 /docs 与 /openapi.json。"""
from typing import Literal

from pydantic import BaseModel, Field

from app.schemas.circle_refinement import BoundingBox, ImageSize, Point, Summary


class EllipseEquation(BaseModel):
    general_form: str
    A: float
    B: float
    C: float
    D: float
    E: float
    F: float
    canonical_form: str
    standard_form: str
    h: float
    k: float
    a: float
    b: float
    theta_deg: float
    theta_rad: float


class Ellipse(BaseModel):
    center_x: float
    center_y: float
    semi_major_axis_px: float
    semi_minor_axis_px: float
    major_axis_length_px: float
    minor_axis_length_px: float
    axis_ratio: float
    eccentricity: float
    linear_eccentricity_px: float
    focal_parameter_px: float
    flattening: float
    rotation_angle_deg: float
    rotation_angle_rad: float
    foci_px: list[Point]
    major_axis_endpoints_px: list[Point]
    minor_axis_endpoints_px: list[Point]
    area_px2: float
    perimeter_px: float
    equation: EllipseEquation


class EllipseVisualization(BaseModel):
    outline_points_px: list[tuple[float, float]]
    edge_points_px: list[tuple[float, float]]


class EllipsePerturbation(BaseModel):
    dx: int
    dy: int
    center_shift_px: float | None
    axis_ratio_change: float | None
    eccentricity_change: float | None
    reason: str


class EllipseFitQuality(BaseModel):
    valid: bool
    flags: list[str]
    reason: str
    stability_checked: bool
    rmse_px: float | None = None
    relative_rmse_major_axis: float | None = None
    angular_coverage: float | None = None
    inlier_fraction: float | None = None
    edge_point_count: int | None = None
    semi_major_axis_px: float | None = None
    semi_minor_axis_px: float | None = None
    axis_ratio: float | None = None
    eccentricity: float | None = None
    rotation_angle_deg: float | None = None
    scale_center_shift_px: float | None = None
    scale_axis_ratio_change: float | None = None
    scale_eccentricity_change: float | None = None
    candidate_count: int | None = None
    bbox_jitter_max_center_shift_px: float | None = None
    bbox_jitter_max_axis_ratio_change: float | None = None
    perturbations: list[EllipsePerturbation] = Field(default_factory=list)


class RefinedEllipseHole(BaseModel):
    detection_index: int
    confidence: float
    detection_center: Point
    detection_bbox: BoundingBox
    expanded_bbox: BoundingBox | None
    roi_bbox: BoundingBox | None
    status: Literal["ok", "review", "failed"]
    center: Point | None
    ellipse: Ellipse | None
    visualization: EllipseVisualization | None
    fit_quality: EllipseFitQuality


class EllipseRefinementData(BaseModel):
    algorithm: str
    center_definition: str
    coordinate_frame: str
    units: str
    image_size: ImageSize
    bbox_expansion_fraction: float
    sigma_px: float
    check_stability: bool
    fit_parameters: dict[str, float | int | str]
    reason: str
    summary: Summary
    holes: list[RefinedEllipseHole]


class EllipseRefinementResponse(BaseModel):
    code: int = 200
    task: str
    data: EllipseRefinementData
