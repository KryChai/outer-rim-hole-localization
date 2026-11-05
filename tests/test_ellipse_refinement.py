"""椭圆外圈精定位的几何一致性、接口契约与真实 multipart 路由测试。"""
import json
import sys
import types

import cv2
import numpy as np
import pytest

from app.services.ellipse_refinement import (
    build_ellipse_refinement_data, ellipse_parameters, fit_ellipse_box, refine_hole,
)
from app.services.circle_refinement_core import expanded_box


def synthetic_ellipse(inner=True):
    # 面积积分后的解析椭圆盘，独立于拟合实现。
    center = np.array([177.35, 142.65])
    a, b, phi = 95.4, 70.6, np.deg2rad(25.)
    yy, xx = np.mgrid[:300*4, :360*4].astype(float)/4-.375
    u = (xx-center[0])*np.cos(phi)+(yy-center[1])*np.sin(phi)
    v = -(xx-center[0])*np.sin(phi)+(yy-center[1])*np.cos(phi)
    q = (u/a)**2+(v/b)**2
    gray = np.where(q <= 1, 70., 195.)
    if inner:
        gray[(q < .55) & (q > .4)] = 220
        gray[np.hypot(xx-center[0]-12, yy-center[1]+8) < 40] = 10
    gray = cv2.resize(gray, (360, 300), interpolation=cv2.INTER_AREA)
    image = cv2.cvtColor(gray.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    ex = np.hypot(a*np.cos(phi), b*np.sin(phi))
    ey = np.hypot(a*np.sin(phi), b*np.cos(phi))
    bbox = [center[0]-ex-2, center[1]-ey-2, center[0]+ex+2, center[1]+ey+2]
    hole = dict(zip(("x1", "y1", "x2", "y2"), bbox))
    hole.update(conf=.95, center_x=center[0], center_y=center[1])
    return image, hole, center, np.array([a, b, phi])


def test_conic_and_derived_parameters_are_consistent():
    cx, cy, a, b, phi = 177.35, 142.65, 95.4, 70.6, np.deg2rad(25.)
    params = ellipse_parameters(cx, cy, a, b, phi)
    outline = np.array([[p["x"], p["y"]] for p in params["major_axis_endpoints_px"]])
    assert np.linalg.norm(outline[0]-outline[1]) == pytest.approx(2*a)
    # 两个焦点到椭圆上任意一点的距离和恒为 2a。
    points = build_outline(cx, cy, a, b, phi)
    focus_1 = np.array([params["foci_px"][0]["x"], params["foci_px"][0]["y"]])
    focus_2 = np.array([params["foci_px"][1]["x"], params["foci_px"][1]["y"]])
    distance_sum = (np.linalg.norm(points-focus_1, axis=1)
                    + np.linalg.norm(points-focus_2, axis=1))
    assert distance_sum == pytest.approx(2*a, abs=1e-9)
    assert params["eccentricity"] == pytest.approx(np.sqrt(1-(b/a)**2))
    assert params["linear_eccentricity_px"] == pytest.approx(np.sqrt(a*a-b*b))
    assert params["focal_parameter_px"] == pytest.approx(b*b/a)
    assert params["axis_ratio"] == pytest.approx(b/a)
    assert params["area_px2"] == pytest.approx(np.pi*a*b)
    assert params["perimeter_px"] > 2*np.pi*min(a, b)
    assert 0 < params["rotation_angle_deg"] < 180
    assert params["equation"]["theta_deg"] == params["rotation_angle_deg"]
    for point in build_outline(cx, cy, a, b, phi):
        u = (point[0]-cx)*np.cos(phi)+(point[1]-cy)*np.sin(phi)
        v = -(point[0]-cx)*np.sin(phi)+(point[1]-cy)*np.cos(phi)
        assert (u/a)**2+(v/b)**2 == pytest.approx(1, abs=1e-9)


def build_outline(cx, cy, a, b, phi):
    theta = np.linspace(0, 2*np.pi, 181)
    rr = a*b/np.sqrt((b*np.cos(theta-phi))**2+(a*np.sin(theta-phi))**2)
    return np.column_stack((cx+rr*np.cos(theta), cy+rr*np.sin(theta)))


@pytest.mark.parametrize("inner", [True, False])
def test_outer_ellipse_parameters_and_original_pixel_geometry(inner):
    image, hole, center, truth = synthetic_ellipse(inner)
    result = refine_hole(image, hole, 0, check_stability=False)
    assert result["status"] == "review", result
    assert result["fit_quality"]["flags"] == ["stability_not_checked"]
    ellipse = result["ellipse"]
    assert [ellipse["center_x"], ellipse["center_y"]] == pytest.approx(center, abs=.3)
    assert [ellipse["semi_major_axis_px"], ellipse["semi_minor_axis_px"]] == pytest.approx(
        truth[:2], rel=.01)
    assert ellipse["rotation_angle_deg"] == pytest.approx(np.rad2deg(truth[2]), abs=.5)
    assert ellipse["eccentricity"] == pytest.approx(np.sqrt(1-(truth[1]/truth[0])**2), rel=.01)
    assert ellipse["axis_ratio"] == pytest.approx(truth[1]/truth[0], rel=.01)
    # 圆锥曲线方程对圆周点成立。
    equation = ellipse["equation"]
    points = np.array(result["visualization"]["outline_points_px"])
    values = (equation["A"]*points[:, 0]**2+equation["B"]*points[:, 0]*points[:, 1]
              + equation["C"]*points[:, 1]**2+equation["D"]*points[:, 0]
              + equation["E"]*points[:, 1]+equation["F"])
    scale = (np.abs(equation["A"]*points[:, 0]**2)+np.abs(equation["B"]*points[:, 0]*points[:, 1])
             + np.abs(equation["C"]*points[:, 1]**2)+np.abs(equation["D"]*points[:, 0])
             + np.abs(equation["E"]*points[:, 1])+abs(equation["F"]))
    assert np.max(np.abs(values)) < 1e-9*np.max(scale)
    edges = np.array(result["visualization"]["edge_points_px"])
    roi = result["roi_bbox"]
    assert ((edges >= [roi["x1"], roi["y1"]]) & (edges < [roi["x2"], roi["y2"]])).all()
    json.dumps(result, allow_nan=False)


def test_default_stability_and_roi_expansion():
    image, hole, center, _ = synthetic_ellipse()
    box = [hole[k] for k in ("x1", "y1", "x2", "y2")]
    result = refine_hole(image, hole, 0)
    assert result["status"] == "ok", result["fit_quality"]
    assert len(result["fit_quality"]["perturbations"]) == 4
    assert result["fit_quality"]["bbox_jitter_max_center_shift_px"] < 1
    assert result["fit_quality"]["bbox_jitter_max_axis_ratio_change"] < .03
    expanded = result["expanded_bbox"]
    assert expanded["x2"]-expanded["x1"] == pytest.approx((box[2]-box[0])*1.10)
    assert (expanded["x1"]+expanded["x2"])/2 == pytest.approx(center[0])
    assert expanded_box([0, 0, 100, 80], (80, 100, 3), .05) == [0, 0, 100, 80]
    # ROI 外的像素变化不影响拟合结果。
    other = image.copy()
    other[:30] = 0
    assert refine_hole(other, hole, 0, False)["ellipse"] == result["ellipse"]


def test_mixed_success_failure_and_multiple_holes():
    image, hole, _, _ = synthetic_ellipse()
    wide = np.full((300, 720, 3), 195, np.uint8)
    wide[:, :360] = image
    wide[:, 360:] = image
    second = {**hole, "x1": hole["x1"]+360, "x2": hole["x2"]+360,
              "center_x": hole["center_x"]+360}
    blank = {"x1": 5, "y1": 5, "x2": 45, "y2": 45, "center_x": 25, "center_y": 25, "conf": .8}
    result = build_ellipse_refinement_data(wide, [hole, second, blank], False)
    assert result["summary"] == {"detected": 3, "ok": 0, "review": 2, "failed": 1}
    assert (result["holes"][1]["ellipse"]["center_x"]
            - result["holes"][0]["ellipse"]["center_x"]) == pytest.approx(360)
    failed = result["holes"][2]
    assert failed["ellipse"] is None and failed["center"] is None
    assert failed["visualization"] is None
    assert not failed["fit_quality"]["valid"]


@pytest.mark.parametrize("box", [[5, 5, 6, 6], [1000, 5, 1050, 55], [5, 5, float("nan"), 50]])
def test_invalid_roi_rejected(box):
    with pytest.raises(ValueError):
        fit_ellipse_box(np.full((100, 100, 3), 128, np.uint8), box)


class Tensor:
    def __init__(self, value): self.value = np.asarray(value)
    def cpu(self): return self
    def numpy(self): return self.value
    def item(self): return self.value.item()


def fake_result(hole):
    return types.SimpleNamespace(boxes=[types.SimpleNamespace(cls=Tensor(0), conf=Tensor(hole["conf"]),
        xyxy=Tensor([[hole[k] for k in ("x1", "y1", "x2", "y2")]]))])


@pytest.fixture
def api(monkeypatch):
    # 只替换 YOLO 边界；解码、拟合和响应校验都真实执行。
    stub = types.ModuleType("ultralytics")
    stub.YOLO = object
    monkeypatch.setitem(sys.modules, "ultralytics", stub)
    from fastapi.testclient import TestClient
    from app.main import create_app
    from app.services.model_manager import ModelManager
    image, hole, center, _ = synthetic_ellipse()
    state = {"result": fake_result(hole), "calls": []}
    async def predict(self, key, frame, conf):
        state["calls"].append((key, conf))
        return state["result"]
    monkeypatch.setattr(ModelManager, "load_all", lambda self: None)
    monkeypatch.setattr(ModelManager, "predict_single_async", predict)
    _, encoded = cv2.imencode(".png", image)
    with TestClient(create_app()) as client:
        yield client, encoded.tobytes(), state, center


@pytest.mark.parametrize("model", ["usb_hole7", "usb_hole11"])
def test_multipart_model_selection_and_openapi(api, model):
    client, encoded, state, center = api
    response = client.post("/api/"+model+"_ellipse_refine",
                           files={"file": ("frame.png", encoded, "image/png")})
    assert response.status_code == 200, response.text
    data = response.json()["data"]
    assert data["summary"]["ok"] == 1
    assert data["algorithm"] == "polar_radial_algebraic_ellipse_v1"
    assert data["image_size"] == {"width": 360, "height": 300}
    assert data["holes"][0]["ellipse"]["center_x"] == pytest.approx(center[0], abs=.3)
    assert data["holes"][0]["ellipse"]["eccentricity"] > .5
    assert state["calls"] == [(model, .5)]
    spec = client.get("/openapi.json").json()
    assert "Ellipse" in spec["components"]["schemas"]
    assert "EllipseEquation" in spec["components"]["schemas"]
    assert "/api/usb_hole7_ellipse_refine" in spec["paths"]
    assert "/api/usb_hole11_ellipse_refine" in spec["paths"]


@pytest.mark.parametrize("result, reason", [(None, "no_result"), (types.SimpleNamespace(boxes=[]), "no_holes")])
def test_empty_results_and_bad_upload(api, result, reason):
    client, encoded, state, _ = api
    state["result"] = result
    response = client.post("/api/usb_hole11_ellipse_refine", files={"file": ("frame.png", encoded)})
    assert response.status_code == 200
    assert response.json()["data"]["holes"] == []
    assert response.json()["data"]["reason"] == reason
    assert client.post("/api/usb_hole11_ellipse_refine",
                       files={"file": ("bad.png", b"invalid")}).status_code == 400
    assert client.post("/api/usb_hole11_ellipse_refine", params={"conf": 2},
                       files={"file": ("frame.png", encoded)}).status_code == 422


def test_review_and_failure_response_contract(api):
    client, encoded, state, _ = api
    route = "/api/usb_hole11_ellipse_refine"
    response = client.post(route, params={"check_stability": False},
                           files={"file": ("frame.png", encoded)})
    hole = response.json()["data"]["holes"][0]
    assert hole["status"] == "review" and not hole["fit_quality"]["valid"]
    assert hole["ellipse"] is not None
    assert hole["fit_quality"]["stability_checked"] is False
    blank = np.full((300, 360, 3), 195, np.uint8)
    _, blank_encoded = cv2.imencode(".png", blank)
    response = client.post(route, files={"file": ("frame.png", blank_encoded.tobytes())})
    assert response.status_code == 200, response.text
    data = response.json()["data"]
    assert data["summary"]["failed"] == 1
    hole = data["holes"][0]
    assert hole["center"] is None and hole["ellipse"] is None and hole["visualization"] is None
    assert hole["fit_quality"]["rmse_px"] is None
