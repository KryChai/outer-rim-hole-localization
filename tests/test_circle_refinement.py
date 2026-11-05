"""Known pixel geometry and real multipart routes, without model weights."""
import json
import sys
import types

import cv2
import numpy as np
import pytest

from app.services.circle_refinement import build_circle_refinement_data, fit_box, refine_hole
from app.services.circle_refinement_core import expanded_box


def synthetic(inner=True):
    # Area-integrated analytic disk: independent of the fitting implementation.
    center = np.array([177.35, 142.65])
    radius = 90.4
    yy, xx = np.mgrid[:300*4, :360*4].astype(float)/4-.375
    r = np.hypot(xx-center[0], yy-center[1])
    gray = np.where(r <= radius, 70., 195.)
    if inner:
        gray[(r < radius*.78) & (r > radius*.65)] = 220
        gray[np.hypot(xx-center[0]-12, yy-center[1]+8) < 40] = 10
    gray = cv2.resize(gray, (360, 300), interpolation=cv2.INTER_AREA)
    image = cv2.cvtColor(gray.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    bbox = [center[0]-radius-2, center[1]-radius-2,
            center[0]+radius+2, center[1]+radius+2]
    hole = dict(zip(("x1", "y1", "x2", "y2"), bbox))
    hole.update(conf=.95, center_x=center[0], center_y=center[1])
    return image, hole, center, radius


@pytest.mark.parametrize("inner", [True, False])
def test_outer_center_equations_and_original_pixel_geometry(inner):
    image, hole, center, radius = synthetic(inner)
    result = refine_hole(image, hole, 0, check_stability=False)
    assert result["status"] == "review", result
    assert result["fit_quality"]["flags"] == ["stability_not_checked"]
    circle = result["circle"]
    assert [circle["center_x"], circle["center_y"], circle["radius"]] == pytest.approx(
        [*center, radius], abs=.3)
    points = np.array(result["visualization"]["outline_points_px"])
    assert points[0] == pytest.approx(points[-1])
    equation = circle["equation"]
    general = (points**2).sum(axis=1)+equation["D"]*points[:, 0]+equation["E"]*points[:, 1]+equation["F"]
    assert np.max(np.abs(general)) < 1e-8
    assert equation["h"] == circle["center_x"]
    assert equation["k"] == circle["center_y"]
    assert equation["r_squared"] == pytest.approx(radius**2, rel=.01)
    edges = np.array(result["visualization"]["edge_points_px"])
    roi = result["roi_bbox"]
    assert ((edges >= [roi["x1"], roi["y1"]]) & (edges < [roi["x2"], roi["y2"]])).all()
    measured_rmse = np.sqrt(np.mean((np.linalg.norm(edges-[circle["center_x"], circle["center_y"]], axis=1)-circle["radius"])**2))
    assert measured_rmse == pytest.approx(result["fit_quality"]["rmse_px"], abs=1e-9)
    json.dumps(result, allow_nan=False)


def test_default_stability_and_roi_expansion():
    image, hole, center, radius = synthetic()
    box = [hole[k] for k in ("x1", "y1", "x2", "y2")]
    result = refine_hole(image, hole, 0)
    assert result["status"] == "ok", result["fit_quality"]
    assert result["fit_quality"]["valid"]
    assert len(result["fit_quality"]["perturbations"]) == 4
    assert result["fit_quality"]["bbox_jitter_max_center_shift_px"] < 1
    expanded = result["expanded_bbox"]
    assert expanded["x2"]-expanded["x1"] == pytest.approx((box[2]-box[0])*1.10)
    assert (expanded["x1"]+expanded["x2"])/2 == pytest.approx(center[0])
    assert expanded_box([0, 0, 100, 80], (80, 100, 3), .05) == [0, 0, 100, 80]
    # An unrelated image change outside the expanded ROI cannot move the fit.
    other = image.copy()
    other[:30] = 0
    result2 = refine_hole(other, hole, 0, False)
    assert result2["circle"] == result["circle"]


def test_mixed_success_failure_and_multiple_holes():
    image, hole, center, radius = synthetic()
    wide = np.full((300, 720, 3), 195, np.uint8)
    wide[:, :360] = image
    wide[:, 360:] = image
    second = {**hole, "x1": hole["x1"]+360, "x2": hole["x2"]+360,
              "center_x": hole["center_x"]+360}
    blank = {"x1": 5, "y1": 5, "x2": 45, "y2": 45, "center_x": 25, "center_y": 25, "conf": .8}
    result = build_circle_refinement_data(wide, [hole, second, blank], False)
    assert result["summary"] == {"detected": 3, "ok": 0, "review": 2, "failed": 1}
    assert result["holes"][1]["center"]["x"]-result["holes"][0]["center"]["x"] == pytest.approx(360)
    failed = result["holes"][2]
    assert failed["circle"] is None and failed["center"] is None
    assert failed["visualization"] is None
    assert not failed["fit_quality"]["valid"]


@pytest.mark.parametrize("box", [[5, 5, 6, 6], [1000, 5, 1050, 55], [5, 5, float("nan"), 50]])
def test_invalid_roi_rejected(box):
    with pytest.raises(ValueError):
        fit_box(np.full((100, 100, 3), 128, np.uint8), box)


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
    # Only the YOLO boundary is mocked; decode, fitting and response validation run.
    stub = types.ModuleType("ultralytics")
    stub.YOLO = object
    monkeypatch.setitem(sys.modules, "ultralytics", stub)
    from fastapi.testclient import TestClient
    from app.main import create_app
    from app.services.model_manager import ModelManager
    image, hole, center, radius = synthetic()
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
    response = client.post("/api/"+model+"_circle_refine", files={"file": ("frame.png", encoded, "image/png")})
    assert response.status_code == 200, response.text
    data = response.json()["data"]
    assert data["summary"]["ok"] == 1
    assert data["image_size"] == {"width": 360, "height": 300}
    assert data["holes"][0]["center"]["x"] == pytest.approx(center[0], abs=.3)
    assert state["calls"] == [(model, .5)]
    spec = client.get("/openapi.json").json()
    assert "CircleEquation" in spec["components"]["schemas"]
    # The replaced dev whole-pattern center endpoints are not reintroduced.
    assert "/api/usb_hole7_center" not in spec["paths"]


@pytest.mark.parametrize("result, reason", [(None, "no_result"), (types.SimpleNamespace(boxes=[]), "no_holes")])
def test_empty_results_and_bad_upload(api, result, reason):
    client, encoded, state, _ = api
    state["result"] = result
    response = client.post("/api/usb_hole11_circle_refine", files={"file": ("frame.png", encoded)})
    assert response.status_code == 200
    assert response.json()["data"]["holes"] == []
    assert response.json()["data"]["reason"] == reason
    assert client.post("/api/usb_hole11_circle_refine", files={"file": ("bad.png", b"invalid")}).status_code == 400
    assert client.post("/api/usb_hole11_circle_refine", params={"conf": 2}, files={"file": ("frame.png", encoded)}).status_code == 422


def test_detection_boxes_and_confidence_filter(api):
    client, encoded, state, _ = api
    response = client.post("/api/usb_hole11_detect", files={"file": ("frame.png", encoded)})
    assert "x1" in response.json()["data"]["max_conf_hole"]
    response = client.post("/api/usb_hole11_circle_refine", params={"conf": .99}, files={"file": ("frame.png", encoded)})
    assert response.json()["data"]["reason"] == "no_holes"


def test_review_and_failure_response_contract(api):
    client, encoded, state, _ = api
    route = "/api/usb_hole11_circle_refine"
    response = client.post(route, params={"check_stability": False}, files={"file": ("frame.png", encoded)})
    hole = response.json()["data"]["holes"][0]
    assert hole["status"] == "review" and not hole["fit_quality"]["valid"]
    assert hole["circle"] is not None
    assert hole["fit_quality"]["stability_checked"] is False
    blank = np.full((300, 360, 3), 195, np.uint8)
    _, blank_encoded = cv2.imencode(".png", blank)
    response = client.post(route, files={"file": ("frame.png", blank_encoded.tobytes())})
    assert response.status_code == 200, response.text
    data = response.json()["data"]
    assert data["summary"]["failed"] == 1
    hole = data["holes"][0]
    assert hole["center"] is None and hole["circle"] is None and hole["visualization"] is None
    assert hole["fit_quality"]["rmse_px"] is None
