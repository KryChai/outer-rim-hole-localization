"""Pure geometry and response compatibility; no weights or hardware required."""
import importlib.util
from pathlib import Path
import numpy as np
import pytest

module_path = Path(__file__).resolve().parents[1] / "app" / "services" / "detection.py"
spec = importlib.util.spec_from_file_location("server_detection_geometry", module_path)
geometry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(geometry)


@pytest.mark.parametrize("center,radius", [((0, 0), 10), ((100, 100), 10),
                                          ((1e6, -1e6), 200)])
def test_radius_and_center(center, radius):
    angles = np.arange(8)*2*np.pi/8
    points = np.column_stack([center[0]+radius*np.cos(angles),
                              center[1]+radius*np.sin(angles)])
    cx, cy, r = geometry.fit_circle(points.tolist())
    assert [cx, cy, r] == pytest.approx([*center, radius], abs=1e-6)


@pytest.mark.parametrize("points", [
    [[1, 1], [1, 1], [1, 1]], [[0, 0], [1, 1], [2, 2]],
    [[0, 0], [1, float("nan")], [2, 2]], [[0, 0], [1, 1]],
])
def test_invalid_circles_are_none(points):
    assert geometry.fit_circle(points) == (None, None, None)


def test_excessive_residual_is_reported():
    angles = np.arange(12)*2*np.pi/12
    points = np.column_stack([10*np.cos(angles), 10*np.sin(angles)])
    points[0] = [25, 0]
    fit, quality = geometry.circle_fit_quality(points.tolist())
    assert fit == (None, None, None)
    assert quality["reason"] == "excessive_residual"


def test_response_retains_holes_when_circle_cannot_be_fitted(monkeypatch):
    monkeypatch.setattr(geometry, "parse_holes",
                        lambda *args: ([{"conf": .9, "center_x": 1, "center_y": 2}], [[1, 2]]))
    data = geometry.build_circle_fit_data(object(), .5)
    assert data["all_holes"][0]["center_x"] == 1
    assert data["fit_circle"] is None
    assert data["fit_quality"]["valid"] is False
