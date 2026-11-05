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


class _FakeTensor:
    """最小化模拟 torch.Tensor：支持 cpu().item() / cpu().numpy()。"""

    def __init__(self, values):
        self._values = np.atleast_1d(np.asarray(values, dtype=float))

    def cpu(self):
        return self

    def item(self):
        return float(self._values[0])

    def numpy(self):
        return self._values.reshape(1, -1)


class _FakeBox:
    def __init__(self, cls, conf, xyxy):
        self.cls = _FakeTensor(cls)
        self.conf = _FakeTensor(conf)
        self.xyxy = _FakeTensor(xyxy)


class _FakeResult:
    def __init__(self, boxes):
        self.boxes = boxes


def test_parse_holes_returns_bbox_coordinates():
    result = _FakeResult(
        [
            _FakeBox(0, 0.9, [10, 20, 30, 40]),
            _FakeBox(1, 0.8, [1, 2, 3, 4]),      # 类别不符
            _FakeBox(0, 0.2, [50, 60, 70, 80]),  # 置信度不足
        ]
    )
    holes, point_set = geometry.parse_holes(result, conf=0.5)
    assert len(holes) == 1
    hole = holes[0]
    assert hole["center_x"] == pytest.approx(20)
    assert hole["center_y"] == pytest.approx(30)
    assert hole["x1"] == pytest.approx(10)
    assert hole["y1"] == pytest.approx(20)
    assert hole["x2"] == pytest.approx(30)
    assert hole["y2"] == pytest.approx(40)
    assert point_set == [[pytest.approx(20), pytest.approx(30)]]


def test_build_best_hole_data_keeps_bbox(monkeypatch):
    hole = {"conf": .9, "center_x": 1, "center_y": 2,
            "x1": 0, "y1": 1, "x2": 2, "y2": 3}
    monkeypatch.setattr(geometry, "parse_holes", lambda *args: ([hole], [[1, 2]]))
    data = geometry.build_best_hole_data(object(), .5)
    assert data["max_conf_hole"] == hole


# ---------------------------------------------------------------------------
# 透视纠正圆心（usb_hole7_center / usb_hole11_center 核心逻辑）
# ---------------------------------------------------------------------------

# 已知「标准等分圆 -> 图像」单应（含透视项 0.15x - 0.10y）：
# 真实圆心像 = H 第三列 = (330, 250)，孔中心点由该单应投影生成
_T = np.deg2rad(30)
_PERSPECTIVE_H = np.array(
    [
        [150 * np.cos(_T), -95 * np.sin(_T), 330],
        [150 * np.sin(_T), 95 * np.cos(_T), 250],
        [0.15, -0.10, 1.0],
    ]
)


def _projected_holes(n_total, drop=None, noise=None, seed=0, slot_jitter=None):
    """用 _PERSPECTIVE_H 把标准等分圆投影成图像点。

    可选：drop 删除一个孔（漏检）、noise 加高斯噪声、
    slot_jitter 在标准空间扰动各 slot 的角度（点仍落在投影圆锥曲线上）。
    """
    k = np.arange(n_total)
    angles = 2 * np.pi * k / n_total
    if slot_jitter is not None:
        angles = angles + slot_jitter(k)
    pattern = np.column_stack([np.cos(angles), np.sin(angles), np.ones(n_total)])
    uv = (_PERSPECTIVE_H @ pattern.T).T
    uv = uv[:, :2] / uv[:, 2:3]
    if drop is not None:
        uv = np.delete(uv, drop, axis=0)
    if noise:
        uv = uv + np.random.default_rng(seed).normal(0.0, noise, uv.shape)
    return uv.tolist()


@pytest.mark.parametrize("n_total", [7, 11])
def test_center_recovered_from_perspective_projection(n_total):
    points = _projected_holes(n_total)
    out = geometry.corrected_circle_center(points, n_total)
    assert out["method"] == "homography"
    assert out["center"]["center_x"] == pytest.approx(330.0, abs=0.5)
    assert out["center"]["center_y"] == pytest.approx(250.0, abs=0.5)
    # 椭圆中心与真实圆心像存在透视偏差（纠正的意义所在）
    ellipse = out["ellipse"]
    assert np.hypot(ellipse["center_x"] - 330.0, ellipse["center_y"] - 250.0) > 2.0
    assert ellipse["semi_major"] >= ellipse["semi_minor"] > 0
    assert 0.0 <= ellipse["angle_deg"] < 180.0
    # correction = 单应圆心 - 椭圆中心
    assert out["correction"]["dx"] == pytest.approx(330.0 - ellipse["center_x"], abs=1e-6)
    assert out["correction"]["dy"] == pytest.approx(250.0 - ellipse["center_y"], abs=1e-6)
    quality = out["quality"]
    assert quality["valid"] is True
    assert quality["point_count"] == n_total
    assert quality["expected_count"] == n_total
    assert quality["homography_residual_rms_px"] < quality["residual_gate_px"]


def test_center_robust_to_detection_noise():
    points = _projected_holes(7, noise=1.0, seed=2)
    out = geometry.corrected_circle_center(points, 7)
    assert out["method"] == "homography"
    assert out["center"]["center_x"] == pytest.approx(330.0, abs=2.0)
    assert out["center"]["center_y"] == pytest.approx(250.0, abs=2.0)
    assert out["quality"]["valid"] is True


@pytest.mark.parametrize("drop", [0, 3, 6])
def test_center_with_one_missing_hole(drop):
    """漏检一孔时通过角度间隙定位缺孔（gap-aware），仍精确恢复圆心。"""
    points = _projected_holes(7, drop=drop)
    out = geometry.corrected_circle_center(points, 7)
    assert out["method"] == "homography"
    assert out["center"]["center_x"] == pytest.approx(330.0, abs=0.5)
    assert out["center"]["center_y"] == pytest.approx(250.0, abs=0.5)
    assert out["quality"]["point_count"] == 6
    assert out["quality"]["expected_count"] == 7


def test_non_equidistant_layout_falls_back_to_ellipse():
    """slot2 角度偏移 0.25 rad：点仍全在投影圆锥曲线上（椭圆拟合无残差），
    但等分布局被破坏 -> 单应残差超标，回退椭圆中心。"""

    def slot_jitter(k):
        jitter = np.zeros_like(k, dtype=float)
        jitter[2] = 0.25
        return jitter

    points = _projected_holes(7, slot_jitter=slot_jitter)
    out = geometry.corrected_circle_center(points, 7)
    assert out["method"] == "ellipse"
    assert out["center"]["method"] == "ellipse"
    assert out["correction"] is None
    quality = out["quality"]
    # 椭圆回退下 center 仍可用（valid=True），reason 说明单应精化未通过
    assert quality["valid"] is True
    assert quality["reason"] == "homography_residual_exceeded"
    assert quality["homography_residual_rms_px"] > quality["residual_gate_px"]


@pytest.mark.parametrize(
    "points",
    [
        [[float(i), float(i)] for i in range(7)],          # 严格共线
        [[float(i), 0.02 * float(i)] for i in range(7)],  # 近共线
    ],
)
def test_collinear_points_rejected(points):
    """共线/近共线点会拟合出极端扁平的退化椭圆，须被扁平率门限拦截。"""
    out = geometry.corrected_circle_center(points, 7)
    assert out["method"] == "none"
    assert out["center"] is None
    assert out["quality"]["reason"] == "degenerate_flat_ellipse"


def test_scattered_points_rejected():
    rng = np.random.default_rng(0)
    points = rng.uniform(0, 200, (8, 2)).tolist()
    out = geometry.corrected_circle_center(points, 7)
    assert out["method"] == "none"
    assert out["quality"]["reason"] == "excessive_ellipse_residual"


@pytest.mark.parametrize(
    "points",
    [
        [],
        [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]],
    ],
)
def test_too_few_points_rejected(points):
    out = geometry.corrected_circle_center(points, 7)
    assert out["method"] == "none"
    assert out["quality"]["reason"] == "fewer_than_five_points"


def test_build_center_fit_data_structure(monkeypatch):
    points = _projected_holes(7)
    holes = [
        {
            "conf": 0.9,
            "center_x": p[0],
            "center_y": p[1],
            "x1": p[0] - 5,
            "y1": p[1] - 5,
            "x2": p[0] + 5,
            "y2": p[1] + 5,
        }
        for p in points
    ]
    monkeypatch.setattr(geometry, "parse_holes", lambda *args: (holes, points))
    data = geometry.build_center_fit_data(object(), 0.5, 7)
    assert data["msg"] == "共检测到 7 个符合阈值螺孔"
    assert len(data["all_holes"]) == 7
    assert data["all_holes"][3]["x2"] == pytest.approx(points[3][0] + 5)
    assert data["center"]["method"] == "homography"
    assert data["center"]["center_x"] == pytest.approx(330.0, abs=0.5)
    assert data["center"]["center_y"] == pytest.approx(250.0, abs=0.5)
    assert data["fit_quality"]["valid"] is True
    assert data["fit_quality"]["expected_count"] == 7


def test_build_center_fit_data_empty_paths(monkeypatch):
    assert geometry.build_center_fit_data(None, 0.5, 7)["fit_quality"]["reason"] == "no_result"
    monkeypatch.setattr(geometry, "parse_holes", lambda *args: ([], []))
    data = geometry.build_center_fit_data(object(), 0.5, 7)
    assert data["center"] is None
    assert data["fit_quality"]["reason"] == "no_holes"
