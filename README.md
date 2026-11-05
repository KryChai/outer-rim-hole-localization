# Outer-Rim Hole Localization

A detector can locate the region of a machined hole, but its bounding-box center can move when the box changes. Inner steps, reflections and partial edges can also make an ellipse fit settle on the wrong boundary. This project refines the image-space center by finding the outer opening within each detection.

## Method

1. **Scope the search.** Use a YOLO candidate box to identify the hole and provide a region of interest. The refinement code also accepts a box supplied by another detector.
2. **Measure the contour.** Sample intensity gradients along ellipse normals with calipers. Use the supported edge locations to refine ellipse candidates and measure their residuals and angular coverage.
3. **Select the outer opening.** Gate candidates by their span relative to the detection and how much of the outline lies inside it. Choose the largest qualified ellipse by area. An inner rim or a box center cannot substitute for an unobserved outer rim.

The default outer-rim mode returns the fitted ellipse center in the uploaded image's pixel coordinates. It requires no camera intrinsics or inner rim. A separate coaxial mode uses camera calibration to estimate the projection of the physical front-circle center under perspective. The two modes return different geometric quantities.

## Reproduce with synthetic images

```sh
python -m pip install -r requirements-test.txt
python -m pytest -q
```

The tests use synthetic contours and mocked detector outputs. On `main`, start with `app/services/caliper.py`, `app/services/outer_rim.py`, and `tests/test_outer_rim.py` for the edge measurement, selection policy, and rejection cases.

To try the FastAPI service with your own detector weights, install `requirements-inference.txt`, put `usb_hole11_detect.pt` in a local `models/` directory, set `MODEL_KEYS=usb_hole11`, and run `python -m uvicorn app.main:app --host 127.0.0.1 --port 8001`. On `main`, the OpenAPI page at `http://127.0.0.1:8001/docs` shows `POST /api/usb_hole11_refine_center` with an uploaded image and a target pixel. `MODELS_DIR` can point to a different local weights directory. Refinement audit storage is opt-in via `REFINEMENT_LOG_DIR` on the outer-rim branch and `BUSINESS_LOG_ENABLED` on `dev`.

## Branches

This snapshot is based on `master`. Detection, target association, session-based tracking, and baseline circle geometry.

- [`codex/coarse-fine-servo`](https://github.com/KryChai/outer-rim-hole-localization/tree/codex/coarse-fine-servo) — Normal calipers, robust ellipse refinement and largest qualified outer-rim selection.
- [`codex/outer-circle-refinement`](https://github.com/KryChai/outer-rim-hole-localization/tree/codex/outer-circle-refinement) — Per-hole circle fitting with residual, coverage and ROI stability diagnostics.
- [`dev`](https://github.com/KryChai/outer-rim-hole-localization/tree/dev) — Per-hole circle and ellipse endpoints, with optional structured diagnostics.
- [`master`](https://github.com/KryChai/outer-rim-hole-localization/tree/master) — Detection, target association, session-based tracking, and baseline circle geometry.

The companion [coarse-to-fine-hole-ibvs project](https://github.com/KryChai/coarse-to-fine-hole-ibvs) uses a refined hole center as the measurement for visual servoing.

## 中文说明

孔的检测框中心会受框偏移影响，台阶孔的内圈也可能被误选成控制参考。这里先用检测框限定目标区域，再通过法向卡尺取边和鲁棒椭圆拟合生成候选。在满足跨度和框内覆盖条件的轮廓中，选择面积最大的外圈。外圈不可观测时，接口返回精定位失败及诊断原因。默认模式输出原图像素坐标中的外圈椭圆中心，适合用合成图像验证方法。
