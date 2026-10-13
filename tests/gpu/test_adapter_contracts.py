"""Common contract of every adapter on a real frame (marked gpu by conftest)."""

import json
import math
import subprocess
import sys

import numpy as np
import pytest

from det_depth_distance.core import registry
from det_depth_distance.core.labels import VOCABULARIES
from det_depth_distance.core.types import DepthResult, Instances


def test_detector_contract(detector_case, frame0, load_ctx, segmenter_names):
    detector, vocabulary = detector_case
    detector.load(load_ctx)
    try:
        out = detector(frame0)  # the base class already validated the contract
        assert isinstance(out, Instances) and len(out) > 0
        # Detection weights keep task_0002's output: no masks.
        assert (out.masks is not None) == (detector.name in segmenter_names)
        assert set(out.source) == {detector.name}
        assert detector.vocabulary == vocabulary
        preset = next(v for v in VOCABULARIES if v.name == vocabulary)
        assert set(out.raw_label) <= set(preset.presets["driving"])
        json.dumps(detector.describe())
    finally:
        detector.close()
    assert not detector.is_loaded
    detector.load(load_ctx)  # reload after close
    try:
        assert len(detector(frame0)) == len(out)
    finally:
        detector.close()


def test_rfdetr_and_da3_do_not_import_ultralytics(video, hf_cache):
    # Fresh interpreter: other tests of this session import ultralytics.
    code = f"""
import sys
from pathlib import Path
from det_depth_distance.config import SourceConfig
from det_depth_distance.core import registry
from det_depth_distance.core.components import LoadContext
from det_depth_distance.core.types import CameraIntrinsics
from det_depth_distance.io.source import VideoSource

source = VideoSource(SourceConfig(video=Path({str(video)!r}), max_frames=1))
frame = next(iter(source))
ctx = LoadContext("cuda:0", source.frame_size, CameraIntrinsics.centered(1390.0, *source.frame_size))
specs = [("detector", {{"type": "rfdetr", "variant": "nano"}}),
         ("depth", {{"type": "da3", "hf_cache": {str(hf_cache)!r}}})]
for kind, spec in specs:
    component = registry.build(kind, spec, path=kind)
    component.load(ctx)
    component(frame)
    component.close()
assert "ultralytics" not in sys.modules
"""
    out = subprocess.run(
        [sys.executable, "-c", code], check=False, capture_output=True, text=True
    )
    assert out.returncode == 0, out.stderr[-2000:]


def test_yolo_flips_rgb_to_bgr(frame0, load_ctx, yolo_weights):
    weights = yolo_weights / "yolo26l-objv1-150.pt"
    if not weights.is_file():
        pytest.skip(f"missing weights {weights}")
    spec = {
        "type": "yolo",
        "weights": str(weights),
        "threshold": 0.25,
        "classes": "all",
    }
    detector = registry.build("detector", spec, path="detector")
    detector.load(load_ctx)
    try:
        out = detector(frame0)
        model, names = detector._model, detector.class_names
        kwargs = {
            "conf": 0.25,
            "imgsz": 640,
            "device": load_ctx.device,
            "verbose": False,
        }
        bgr = model.predict(np.ascontiguousarray(frame0.rgb[..., ::-1]), **kwargs)[0]
        rgb = model.predict(frame0.rgb, **kwargs)[0]
    finally:
        detector.close()
    assert np.array_equal(out.xyxy, bgr.boxes.xyxy.cpu().numpy())
    assert np.array_equal(out.score, bgr.boxes.conf.cpu().numpy())
    # Feeding the RGB frame as-is changes the detections (Fact 2: labels change).
    rgb_labels = tuple(names[int(c)] for c in rgb.boxes.cls.cpu().numpy())
    assert rgb_labels != out.raw_label or not np.array_equal(
        out.xyxy, rgb.boxes.xyxy.cpu().numpy()
    )


def test_da3_metric_contract(da3_metric, frame0, load_ctx):
    da3_metric.load(load_ctx)
    try:
        out = da3_metric(frame0)
        g = da3_metric.geometry
        assert isinstance(out, DepthResult) and out.geometry == g
        assert out.depth_m.shape == (g.map_h, g.map_w)
        assert out.depth_m.dtype == np.float32
        assert 1.0 < float(np.median(out.depth_m)) < 200.0
        assert out.sky is None or out.sky.dtype == bool
        assert out.intrinsics == da3_metric.fixed_intrinsics
        json.dumps(da3_metric.describe())
    finally:
        da3_metric.close()


def test_da3_nested_intrinsics_convention(frame0, load_ctx, hf_cache):
    """Open check (task_0002, Passo 8): is the predicted principal point continuous
    (cx = W/2) or OpenCV ((W - 1)/2)? Needs a NESTED checkpoint in the cache."""
    nested = sorted(hf_cache.glob("models--depth-anything--DA3NESTED-*"))
    if not nested:
        pytest.skip(
            f"no DA3NESTED checkpoint in {hf_cache} (not downloaded on purpose)"
        )
    repo_id = nested[0].name.split("--", 1)[1].replace("--", "/")
    spec = {"type": "da3", "model": repo_id, "hf_cache": str(hf_cache)}
    depth = registry.build("depth", spec, path="depth")
    depth.load(load_ctx)
    try:
        k = depth(frame0).intrinsics
        g = depth.geometry
    finally:
        depth.close()
    print(f"cx={k.cx} cy={k.cy} map={g.map_w}x{g.map_h}")
    assert abs(k.cx - g.map_w / 2) < abs(k.cx - (g.map_w - 1) / 2)


def _frame_window(box) -> tuple[int, int, int, int]:
    """Frame pixels [c0, c1) x [r0, r1) kept by ultralytics' crop_mask: pixel
    index (not centre) in [x0, x1) x [y0, y1) (utils/ops.py crop_mask)."""
    x0, y0, x1, y1 = (float(v) for v in box)
    return math.ceil(x0), math.ceil(y0), math.ceil(x1), math.ceil(y1)


def _bbox_iou(box, mask: np.ndarray) -> float:
    """IoU between a box and the bounding rectangle of its mask (frame pixels)."""
    rows, cols = np.flatnonzero(mask.any(axis=1)), np.flatnonzero(mask.any(axis=0))
    rect = (cols[0], rows[0], cols[-1] + 1, rows[-1] + 1)
    x0, y0, x1, y1 = (float(v) for v in box)
    iw = max(0.0, min(x1, rect[2]) - max(x0, rect[0]))
    ih = max(0.0, min(y1, rect[3]) - max(y0, rect[1]))
    inter = iw * ih
    union = (x1 - x0) * (y1 - y0) + (rect[2] - rect[0]) * (rect[3] - rect[1]) - inter
    return inter / union


def test_segmenter_masks(segmenter, frame0, load_ctx):
    """Masks at frame resolution, aligned with their boxes (catches a wrong
    letterbox or scale), and not empty."""
    segmenter.load(load_ctx)
    try:
        out = segmenter(frame0)
        desc = segmenter.describe()
    finally:
        segmenter.close()
    assert len(out) > 0
    assert out.masks.shape == (len(out), frame0.height, frame0.width)
    assert out.masks.dtype == bool
    assert out.has_mask().mean() >= 0.9
    used = out.masks.any(axis=(1, 2))
    inside = []
    for box, mask in zip(out.xyxy[used], out.masks[used], strict=True):
        c0, r0, c1, r1 = _frame_window(box)
        inside.append(mask[r0:r1, c0:c1].sum() / mask.sum())
    ious = [
        _bbox_iou(b, m) for b, m in zip(out.xyxy[used], out.masks[used], strict=True)
    ]
    if segmenter.type_name == "yolo":
        assert min(inside) == 1.0  # ultralytics crops masks to their boxes
        assert desc["task"] == "segment"
    else:
        assert np.median(inside) >= 0.95  # RF-DETR masks are not cropped
        assert desc["segmentation"] is True
        assert desc["mask_resolution"] == desc["resolution"] // 4
    assert np.median(ious) >= 0.7
    json.dumps(desc)


def _mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else 1.0


def test_yolo_seg_masks_match_ultralytics(frame0, load_ctx, yolo_weights):
    """The adapter's masks equal predict(retina_masks=True), and are close to the
    letterbox-resolution masks rescaled independently with ops.scale_masks."""
    weights = yolo_weights / "yolo26n-seg.pt"
    if not weights.is_file():
        pytest.skip(f"missing weights {weights}")
    import torch
    from ultralytics.utils import ops

    spec = {
        "type": "yolo",
        "weights": str(weights),
        "threshold": 0.25,
        "classes": "all",
    }
    detector = registry.build("detector", spec, path="detector")
    detector.load(load_ctx)
    try:
        out = detector(frame0)
        kwargs = {
            "conf": 0.25,
            "imgsz": 640,
            "device": load_ctx.device,
            "verbose": False,
        }
        bgr = np.ascontiguousarray(frame0.rgb[..., ::-1])
        retina = detector._model.predict(bgr, retina_masks=True, **kwargs)[0]
        native = detector._model.predict(bgr, **kwargs)[0]
    finally:
        detector.close()
    np.testing.assert_array_equal(out.masks, retina.masks.data.bool().cpu().numpy())
    assert len(native.masks.data) == len(out)
    with torch.no_grad():
        rescaled = ops.scale_masks(
            native.masks.data[None].float(), (frame0.height, frame0.width)
        )[0]
    rescaled = (rescaled > 0.5).cpu().numpy()
    ious = [_mask_iou(a, b) for a, b in zip(out.masks, rescaled, strict=True)]
    assert np.median(ious) >= 0.9
