"""Common contract of every adapter on a real frame (marked gpu by conftest)."""

import json
import subprocess
import sys

import numpy as np
import pytest

from det_depth_distance.core import registry
from det_depth_distance.core.labels import VOCABULARIES
from det_depth_distance.core.types import DepthResult, Instances


def test_detector_contract(detector_case, frame0, load_ctx):
    detector, vocabulary = detector_case
    detector.load(load_ctx)
    try:
        out = detector(frame0)  # the base class already validated the contract
        assert isinstance(out, Instances) and len(out) > 0
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
