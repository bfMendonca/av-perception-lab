"""Common contract of every adapter on a real frame (marked gpu by conftest)."""

import dataclasses
import json
import math
import subprocess
import sys

import numpy as np
import pytest

from av_perception_lab.core import registry
from av_perception_lab.core.aggregate import Concat
from av_perception_lab.core.labels import VOCABULARIES
from av_perception_lab.core.pipeline import Pipeline, SetupInfo
from av_perception_lab.core.types import DepthResult, Instances
from av_perception_lab.distance import (
    DistanceConfig,
    box_depth_stats,
    map_eligibility,
    resample_mask,
)


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
from av_perception_lab.config import SourceConfig
from av_perception_lab.core import registry
from av_perception_lab.core.components import LoadContext
from av_perception_lab.core.types import CameraIntrinsics
from av_perception_lab.io.source import VideoSource

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
# task_0005: the refiner/segmenter adapters import nothing at module import
# (transformers itself is imported by rfdetr's DINOv2 backbone, so it is not
# checked here; the light-env test_registry covers our modules).
assert not {{"model", "timm"}} & set(sys.modules), sorted(sys.modules)
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


# task_0005: one contract test for every mask refiner (LSP), then the semantic
# segmenters alone and combined in one pipeline.


def _with_options(component, **changes):
    """Swap a loaded component's (frozen) options for one call path, without
    loading the model again; restored by the caller."""
    previous = component.options
    component.options = dataclasses.replace(previous, **changes)
    return previous


def test_refiner_contract(refiner, rfdetr_boxes, frame0, frame_ctx, yolo_weights):
    inst = rfdetr_boxes
    assert len(inst) > 0 and inst.masks is None  # RF-DETR detection: boxes only
    out = refiner(frame0, inst)
    n, h, w = len(inst), frame0.height, frame0.width
    assert out.masks.shape == (n, h, w) and out.masks.dtype == bool
    has = out.has_mask()
    assert has.mean() >= 0.9
    used = out.masks.any(axis=(1, 2))
    inside = []
    for box, mask in zip(out.xyxy[used], out.masks[used], strict=True):
        c0, r0, c1, r1 = _frame_window(box)
        inside.append(mask[r0:r1, c0:c1].sum() / mask.sum())
    assert np.median(inside) >= 0.95
    ious = [
        _bbox_iou(b, m) for b, m in zip(out.xyxy[used], out.masks[used], strict=True)
    ]
    assert np.median(ious) >= 0.7  # same thresholds as the native masks (task_0004)
    refined = np.array([s == refiner.name for s in out.mask_source])
    assert refined.any() and np.isfinite(out.mask_score[refined]).all()
    if refiner.has_object_score:  # SAM 2 convention: sigmoid IoU head
        assert ((out.mask_score[refined] >= 0) & (out.mask_score[refined] <= 1)).all()
    # Deterministic, and batching does not change the masks.
    again = refiner(frame0, inst)
    np.testing.assert_array_equal(again.masks, out.masks)
    previous = _with_options(refiner, max_boxes=1)
    try:
        one_by_one = refiner(frame0, inst)
    finally:
        refiner.options = previous
    exact = np.array_equal(one_by_one.masks, out.masks)
    print(f"{refiner.name}: max_boxes 1 vs 32 identical: {exact}")
    if not exact:  # GPU kernels may vary with the batch size: near-identical
        assert (
            min(_mask_iou(a, b) for a, b in zip(one_by_one.masks, out.masks)) >= 0.999
        )
    if refiner.has_object_score:
        previous = _with_options(refiner, score="object", min_object_score=0.5)
        try:
            filtered = refiner(frame0, inst)
        finally:
            refiner.options = previous
        kept = np.array([s == refiner.name for s in filtered.mask_source])
        assert (filtered.mask_score[kept] >= 0.5).all()
    json.dumps(refiner.describe())
    # Against YOLO26-seg native masks, refining every instance (targets: all).
    weights = yolo_weights / "yolo26n-seg.pt"
    if weights.is_file():
        spec = {"type": "yolo", "weights": str(weights), "threshold": 0.25}
        seg = registry.build("detector", spec, path="detector")
        seg.load(frame_ctx)
        try:
            native = seg(frame0)
        finally:
            seg.close()
        previous = _with_options(refiner, targets="all")
        try:
            versus = refiner(frame0, native)
        finally:
            refiner.options = previous
        pairs = [
            _mask_iou(a, b)
            for a, b, keep in zip(
                native.masks, versus.masks, native.has_mask(), strict=True
            )
            if keep
        ]
        print(f"{refiner.name}: median IoU vs yolo26n-seg {np.median(pairs):.3f}")
        assert np.median(pairs) >= 0.7
    # close() and reload give the same masks.
    refiner.close()
    assert not refiner.is_loaded
    refiner.load(frame_ctx)
    np.testing.assert_array_equal(refiner(frame0, inst).masks, out.masks)


def test_sam2_keeps_the_planned_defaults(refiner):
    if refiner.type_name != "sam2":
        pytest.skip("sam2 only")
    assert refiner.options.model == "facebook/sam2.1-hiera-large"
    assert refiner.name == "sam2.1-hiera-large"
    assert refiner.describe()["model_class"] == "Sam2Model"


CITYSCAPES = (
    "road",
    "sidewalk",
    "building",
    "wall",
    "fence",
    "pole",
    "traffic light",
    "traffic sign",
    "vegetation",
    "terrain",
    "sky",
    "person",
    "rider",
    "car",
    "truck",
    "bus",
    "train",
    "motorcycle",
    "bicycle",
)


def test_segformer_contract(segformer, da3_metric, frame0, load_ctx):
    segformer.load(load_ctx)
    try:
        out = segformer(frame0)
        json.dumps(segformer.describe())
    finally:
        segformer.close()
    assert out.labels.shape == (1080, 1920) and out.labels.dtype == np.int32
    assert out.class_names == CITYSCAPES
    road, sky = CITYSCAPES.index("road"), CITYSCAPES.index("sky")
    assert (out.labels[720:, 640:1280] == road).any()  # lower third, centre
    assert (out.labels[:360] == sky).any()  # upper third
    da3_metric.load(load_ctx)
    try:
        depth = da3_metric(frame0)
    finally:
        da3_metric.close()
    if depth.sky is None:
        pytest.skip("depth model has no sky head")
    g = depth.geometry
    sky_map = resample_mask(out.labels == sky, g, (0, 0, g.map_w, g.map_h))
    iou = _mask_iou(sky_map, depth.sky)
    print(f"SegFormer sky vs DA3 sky IoU {iou:.3f}")
    assert iou >= 0.5  # heuristic cross-model coherence


def _twinlitenet_reference(frame_rgb, repo, net, device):
    """The repo's demo path: letterbox_for_img on BGR, BGR->RGB, /255, padding crop,
    interpolate(scale_factor=int(1/ratio)), max over channels."""
    import torch

    sys.path.insert(0, str(repo))
    try:
        from demoDataset import letterbox_for_img
    finally:
        sys.path.remove(str(repo))
    bgr = np.ascontiguousarray(frame_rgb[..., ::-1])
    img, ratio, (pad_w, pad_h) = letterbox_for_img(bgr, new_shape=640, auto=True)
    img = np.ascontiguousarray(img[:, :, ::-1].transpose(2, 0, 1))
    x = torch.from_numpy(img)[None].to(device).float() / 255.0
    _, _, height, width = x.shape
    pad_w, pad_h = int(pad_w), int(pad_h)
    heads = []
    with torch.inference_mode():
        for out in net(x):
            crop = out[:, :, pad_h : height - pad_h, pad_w : width - pad_w]
            up = torch.nn.functional.interpolate(
                crop, scale_factor=int(1 / ratio[0]), mode="bilinear"
            )
            heads.append(torch.max(up, 1)[1][0].cpu().numpy())
    return heads


def test_twinlitenet_matches_the_reference(twinlitenet, frame0, load_ctx):
    from av_perception_lab.adapters.semantic.twinlitenet import _merge_heads

    twinlitenet.load(load_ctx)
    try:
        out = twinlitenet(frame0)
        reference = _twinlitenet_reference(
            frame0.rgb,
            twinlitenet.options.repo.expanduser(),
            twinlitenet._net,
            load_ctx.device,
        )
        json.dumps(twinlitenet.describe())
    finally:
        twinlitenet.close()
    assert out.labels.shape == (1080, 1920)
    np.testing.assert_array_equal(out.labels, _merge_heads(*reference))
    fractions = out.fractions()
    print(f"TwinLiteNet+ frame 0: {fractions}")


def test_two_segmenters_in_one_pipeline(
    twinlitenet, segformer, da3_metric, source, frame0, load_ctx
):
    alone = {}
    for segmenter in (twinlitenet, segformer):
        segmenter.load(load_ctx)
        try:
            alone[segmenter.name] = segmenter(frame0)
        finally:
            segmenter.close()
    detector = registry.build(
        "detector", {"type": "rfdetr", "variant": "nano"}, path="d"
    )

    def run(exclude):
        pipeline = Pipeline(
            [detector],
            da3_metric,
            Concat(Concat.Options()),
            DistanceConfig(),
            device=load_ctx.device,
            semantic=[twinlitenet, segformer],
            exclude_semantic=exclude,
        )
        with pipeline:
            pipeline.load(source, SetupInfo(load_ctx.intrinsics, 1390.0, "test"))
            return pipeline.process(frame0)

    both = run(("scene.sky", "lanes.lane"))
    assert list(both.semantic) == ["lanes", "scene"]
    for name, m in both.semantic.items():
        np.testing.assert_array_equal(m.labels, alone[name].labels)
    assert {"semantic.lanes", "semantic.scene", "semantic"} <= set(both.timings_ms)
    sky_only, lane_only = run(("scene.sky",)), run(("lanes.lane",))
    for i, stats in enumerate(both.stats):
        assert stats.n_valid <= min(
            sky_only.stats[i].n_valid, lane_only.stats[i].n_valid
        )
    scene, lanes = both.semantic["scene"], both.semantic["lanes"]
    expected = map_eligibility(
        both.depth.geometry,
        sky=both.depth.sky,
        exclusions=[
            (lanes.labels, [lanes.class_names.index("lane")]),
            (scene.labels, [scene.class_names.index("sky")]),
        ],
    )
    depth = both.depth
    union = tuple(
        box_depth_stats(
            depth.depth_m,
            box,
            depth.geometry,
            DistanceConfig(),
            intrinsics=depth.intrinsics,
            mask=expected,
        )
        for box in both.instances.xyxy
    )
    assert both.stats == union  # eligibility = ~sky & ~(lane | scene sky)
