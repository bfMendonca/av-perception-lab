"""Deterministic, torch-free test doubles that read the synthetic scene from
the decoded frame (grey level -> class), so they follow motion and lens
undistortion like real models would.

SceneDetector (largest connected component per object class, optional
masks, COCO raw labels), SceneRefiner (BoxPromptRefiner: object pixels in
the box), SceneSegmenter (semantic map) and SceneDA3Net, which replaces
only the network of the real DA3Depth adapter: it returns
`z * 300 / mean(fx, fy)` at map resolution for the focal the oracle says
the image has, the exact inverse of the DA3METRIC scaling. None of them is
registered at import: `install_scene_fakes` inserts them with
monkeypatch.setitem into the registry and patches `cli.DeviceMonitor`,
`da3._load_da3` and `DA3Depth._close` (the only torch imports on the run
path); pytest restores everything at teardown.
"""

import math
from dataclasses import dataclass
from types import SimpleNamespace
from typing import ClassVar, Literal

import cv2
import numpy as np
import pytest
from smoke_scene import SCENE, SceneSpec, classify

from av_perception_lab import cli
from av_perception_lab.adapters.depth import da3
from av_perception_lab.adapters.refiners.box_prompt import (
    BoxPromptOptions,
    BoxPromptRefiner,
    DecodedMasks,
)
from av_perception_lab.core import registry
from av_perception_lab.core.components import (
    Detector,
    DetectorOptions,
    LoadContext,
    SemanticSegmenter,
)
from av_perception_lab.core.labels import COCO
from av_perception_lab.core.requirements import Requirements
from av_perception_lab.core.types import (
    CameraIntrinsics,
    Instances,
    MapGeometry,
    SemanticMap,
)
from av_perception_lab.distance import resample_mask


def _box_window(box: np.ndarray, w: int, h: int) -> tuple[int, int, int, int]:
    """Pixels whose centre lies in the box (the rule of Instances.has_mask)."""
    x0, y0, x1, y1 = (float(v) for v in box)
    c0 = min(max(math.ceil(x0 - 0.5), 0), w)
    r0 = min(max(math.ceil(y0 - 0.5), 0), h)
    c1 = min(max(math.floor(x1 - 0.5) + 1, c0), w)
    r1 = min(max(math.floor(y1 - 0.5) + 1, r0), h)
    return c0, r0, c1, r1


class SceneDetector(Detector):
    """One instance per scene object: the bounding box of the largest connected
    component of its class (robust to the mixed grey levels a bilinear
    undistortion leaves on edges), padded by `pad_px` and clipped."""

    type_name = "scene_det"

    @dataclass(frozen=True, slots=True, kw_only=True)
    class Options(DetectorOptions):
        masks: bool = False  # return the component as the instance mask
        pad_px: int = 0
        score: float = 0.9
        vehicle_label: Literal["truck", "bus"] = "truck"  # raw label of the truck

    # The COCO "driving" preset: vocabulary "coco", so presets and label maps work.
    CLASS_NAMES: ClassVar[tuple[str, ...]] = COCO.presets["driving"]
    options: Options

    def _load(self, ctx: LoadContext) -> None:
        """Nothing to load."""

    @property
    def class_names(self) -> list[str]:
        return list(self.CLASS_NAMES)

    def _predict(self, rgb: np.ndarray) -> Instances:
        labels = classify(rgb, SCENE)
        h, w = labels.shape
        opts, pad = self.options, self.options.pad_px
        boxes, names, masks = [], [], []
        for obj in SCENE.objects:
            n, components, stats, _ = cv2.connectedComponentsWithStats(
                (labels == SCENE.index(obj.cls)).astype(np.uint8), connectivity=8
            )
            if n < 2:  # background only: the object left the frame
                continue
            k = 1 + int(stats[1:, cv2.CC_STAT_AREA].argmax())
            x, y, bw, bh = stats[k, :4]
            boxes.append(
                [
                    max(x - pad, 0),
                    max(y - pad, 0),
                    min(x + bw + pad, w),
                    min(y + bh + pad, h),
                ]
            )
            names.append(
                opts.vehicle_label if obj.raw_label == "truck" else obj.raw_label
            )
            masks.append(components == k)
        return self._instances(
            np.asarray(boxes, dtype=np.float32).reshape(-1, 4),
            np.full(len(names), opts.score, dtype=np.float32),
            [self.CLASS_NAMES.index(n) for n in names],
            names,
            np.asarray(masks, dtype=bool).reshape(-1, h, w) if opts.masks else None,
        )


class SceneRefiner(BoxPromptRefiner):
    """Box prompt -> the object-class pixels inside the box (one candidate,
    predicted IoU 0.95); runs through BoxPromptRefiner's batching and choice."""

    type_name = "scene_sam"
    Options = BoxPromptOptions

    def _load(self, ctx: LoadContext) -> None:
        """Nothing to load."""

    def _prepare(
        self, rgb: np.ndarray, xyxy: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        objects = [i for i, c in enumerate(SCENE.classes) if c.is_object]
        return np.isin(classify(rgb, SCENE), objects), xyxy

    def _decode(
        self, state: tuple[np.ndarray, np.ndarray], batch: slice
    ) -> DecodedMasks:
        is_object, xyxy = state
        h, w = is_object.shape
        boxes = xyxy[batch]
        masks = np.zeros((len(boxes), 1, h, w), dtype=bool)
        for j, box in enumerate(boxes):
            c0, r0, c1, r1 = _box_window(box, w, h)
            masks[j, 0, r0:r1, c0:c1] = is_object[r0:r1, c0:c1]
        return DecodedMasks(masks, np.full((len(boxes), 1), 0.95, dtype=np.float32))


class SceneSegmenter(SemanticSegmenter):
    """Semantic classes of the scene (road, sky, person, vehicle) per pixel."""

    type_name = "scene_seg"

    @dataclass(frozen=True, slots=True)
    class Options:
        pass

    CLASS_NAMES: ClassVar[tuple[str, ...]] = ("road", "sky", "person", "vehicle")

    def _load(self, ctx: LoadContext) -> None:
        """Nothing to load."""

    @property
    def class_names(self) -> tuple[str, ...]:
        return self.CLASS_NAMES

    def _segment(self, rgb: np.ndarray) -> SemanticMap:
        lut = np.array(
            [self.CLASS_NAMES.index(c.semantic) for c in SCENE.classes], dtype=np.int32
        )
        return SemanticMap(lut[classify(rgb, SCENE)], self.CLASS_NAMES)


# DIP: the fake network depends on the oracle (what the "camera" sees), not on the test.
@dataclass
class DepthOracle:
    """What the fake DA3 network perceives: the scene and the frame-resolution
    focal of the image it is given (set by the test before each run)."""

    spec: SceneSpec
    perceived: CameraIntrinsics  # frame-resolution K the net "sees"

    def loader(self, repo_id: str, device: str, cache_dir: str | None) -> "SceneDA3Net":
        """Drop-in for da3._load_da3."""
        return SceneDA3Net(self)


def _network_focal(
    k: CameraIntrinsics,
    frame: tuple[int, int],
    g: MapGeometry,
    process_res: int,
    method: str,
) -> tuple[float, float]:
    """(fx, fy) of the image the DA3 network sees, from DA3's own resizing
    rule rather than MapGeometry.scale_intrinsics: *_resize stretches the
    frame to the map size; *_crop scales the reference side to process_res
    (Python round) and centre-crops, which keeps the focal. Written out here
    so that a per-axis scaling bug in the adapter's geometry does not cancel
    out between this fake and the pipeline."""
    w, h = frame
    if method.endswith("resize"):
        return k.fx * g.map_w / w, k.fy * g.map_h / h
    ref = max(w, h) if method.startswith("upper") else min(w, h)
    s = process_res / ref
    return k.fx * round(w * s) / w, k.fy * round(h * s) / h


class SceneDA3Net:
    """The network object DA3Depth drives: `model_name` and `inference`."""

    model_name: str = "da3metric-large"

    def __init__(self, oracle: DepthOracle) -> None:
        self.oracle = oracle

    def inference(
        self, images: list[np.ndarray], process_res: int, process_res_method: str
    ) -> SimpleNamespace:
        """One-frame DA3-like prediction: depth (1, map_h, map_w) float32 in
        DA3METRIC units, sky (1, map_h, map_w) bool, intrinsics None (no camera
        head), on the grid of `da3_map_geometry` for this frame size."""
        (rgb,) = images
        h, w = rgb.shape[:2]
        g = da3.da3_map_geometry(w, h, process_res, process_res_method)
        spec = self.oracle.spec
        grid = resample_mask(classify(rgb, spec), g, (0, 0, g.map_w, g.map_h))
        fx, fy = _network_focal(
            self.oracle.perceived, (w, h), g, process_res, process_res_method
        )
        raw = spec.depth_lut()[grid] * (da3.DA3_METRIC_SCALE / ((fx + fy) / 2))
        sky = grid == spec.index("sky")
        return SimpleNamespace(
            depth=raw.astype(np.float32)[None], sky=sky[None], intrinsics=None
        )


class NullDeviceMonitor:
    """runtime.DeviceMonitor without torch: no baseline, every figure None."""

    KEYS: ClassVar[tuple[str, ...]] = (
        "peak_vram_torch_allocated_gb",
        "device_mem_used_delta_gb_approx",
        "torch",
        "cuda",
        "gpu",
    )

    def __init__(self, device: str) -> None:
        self.device = device

    def start(self) -> None:
        """Nothing to measure."""

    def summary(self) -> dict[str, None]:
        return dict.fromkeys(self.KEYS)


# SRP: the only place touching global state; everything through monkeypatch.
def install_scene_fakes(monkeypatch: pytest.MonkeyPatch, oracle: DepthOracle) -> None:
    """Register the scene doubles and patch the torch-only seams for one test.

    Args:
        monkeypatch: the test's monkeypatch (every change is undone at teardown).
        oracle: what the fake depth network perceives (scene and focal).
    """
    registry.load_adapters()  # build() would reload later; insert after it
    for cls in (SceneDetector, SceneRefiner, SceneSegmenter):
        monkeypatch.setitem(registry._REGISTRY[cls.kind], cls.type_name, cls)
    monkeypatch.setattr(cli, "DeviceMonitor", NullDeviceMonitor)
    monkeypatch.setattr(da3, "_load_da3", oracle.loader)
    monkeypatch.setattr(da3.DA3Depth, "_close", lambda self: None)
    # The fake network needs no checkpoint nor torch: the run's readiness gate
    # (cli.require_ready) sees DA3 (and the focal model) as ready.
    monkeypatch.setattr(
        da3.DA3Depth, "requirements", classmethod(lambda cls, options: Requirements())
    )
    monkeypatch.setattr(cli, "da3_requirements", lambda model, hf_cache: Requirements())
