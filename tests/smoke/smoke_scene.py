"""Synthetic scene with known metric geometry for the end-to-end smoke tests.

A 320x180, 10 fps, 12-frame video of fronto-parallel planes, one grey level
per class (luma only, >= 40 levels apart, so yuv420p chroma subsampling and
crf=0 encoding keep every pixel classifiable): road and sky backgrounds, a
solid moving car (A) and two static rings, a person whose hole shows the
road (B) and a truck whose hole shows the sky (C). The rings make the box
median the plane seen through the hole (61 % of the 10 %-shrunk box) and
the mask median the object, so box_shrink, sky and semantic exclusion,
region policy and erosion all have analytic expected distances
(`expected_box_distance`). All coordinates are even. Pure numpy/OpenCV.
"""

from collections.abc import Callable, Collection, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from det_depth_distance.io.video import VideoWriter


@dataclass(frozen=True, slots=True)
class SceneClass:
    """One label of the scene: its grey level in the video and its plane."""

    name: str  # "road", "sky", "car", "person", "truck"
    grey: int  # luma level in the video, >= 40 apart
    depth_m: float  # z of the fronto-parallel plane
    semantic: str  # class of SceneSegmenter
    is_object: bool


@dataclass(frozen=True, slots=True)
class SceneObject:
    """A detectable object: a solid block or a ring whose hole shows another class."""

    cls: str  # SceneClass.name
    raw_label: str  # COCO raw label of SceneDetector
    box: tuple[int, int, int, int]  # x0, y0, x1, y1 at frame 0 (even)
    step_px: int = 0  # x motion per frame
    wall_px: int | None = None  # ring wall; None = solid
    hole: str | None = None  # class seen through the ring

    def box_at(self, i: int) -> tuple[int, int, int, int]:
        """Box (half-open pixel edges) at frame `i`."""
        x0, y0, x1, y1 = self.box
        dx = self.step_px * i
        return x0 + dx, y0, x1 + dx, y1


@dataclass(frozen=True, slots=True)
class SceneSpec:
    """Size, timing, classes (label id = position) and objects of the scene."""

    size: tuple[int, int]  # (W, H)
    fps: str  # VideoWriter frame rate, "num/den"
    n_frames: int
    horizon_y: int  # sky above, road from this row down
    classes: tuple[SceneClass, ...]
    objects: tuple[SceneObject, ...]

    def index(self, name: str) -> int:
        """Label id of class `name`."""
        return next(i for i, c in enumerate(self.classes) if c.name == name)

    def depth_lut(self) -> np.ndarray:
        """(n_classes,) float32 plane depth of every label id."""
        return np.array([c.depth_m for c in self.classes], dtype=np.float32)

    def cls(self, name: str) -> SceneClass:
        return self.classes[self.index(name)]


SCENE = SceneSpec(
    size=(320, 180),
    fps="10/1",
    n_frames=12,
    horizon_y=80,
    classes=(
        SceneClass("road", 40, 20.0, "road", False),
        SceneClass("sky", 230, 100.0, "sky", False),
        SceneClass("car", 90, 8.0, "vehicle", True),
        SceneClass("person", 140, 12.0, "person", True),
        SceneClass("truck", 190, 15.0, "vehicle", True),
    ),
    objects=(
        # A: solid, 8 px per frame (IoU 0.67 between consecutive frames).
        SceneObject("car", "car", (16, 92, 56, 116), step_px=8),
        # B: ring on the road, its hole shows the road.
        SceneObject("person", "person", (200, 112, 248, 160), wall_px=9, hole="road"),
        # C: ring in the sky, its hole shows the sky.
        SceneObject("truck", "truck", (60, 16, 108, 64), wall_px=9, hole="sky"),
    ),
)


def render_labels(spec: SceneSpec, i: int) -> np.ndarray:
    """(H, W) uint8 class ids of the ideal (pinhole) frame `i`."""
    w, h = spec.size
    labels = np.full((h, w), spec.index("road"), np.uint8)
    labels[: spec.horizon_y] = spec.index("sky")
    for obj in spec.objects:
        x0, y0, x1, y1 = obj.box_at(i)
        labels[y0:y1, x0:x1] = spec.index(obj.cls)
        if obj.wall_px is not None:
            t = obj.wall_px
            labels[y0 + t : y1 - t, x0 + t : x1 - t] = spec.index(obj.hole)
    return labels


def classify(rgb: np.ndarray, spec: SceneSpec) -> np.ndarray:
    """(H, W) int32 class ids of a decoded RGB frame: the nearest grey level
    (the frame is grey, so channel 0 is the luma)."""
    greys = np.array([c.grey for c in spec.classes], dtype=np.int16)
    distance = np.abs(rgb[..., 0].astype(np.int16)[..., None] - greys)
    return distance.argmin(axis=-1).astype(np.int32)


def scene_frames(
    spec: SceneSpec, distort: Callable[[np.ndarray], np.ndarray] | None = None
) -> list[np.ndarray]:
    """Label images of every frame, seen through `distort` (a label-exact lens,
    e.g. nearest-neighbour remap) when given."""
    frames = [render_labels(spec, i) for i in range(spec.n_frames)]
    return frames if distort is None else [distort(f) for f in frames]


def write_scene_video(
    spec: SceneSpec, path: Path, labels: Sequence[np.ndarray]
) -> Path:
    """Encode label images as a grey video with the project's VideoWriter;
    crf=0 is lossless for flat grey levels (decoded labels are exact)."""
    greys = np.array([c.grey for c in spec.classes], dtype=np.uint8)
    w, h = spec.size
    with VideoWriter(path, w, h, spec.fps, crf=0) as writer:
        for frame in labels:
            writer.write(np.repeat(greys[frame][..., None], 3, axis=-1))
    return path


def true_box(
    labels: np.ndarray, spec: SceneSpec, cls: str
) -> tuple[float, float, float, float]:
    """Continuous edges (x0, y0, x1, y1) of the pixels of class `cls` (each
    object class appears once in the scene)."""
    ys, xs = np.nonzero(labels == spec.index(cls))
    return float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)


def expected_box_distance(
    spec: SceneSpec,
    cls: str,
    *,
    exclude_sky: bool,
    excluded: Collection[str],
    box_shrink: float,
) -> float:
    """z distance (m) of the box region of object `cls` under the given policy.

    Derived from the scene constants only (never from distance.py): the plane
    holding more than half of the eligible pixels of the shrunk box wins,
    because the median of a two-plane region is the majority plane and the
    IQR rule keeps both planes. The box is the object's own box (a detector
    pad of a pixel never changes the winner in this scene). `excluded` holds
    semantic class names of SceneSegmenter.

    Raises:
        ValueError: no plane of the box is eligible.
    """
    obj = next(o for o in spec.objects if o.cls == cls)
    x0, y0, x1, y1 = obj.box
    dx, dy = (x1 - x0) * box_shrink, (y1 - y0) * box_shrink
    inner = (x1 - x0 - 2 * dx) * (y1 - y0 - 2 * dy)
    areas = {cls: inner}
    if obj.wall_px is not None:
        t = obj.wall_px
        # The hole lies inside the shrunk box whenever box_shrink * side <= wall.
        hole = (x1 - x0 - 2 * max(t, dx)) * (y1 - y0 - 2 * max(t, dy))
        areas = {cls: inner - hole, obj.hole: hole}

    def eligible(name: str) -> bool:
        c = spec.cls(name)
        return not (exclude_sky and name == "sky") and c.semantic not in excluded

    planes = {name: area for name, area in areas.items() if eligible(name)}
    if not planes:
        raise ValueError(f"no eligible plane in the box of {cls}")
    return spec.cls(max(planes, key=planes.get)).depth_m
