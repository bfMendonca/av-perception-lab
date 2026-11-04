"""Data contracts shared by every pipeline component (numpy only).

Everything that crosses a component boundary is an immutable dataclass of
numpy arrays and builtins, never a framework object. Conventions: frames are
RGB uint8 and upright; boxes are continuous edges in frame pixels; masks are
at frame resolution; depth stays at the model's native map resolution and
carries the frame -> map `MapGeometry` (resampling would cost time and blur
edges). Per-instance fields of `Instances` carry `per_instance(check)`
metadata - their own validator and optional concat fill - so
`select`/`concat`/`validate` handle fields added later (fused `sources`,
`track_id`, ...) without changes.

Import rule: this module never imports `distance` at run time (distance
imports MapGeometry/CameraIntrinsics from here); distance types used in
annotations are imported under TYPE_CHECKING only. It is contract-only:
it never imports `core.pipeline`, `config`, `camera`, `io` or adapters
(`core.pipeline` imports from those modules, never the reverse).
"""

import dataclasses
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

if TYPE_CHECKING:
    from av_perception_lab.distance import BoxDepthStats


class ContractError(RuntimeError):
    """A component broke its contract (a bug in the adapter, not a configuration problem)."""


@dataclass(frozen=True, slots=True)
class MapGeometry:
    """Per-axis affine from frame pixels to depth-map pixels.

    `x_map = x * sx + ox`, `y_map = y * sy + oy`, applied to continuous box
    edges. sx != sy in general (DA3 rounds each side to a multiple of 14), and
    ox/oy are negative crop offsets for the *_crop processing methods.
    """

    frame_w: int
    frame_h: int
    map_w: int
    map_h: int
    sx: float
    sy: float
    ox: float = 0.0
    oy: float = 0.0

    def frame_to_map(self, xyxy: Sequence[float]) -> tuple[float, float, float, float]:
        """Map box edges (x0, y0, x1, y1) from frame to depth-map coordinates."""
        x0, y0, x1, y1 = xyxy
        return (
            x0 * self.sx + self.ox,
            y0 * self.sy + self.oy,
            x1 * self.sx + self.ox,
            y1 * self.sy + self.oy,
        )

    def map_to_frame(self, xyxy: Sequence[float]) -> tuple[float, float, float, float]:
        """Inverse of `frame_to_map`."""
        x0, y0, x1, y1 = xyxy
        return (
            (x0 - self.ox) / self.sx,
            (y0 - self.oy) / self.sy,
            (x1 - self.ox) / self.sx,
            (y1 - self.oy) / self.sy,
        )

    def frame_to_map_points(self, xy: np.ndarray) -> np.ndarray:
        """Same affine as `frame_to_map` for (N, 2) points; (N, 2) float64."""
        xy = np.asarray(xy, dtype=np.float64)
        return xy * (self.sx, self.sy) + (self.ox, self.oy)

    def map_to_frame_points(self, xy: np.ndarray) -> np.ndarray:
        """Inverse of `frame_to_map_points`."""
        xy = np.asarray(xy, dtype=np.float64)
        return (xy - (self.ox, self.oy)) / (self.sx, self.sy)

    def scale_intrinsics(self, k: "CameraIntrinsics") -> "CameraIntrinsics":
        """Frame-resolution intrinsics -> depth-map-resolution intrinsics.

        Same affine as `frame_to_map` (continuous pixel coordinates), applied to
        the principal point; focal lengths scale per axis.
        """
        return CameraIntrinsics(
            fx=k.fx * self.sx,
            fy=k.fy * self.sy,
            cx=k.cx * self.sx + self.ox,
            cy=k.cy * self.sy + self.oy,
        )


@dataclass(frozen=True, slots=True)
class CameraIntrinsics:
    """Pinhole intrinsics in pixels, at frame or depth-map resolution (the
    holder says which).

    Continuous pixel coordinates: pixel i is centred at i + 0.5, so a centred
    camera has its principal point at (W/2, H/2), consistent with box edges,
    MapGeometry and ray_factors. OpenCV centres pixel i at i, so its cx/cy are
    0.5 smaller; convert when reading or writing OpenCV matrices.
    """

    fx: float
    fy: float
    cx: float
    cy: float

    @classmethod
    def centered(cls, focal_px: float, width: int, height: int) -> "CameraIntrinsics":
        """Square pixels with the principal point at the image centre (W/2, H/2)."""
        return cls(fx=focal_px, fy=focal_px, cx=width / 2, cy=height / 2)


@dataclass(frozen=True, slots=True)
class Frame:
    """One decoded frame."""

    index: int  # index in the source video
    t: float  # seconds from the start of the source video
    rgb: np.ndarray  # (H, W, 3) uint8, RGB, upright

    @property
    def width(self) -> int:
        return self.rgb.shape[1]

    @property
    def height(self) -> int:
        return self.rgb.shape[0]


# Tolerance (s) of time limits compared with differences of Frame.t (track
# age, smoother reset, median window). Frame.t = start + decoded / fps, so a
# gap of exactly a limit (e.g. 15 frames = 0.5 s at 30 fps) can come out a
# few ulps above it depending on the frame index; the limit is inclusive.
FRAME_TIME_EPS_S = 1e-6

# (field name, value, N, frame_w, frame_h) -> None; raises ContractError.
FieldCheck = Callable[[str, Any, int, int, int], None]


def per_instance(
    check: FieldCheck, *, concat_fill: Literal["zeros"] | None = None
) -> dict[str, Any]:
    """Field metadata marking a per-instance field of `Instances`.

    Args:
        check: validator of the field's value for N instances in a frame.
        concat_fill: how `Instances.concat` fills parts where this optional
            field is None ("zeros"); None makes the concatenated field None.
    """
    return {"per_instance": True, "check": check, "concat_fill": concat_fill}


def check_array(dtype: type, tail: tuple[int | str, ...] = ()) -> FieldCheck:
    """Validator of an (N, *tail) array of `dtype`; "w"/"h" in `tail` mean the frame size."""

    def check(name: str, value: Any, n: int, frame_w: int, frame_h: int) -> None:
        sizes = {"w": frame_w, "h": frame_h}
        shape = (n, *(sizes.get(d, d) for d in tail))
        if not isinstance(value, np.ndarray) or value.dtype != dtype:
            got = getattr(value, "dtype", type(value).__name__)
            raise ContractError(f"{name}: expected {np.dtype(dtype)} array, got {got}")
        if value.shape != shape:
            raise ContractError(f"{name}: shape {value.shape} != {shape}")

    return check


def check_str_tuple() -> FieldCheck:
    """Validator of a tuple of N strings."""

    def check(name: str, value: Any, n: int, frame_w: int, frame_h: int) -> None:
        if not isinstance(value, tuple) or len(value) != n:
            raise ContractError(f"{name}: expected a tuple of {n} strings")
        if not all(isinstance(v, str) for v in value):
            raise ContractError(f"{name}: all items must be str")

    return check


_check_box_array = check_array(np.float32, (4,))


def _check_xyxy(name: str, value: Any, n: int, frame_w: int, frame_h: int) -> None:
    # Strict: both detectors in use clip boxes to the frame, so anything outside is a bug.
    _check_box_array(name, value, n, frame_w, frame_h)
    if not np.isfinite(value).all():
        raise ContractError(f"{name}: non-finite box coordinates")
    x0, y0, x1, y1 = value.T
    if (x1 < x0).any() or (y1 < y0).any():
        raise ContractError(f"{name}: inverted box (x1 < x0 or y1 < y0)")
    if (value < 0).any() or (x1 > frame_w).any() or (y1 > frame_h).any():
        raise ContractError(f"{name}: box outside the {frame_w}x{frame_h} frame")


def _check_track_id(name: str, value: Any, n: int, frame_w: int, frame_h: int) -> None:
    check_array(np.int64)(name, value, n, frame_w, frame_h)
    if (value < 1).any():
        raise ContractError(f"{name}: track ids must be >= 1")


@dataclass(frozen=True, slots=True)
class Provenance:
    """One detection that contributed to an instance: the detector's name, its
    raw label and its original (unweighted) score."""

    source: str
    raw_label: str
    score: float


def _check_sources(name: str, value: Any, n: int, frame_w: int, frame_h: int) -> None:
    if not isinstance(value, tuple) or len(value) != n:
        raise ContractError(f"{name}: expected a tuple of {n} member tuples")
    for members in value:
        if not isinstance(members, tuple) or not members:
            raise ContractError(f"{name}: every instance needs a non-empty tuple")
        if not all(isinstance(m, Provenance) for m in members):
            raise ContractError(f"{name}: members must be Provenance")


def best_per_source(members: Sequence[Provenance]) -> tuple[Provenance, ...]:
    """One member per detector (its highest score), by decreasing score; ties
    keep the order of `members`."""
    best: dict[str, Provenance] = {}
    for m in members:
        if m.source not in best or m.score > best[m.source].score:
            best[m.source] = m
    return tuple(sorted(best.values(), key=lambda m: -m.score))


def _per_instance_fields(cls: type) -> tuple[dataclasses.Field, ...]:
    return tuple(f for f in dataclasses.fields(cls) if f.metadata.get("per_instance"))


@dataclass(frozen=True, slots=True)
class Instances:
    """Instances detected (or segmented) in one frame, possibly by several models.

    `score` is a confidence in [0, 1] (written as the CSV `confidence` and
    shown as a percentage; fused scores keep that range).
    `label` is the canonical name used downstream, `raw_label` the name in the
    producing model's vocabulary, `source` the producing component's name.
    `masks`, when present, are boolean (N, H, W) at frame resolution; an
    all-False mask means "no mask" for that instance, which lets detectors with
    and without masks be concatenated (missing masks are filled with False).
    `sources`, set by fusion aggregators, lists the detections merged into
    each instance; `provenance()` reads it for fused and plain instances alike.
    `mask_source` / `mask_score` are set by mask refiners: the component that
    produced each mask ("" = no mask) and the refiner's quality score (NaN for
    detector masks). Fusion aggregators also set them, naming the member whose
    mask a fused instance kept (it may not be the representative).
    `mask_sources()` derives the sources when neither ran.
    `track_id`, set by the tracker, identifies the same object across frames.
    """

    # Open for extension: select/concat/validate iterate the fields carrying
    # per_instance metadata and run each field's own validator, so new optional
    # fields (default=None) need no change to these methods.
    xyxy: np.ndarray = field(metadata=per_instance(_check_xyxy))  # (N, 4) frame px
    score: np.ndarray = field(metadata=per_instance(check_array(np.float32)))
    class_id: np.ndarray = field(metadata=per_instance(check_array(np.int64)))
    label: tuple[str, ...] = field(metadata=per_instance(check_str_tuple()))
    raw_label: tuple[str, ...] = field(metadata=per_instance(check_str_tuple()))
    source: tuple[str, ...] = field(metadata=per_instance(check_str_tuple()))
    masks: np.ndarray | None = field(
        default=None,
        metadata=per_instance(check_array(np.bool_, ("h", "w")), concat_fill="zeros"),
    )
    # Members of each fused instance, best weighted score first; None = not
    # fused. Only fusion aggregators fill it and their output is never
    # concatenated again, hence no concat_fill.
    sources: tuple[tuple[Provenance, ...], ...] | None = field(
        default=None, metadata=per_instance(_check_sources)
    )
    # Producer of each mask ("" = no mask): a detector or refiner name. Set by
    # refiners and fusion aggregators, both downstream of every concat.
    mask_source: tuple[str, ...] | None = field(
        default=None, metadata=per_instance(check_str_tuple())
    )
    # (N,) float32 refiner quality score of each mask; NaN for detector masks.
    mask_score: np.ndarray | None = field(
        default=None, metadata=per_instance(check_array(np.float32))
    )
    # (N,) int64 >= 1, stable across frames; None = no tracker. Set by the
    # tracker, downstream of every concat, hence no concat_fill.
    track_id: np.ndarray | None = field(
        default=None, metadata=per_instance(_check_track_id)
    )

    def __len__(self) -> int:
        return len(self.xyxy)

    @classmethod
    def empty(cls) -> "Instances":
        """No instances (contract-valid for any frame size)."""
        return cls(
            np.zeros((0, 4), np.float32),
            np.zeros(0, np.float32),
            np.zeros(0, np.int64),
            (),
            (),
            (),
        )

    def select(self, keep: np.ndarray) -> "Instances":
        """Subset by a boolean mask of length N or by integer indices."""
        keep = np.asarray(keep)
        idx = np.flatnonzero(keep) if keep.dtype == bool else keep.astype(np.intp)
        values = {}
        for f in _per_instance_fields(type(self)):
            value = getattr(self, f.name)
            if value is None:
                values[f.name] = None
            elif isinstance(value, tuple):
                values[f.name] = tuple(value[i] for i in idx)
            else:
                values[f.name] = value[idx]
        return dataclasses.replace(self, **values)

    def provenance(self) -> tuple[tuple[Provenance, ...], ...]:
        """Members of every instance: `sources` when instances were fused, else one
        member per instance built from (source, raw_label, score). Lets consumers
        treat fused and plain instances alike."""
        if self.sources is not None:
            return self.sources
        return tuple(
            (Provenance(source, raw_label, float(score)),)
            for source, raw_label, score in zip(
                self.source, self.raw_label, self.score, strict=True
            )
        )

    def mask_sources(self) -> tuple[str, ...]:
        """Producer of every instance's mask ("" = none): `mask_source` when a
        refiner or fusion set it, else the instance's `source` where `has_mask()`."""
        if self.mask_source is not None:
            return self.mask_source
        return tuple(
            source if has else ""
            for source, has in zip(self.source, self.has_mask(), strict=True)
        )

    def with_labels(self, labels: Sequence[str]) -> "Instances":
        """Same instances with new canonical labels (raw_label is kept)."""
        return dataclasses.replace(self, label=tuple(labels))

    def has_mask(self) -> np.ndarray:
        """(N,) bool: whether each instance has a usable mask, i.e. at least one
        True pixel inside its box window (pixels whose centre lies in the box, the
        rule of box_to_map_pixels). Mask pixels outside the box are ignored, as the
        mask distance does."""
        if self.masks is None:
            return np.zeros(len(self), dtype=bool)
        _, h, w = self.masks.shape
        out = np.zeros(len(self), dtype=bool)
        for i, (x0, y0, x1, y1) in enumerate(self.xyxy.tolist()):
            c0 = min(max(math.ceil(x0 - 0.5), 0), w)
            r0 = min(max(math.ceil(y0 - 0.5), 0), h)
            c1 = min(max(math.floor(x1 - 0.5) + 1, c0), w)
            r1 = min(max(math.floor(y1 - 0.5) + 1, r0), h)
            out[i] = bool(self.masks[i, r0:r1, c0:c1].any())
        return out

    @staticmethod
    def concat(parts: Sequence["Instances"]) -> "Instances":
        """Instances of all `parts`, in order.

        An optional field that is None in every part stays None; None in only
        some parts is filled with zeros when the field declares
        `concat_fill="zeros"`, otherwise the result's field is None.
        """
        if not parts:
            return Instances.empty()
        cls = type(parts[0])
        values = {}
        for f in _per_instance_fields(cls):
            items = [getattr(p, f.name) for p in parts]
            present = [v for v in items if v is not None]
            if not present:
                values[f.name] = None
                continue
            if len(present) < len(items):
                if f.metadata["concat_fill"] != "zeros":
                    values[f.name] = None
                    continue
                ref = present[0]
                items = [
                    np.zeros((len(p), *ref.shape[1:]), ref.dtype) if v is None else v
                    for p, v in zip(parts, items, strict=True)
                ]
            if isinstance(items[0], tuple):
                values[f.name] = tuple(x for item in items for x in item)
            else:
                values[f.name] = np.concatenate(items, axis=0)
        return cls(**values)

    def validate(self, frame_w: int, frame_h: int) -> None:
        """Check shapes, dtypes and geometry against a frame of `frame_w` x `frame_h`
        by running the validator declared in each per-instance field's metadata.

        Raises:
            ContractError: an adapter returned malformed instances (a bug in the
                adapter, never a user configuration problem).
        """
        n = len(self)
        for f in _per_instance_fields(type(self)):
            value = getattr(self, f.name)
            if value is None:
                if f.default is not None:
                    raise ContractError(
                        f"{f.name}: required per-instance field is None"
                    )
                continue
            f.metadata["check"](f.name, value, n, frame_w, frame_h)


@dataclass(frozen=True, slots=True)
class RefinedMasks:
    """Output of `MaskRefiner._refine` for its K target instances, in order."""

    masks: np.ndarray  # (K, H, W) bool at frame resolution; all-False = no mask
    score: np.ndarray  # (K,) float32 quality predicted by the refiner

    def validate(self, k: int, frame_w: int, frame_h: int) -> None:
        """Raises ContractError unless masks is (k, frame_h, frame_w) bool and
        score is (k,) float32."""
        check_array(np.bool_, ("h", "w"))("masks", self.masks, k, frame_w, frame_h)
        check_array(np.float32)("score", self.score, k, frame_w, frame_h)


@dataclass(frozen=True, slots=True)
class SemanticMap:
    """Class label per frame pixel: `labels` (H, W) int32 indexes `class_names`."""

    labels: np.ndarray
    class_names: tuple[str, ...]

    def validate(self, frame_w: int, frame_h: int) -> None:
        """Raises ContractError unless labels is (frame_h, frame_w) int32 with every
        value a valid index of `class_names`."""
        labels = self.labels
        if not isinstance(labels, np.ndarray) or labels.dtype != np.int32:
            got = getattr(labels, "dtype", type(labels).__name__)
            raise ContractError(f"labels: expected int32 array, got {got}")
        if labels.shape != (frame_h, frame_w):
            raise ContractError(f"labels: shape {labels.shape} != {(frame_h, frame_w)}")
        if labels.size and (labels.min() < 0 or labels.max() >= len(self.class_names)):
            raise ContractError(
                f"labels: values outside [0, {len(self.class_names)}) (class_names)"
            )

    def fractions(self) -> dict[str, float]:
        """Fraction of the frame's pixels per class name (sums to 1)."""
        counts = np.bincount(self.labels.ravel(), minlength=len(self.class_names))
        return {
            name: float(c) / self.labels.size
            for name, c in zip(self.class_names, counts, strict=True)
        }


@dataclass(frozen=True, slots=True)
class DepthResult:
    """Metric depth of one frame at depth-map resolution."""

    depth_m: np.ndarray  # (map_h, map_w) float32, metres along the optical axis (z)
    intrinsics: CameraIntrinsics  # at depth-map resolution
    # bool (map_h, map_w), or None when the model has no sky head
    sky: np.ndarray | None
    geometry: MapGeometry  # frame -> depth map


@dataclass(frozen=True, slots=True)
class PairAgreement:
    """One-to-one IoU matches between the deduplicated instances of two detectors."""

    a: str
    b: str
    matched: int
    only_a: int
    only_b: int
    iou_sum: float  # sum of the matched IoUs (mean = iou_sum / matched)


@dataclass(frozen=True, slots=True)
class FrameAgreement:
    """Cross-detector agreement in one frame (see core.agreement)."""

    instances: Mapping[str, int]  # per detector, after deduplication
    duplicates: Mapping[str, int]  # per detector, suppressed by the deduplication
    pairs: tuple[PairAgreement, ...]  # combinations in detector order


@dataclass(frozen=True, slots=True)
class TrackEstimate:
    """Smoothed state of one track at one frame: distance (m) and, for the
    Kalman filter, its std, the relative speed d(distance)/dt (m/s, negative =
    approaching) and its std. n_updates counts measurements since the filter
    (re)started."""

    distance: float  # m, same quantity as BoxDepthStats.distance (z or euclidean)
    distance_std: float  # m; NaN without a state model (ema, median)
    rel_speed: float  # m/s; NaN without a state model
    rel_speed_std: float  # m/s; NaN without a state model
    n_updates: int


@dataclass(frozen=True, slots=True)
class Trajectory:
    """Predicted positions of one track after the frame time, in the current
    camera frame, with covariances; `y` is the measured height of the
    reference point used to draw it."""

    t: np.ndarray  # (K,) s after the frame time, step_s .. horizon_s
    xz: np.ndarray  # (K, 2) m, camera frame of the current frame
    cov: np.ndarray  # (K, 2, 2) m^2
    y: float  # m, camera y (+ down) of the reference point


@dataclass(frozen=True, slots=True)
class MotionEstimate:
    """2D motion state of one track at one frame, on the camera plane (x right,
    z forward; see av_perception_lab.motion for the frames and signs)."""

    model: str  # key of motion.models
    x: float  # m, camera frame
    z: float
    x_std: float
    z_std: float
    vx: float  # m/s, d/dt of the position in the camera frame
    vz: float
    speed: float  # m/s, |(vx, vz)|
    speed_std: float
    # 0 = moving away, +90 = moving right; NaN when slow. With ego-motion the
    # heading, its min speed and the yaw rate are those of the motion over
    # ground (camera axes of this frame), not atan2(vx, vz).
    heading_deg: float
    heading_std_deg: float
    yaw_rate_dps: float  # + = turning right; NaN for models without turn rate
    yaw_rate_std_dps: float
    abs_speed: float  # m/s over ground; NaN without ego-motion
    nis: float  # normalised innovation squared; NaN without a measurement
    n_updates: int  # measurements since the filter (re)started
    restarted: bool  # the filter restarted at this frame (NIS gate)
    trajectory: Trajectory | None


@dataclass(frozen=True, slots=True)
class FrameResult:
    """Everything computed for one frame; input of the sinks."""

    frame: Frame
    instances: Instances
    stats: "tuple[BoxDepthStats, ...]"  # aligned with instances
    depth: DepthResult
    # "det", "det.<name>", "agree" (when computed), "aggregate", "refine" and
    # "refine.<name>" (with refiners), "track" (with a tracker), "depth",
    # "semantic" and "semantic.<name>" (with semantic segmenters), "dist",
    # "smooth" (with smoothing), "motion" (with motion)
    timings_ms: dict[str, float]
    agreement: FrameAgreement | None = None  # None with one detector or disabled
    # Box statistics aligned with instances, only with region "both": set where
    # `stats` came from the instance mask, None elsewhere.
    box_stats: "tuple[BoxDepthStats | None, ...] | None" = None
    # Segmenter name -> its map, in configuration (= drawing) order; {} without
    # semantic segmenters. Maps are never merged: each keeps its own classes.
    semantic: Mapping[str, SemanticMap] = field(default_factory=dict)
    # Smoothed per-track state aligned with instances, only with smoothing;
    # None for an instance whose track has no distance yet.
    temporal: tuple[TrackEstimate | None, ...] | None = None
    # Per-track motion estimate aligned with instances, only with motion;
    # None where the track has no estimate.
    motion: tuple[MotionEstimate | None, ...] | None = None
