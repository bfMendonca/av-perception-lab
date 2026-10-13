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
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

if TYPE_CHECKING:
    from det_depth_distance.distance import BoxDepthStats


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


def _per_instance_fields(cls: type) -> tuple[dataclasses.Field, ...]:
    return tuple(f for f in dataclasses.fields(cls) if f.metadata.get("per_instance"))


@dataclass(frozen=True, slots=True)
class Instances:
    """Instances detected (or segmented) in one frame, possibly by several models.

    `label` is the canonical name used downstream, `raw_label` the name in the
    producing model's vocabulary, `source` the producing component's name.
    `masks`, when present, are boolean (N, H, W) at frame resolution; an
    all-False mask means "no mask" for that instance, which lets detectors with
    and without masks be concatenated (missing masks are filled with False).
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
class DepthResult:
    """Metric depth of one frame at depth-map resolution."""

    depth_m: np.ndarray  # (map_h, map_w) float32, metres along the optical axis (z)
    intrinsics: CameraIntrinsics  # at depth-map resolution
    # bool (map_h, map_w), or None when the model has no sky head
    sky: np.ndarray | None
    geometry: MapGeometry  # frame -> depth map


@dataclass(frozen=True, slots=True)
class FrameResult:
    """Everything computed for one frame; input of the sinks."""

    frame: Frame
    instances: Instances
    stats: "tuple[BoxDepthStats, ...]"  # aligned with instances
    depth: DepthResult
    timings_ms: dict[str, float]  # "det", "det.<name>", "aggregate", "depth", "dist"
