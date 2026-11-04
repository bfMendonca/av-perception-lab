"""Ground-truth reference distances for one video (YAML).

Each reference selects one object over a frame (or time) range by a box, a
point (video pixels of the original, not undistorted, video) or a track id,
and gives the measured distance (constant or a time series), the instrument
and the quantity measured: "range" (straight line from the lens) or "z"
(along the optical axis). See docs/refs/protocolo_calibracao_e_referencias.md.

Shape errors come from core.options.from_mapping, domain rules from the
__post_init__ of each dataclass; both are reported as ConfigError with the
file and the dotted path (`refs.yaml: references[1].select: ...`).
"""

import itertools
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import yaml

from av_perception_lab.core.options import ConfigError, from_mapping
from av_perception_lab.core.types import FRAME_TIME_EPS_S

Quantity = Literal["range", "z"]


@dataclass(frozen=True, slots=True)
class Instrument:
    """A measuring instrument; per-measurement uncertainty is
    hypot(uncertainty_m, uncertainty_rel * distance)."""

    description: str = ""
    uncertainty_m: float = 0.0
    uncertainty_rel: float = 0.0  # fraction of the distance

    def __post_init__(self) -> None:
        if self.uncertainty_m < 0 or self.uncertainty_rel < 0:
            raise ValueError("uncertainty_m and uncertainty_rel must be >= 0")


@dataclass(frozen=True, slots=True)
class Selector:
    """Which instance of a frame is the referenced object."""

    box: tuple[float, float, float, float] | None = (
        None  # x0 y0 x1 y1, original video px
    )
    point: tuple[float, float] | None = None  # x y, original video px
    track_id: int | None = None  # id of one run (not reproducible across runs)
    label: str | None = None  # only instances with this class_name

    def __post_init__(self) -> None:
        if sum(v is not None for v in (self.box, self.point, self.track_id)) != 1:
            raise ValueError("exactly one of box, point, track_id")
        if self.box is not None and not (
            self.box[2] > self.box[0] and self.box[3] > self.box[1]
        ):
            raise ValueError(f"box must have x1 > x0 and y1 > y0, got {list(self.box)}")


def _increasing(pair: tuple[float, float], name: str) -> None:
    if pair[0] < 0 or pair[1] < pair[0]:
        raise ValueError(f"{name} must be [first, last] with 0 <= first <= last")


@dataclass(frozen=True, slots=True)
class Reference:
    """One measured object over a frame or time range."""

    id: str
    select: Selector
    frames: tuple[int, int] | None = None  # inclusive source frame indices
    t: tuple[float, float] | None = None  # inclusive, seconds (Frame.t)
    distance_m: float | None = None  # constant distance
    series: tuple[tuple[float, float], ...] | None = None  # (t_s, d_m), increasing t
    instrument: str | None = None  # key of ReferenceSet.instruments
    uncertainty_m: float | None = None  # overrides the instrument
    quantity: Quantity = "range"
    target: str = "nearest_face"  # informative: the measured face
    notes: str = ""

    def __post_init__(self) -> None:
        if (self.frames is None) == (self.t is None):
            raise ValueError("exactly one of frames, t")
        if self.frames is not None:
            _increasing(self.frames, "frames")
        if self.t is not None:
            _increasing(self.t, "t")
        if (self.distance_m is None) == (self.series is None):
            raise ValueError("exactly one of distance_m, series")
        if self.distance_m is not None and self.distance_m <= 0:
            raise ValueError("distance_m must be > 0")
        if self.series is not None:
            ts = [p[0] for p in self.series]
            if any(b <= a for a, b in itertools.pairwise(ts)) or not ts:
                raise ValueError("series times must be strictly increasing")
            if any(d <= 0 for _, d in self.series):
                raise ValueError("series distances must be > 0")
            if self.t is not None and not (ts[0] <= self.t[0] and self.t[1] <= ts[-1]):
                raise ValueError("series must cover the t range")
        if self.uncertainty_m is not None and self.uncertainty_m < 0:
            raise ValueError("uncertainty_m must be >= 0")

    def covers(self, frame: int, t: float) -> bool:
        """Whether a processed frame (index, time) is inside the range."""
        if self.frames is not None:
            return self.frames[0] <= frame <= self.frames[1]
        return self.t[0] - FRAME_TIME_EPS_S <= t <= self.t[1] + FRAME_TIME_EPS_S

    def distance_at(self, t: float) -> float:
        """Reference distance at time `t` (linear interpolation of a series).

        Raises:
            ValueError: `t` outside the series.
        """
        if self.distance_m is not None:
            return self.distance_m
        ts, ds = np.array(self.series).T
        if not ts[0] - FRAME_TIME_EPS_S <= t <= ts[-1] + FRAME_TIME_EPS_S:
            raise ValueError(f"t={t:.3f} s is outside the series [{ts[0]}, {ts[-1]}]")
        return float(np.interp(t, ts, ds))


@dataclass(frozen=True, slots=True)
class ReferenceSet:
    """The reference file of one video."""

    video: str  # file name of the video, checked against metrics.json
    references: tuple[Reference, ...]
    instruments: Mapping[str, Instrument] = field(default_factory=dict)
    version: int = 1

    def __post_init__(self) -> None:
        if self.version != 1:
            raise ValueError(f"unsupported version {self.version}")
        if not self.references:
            raise ValueError("references: at least one reference is required")
        seen: set[str] = set()
        for i, ref in enumerate(self.references):
            where = f"references[{i}]"
            if ref.id in seen:
                raise ValueError(f"{where}.id: duplicate id {ref.id!r}")
            seen.add(ref.id)
            if ref.instrument is not None and ref.instrument not in self.instruments:
                raise ValueError(
                    f"{where}.instrument: unknown instrument {ref.instrument!r}"
                )
            if ref.instrument is None and ref.uncertainty_m is None:
                raise ValueError(f"{where}: set instrument or uncertainty_m")

    def uncertainty(self, ref: Reference, distance: float) -> float:
        """Standard uncertainty (m) of `ref` at `distance`."""
        if ref.uncertainty_m is not None:
            return ref.uncertainty_m
        inst = self.instruments[ref.instrument]
        return math.hypot(inst.uncertainty_m, inst.uncertainty_rel * distance)


def load_references(path: Path) -> ReferenceSet:
    """Read and validate a reference YAML.

    Raises:
        ConfigError: missing file, invalid YAML or invalid content; the message
            starts with the file path and the dotted key.
    """
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"references not found: {path}")
    try:
        data = yaml.safe_load(path.read_text())
    except yaml.YAMLError as exc:
        raise ConfigError(f"{path}: invalid YAML ({exc})") from None
    try:
        return from_mapping(ReferenceSet, data, "")
    except ConfigError as exc:
        raise ConfigError(f"{path}: {exc}") from None
