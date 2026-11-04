"""Per-track temporal smoothing of the per-frame distance (pure numpy).

Each track gets its own filter: "kalman" (constant velocity, white
piecewise acceleration of std accel_std; the only one that estimates the
relative speed and the uncertainties), "ema" (alpha from a time constant,
so stride does not change its lag in seconds) or "median" (causal window
in seconds). dt comes from frame timestamps.

The measurement std is max(std_scale * box std, rel_floor * distance): on
the task_0001 videos the box depth std overestimates the frame-to-frame
noise 1.3-3x (it also measures the object's own depth extent), and
std / sqrt(n_inliers) underestimates it 10-30x (neighbouring depth pixels
are correlated). The depth noise is also correlated in time (lag-1
correlation ~0.7), so smoothing removes jitter (3.5-4.5x) much more than
level error (~20 %); it cannot remove depth or focal bias. The raw
distance is always kept alongside.

Import rule: depends on core.types and distance only; core.pipeline and
config import SmoothingConfig from here, and motion imports distance_sigma
and white_accel_cov from here, never the reverse.
"""

import math
from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from av_perception_lab.core.types import FRAME_TIME_EPS_S, TrackEstimate
from av_perception_lab.distance import BoxDepthStats

FilterName = Literal["kalman", "ema", "median"]


@dataclass(frozen=True, slots=True)
class SmoothingConfig:
    """Smoothing parameters (YAML section "smoothing"); times in seconds.
    Defaults were chosen on the task_0001 videos (see docs/task_0006.md)."""

    filter: FilterName = "kalman"
    accel_std: float = field(  # kalman: white acceleration std q, m/s^2
        default=2.0,
        metadata={
            "ui": {
                "label": "Acceleration noise",
                "unit": "m/s²",
                "min": 0.1,
                "max": 10.0,
                "step": 0.1,
                "help": "Kalman process noise: higher follows speed changes faster.",
            }
        },
    )
    # sigma = max(std_scale * box std, rel_floor * distance)
    std_scale: float = field(
        default=0.5,
        metadata={
            "ui": {
                "label": "Box std scale",
                "min": 0.0,
                "max": 2.0,
                "step": 0.05,
                "help": "Weight of the per-box depth std in the measurement noise.",
            }
        },
    )
    rel_floor: float = field(
        default=0.01,
        metadata={
            "ui": {
                "label": "Relative noise floor",
                "min": 0.0,
                "max": 0.1,
                "step": 0.005,
                "help": "Measurement noise never goes below this fraction of the distance.",
            }
        },
    )
    init_speed_std: float = 5.0  # kalman: initial speed std, m/s
    # ema: alpha = 1 - exp(-dt / tau_s)
    tau_s: float = field(
        default=0.1,
        metadata={
            "ui": {
                "label": "EMA time constant",
                "unit": "s",
                "min": 0.01,
                "max": 2.0,
                "step": 0.01,
                "advanced": True,
                "help": "Only for the ema filter: larger smooths more.",
            }
        },
    )
    # median: causal window, samples with t_i >= t - window_s (5 frames at 30 fps)
    window_s: float = 0.15
    # state dropped after this long without a measurement of the track
    reset_after_s: float = field(
        default=1.0,
        metadata={
            "ui": {
                "label": "Reset after",
                "unit": "s",
                "min": 0.1,
                "max": 10.0,
                "step": 0.1,
                "help": "Drop a track's filter state after this long without a measurement.",
            }
        },
    )

    def __post_init__(self) -> None:
        if self.filter not in FILTERS:
            raise ValueError(
                f"filter must be one of {list(FILTERS)}, got {self.filter!r}"
            )
        if self.std_scale < 0 or self.rel_floor < 0:
            raise ValueError("std_scale and rel_floor must be >= 0")
        if self.std_scale == 0 and self.rel_floor == 0:
            raise ValueError("std_scale and rel_floor cannot both be 0")
        positive = ("accel_std", "init_speed_std", "tau_s", "window_s", "reset_after_s")
        for name in positive:
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be > 0, got {getattr(self, name)}")


def distance_sigma(
    distance: float, std: float, *, std_scale: float, rel_floor: float
) -> float:
    """Measurement std (m) of a distance: max(std_scale * std, rel_floor * |distance|);
    a non-finite `std` leaves the relative floor only."""
    scaled = std_scale * std if math.isfinite(std) else 0.0
    return max(scaled, rel_floor * abs(distance))


def measurement_sigma(distance: float, std: float, cfg: SmoothingConfig) -> float:
    """distance_sigma with the smoothing section's std_scale and rel_floor."""
    return distance_sigma(
        distance, std, std_scale=cfg.std_scale, rel_floor=cfg.rel_floor
    )


def white_accel_cov(dt: float, accel_var: float) -> np.ndarray:
    """Process noise (2, 2) of one position/velocity axis under white
    piecewise-constant acceleration of variance `accel_var` over `dt` seconds."""
    return accel_var * np.array(
        [[dt**4 / 4, dt**3 / 2], [dt**3 / 2, dt**2]], dtype=np.float64
    )


class TemporalFilter(ABC):
    """Filter of one track's distance series (one instance per track)."""

    @abstractmethod
    def update(self, t: float, z: float | None, sigma: float) -> TrackEstimate:
        """Advance to time `t` and incorporate measurement `z` (metres) with
        standard deviation `sigma`; z None means no measurement this frame.

        `t` is non-decreasing; the first call carries a measurement
        (TrackSmoother creates a filter at its track's first distance).

        Returns:
            TrackEstimate. Filters without a state model (ema, median) return NaN
            for distance_std, rel_speed and rel_speed_std.
        """


class KalmanFilter1D(TemporalFilter):
    """Constant-velocity Kalman filter on (distance, rate of change).

    Process noise: white piecewise-constant acceleration of std `accel_std`,
    Q = q^2 [[dt^4/4, dt^3/2], [dt^3/2, dt^2]]. Initialised at the first
    measurement with speed 0 +- init_speed_std. A missing measurement only
    predicts (uncertainty grows). No innovation gating: a gate without
    re-initialisation was seen to lock out and diverge.
    """

    def __init__(self, cfg: SmoothingConfig) -> None:
        self.q2 = cfg.accel_std**2
        self.init_speed_var = cfg.init_speed_std**2
        self.x: np.ndarray | None = None  # [distance, rate]
        self.p = np.zeros((2, 2))
        self.t = 0.0
        self.n_updates = 0

    def update(self, t: float, z: float | None, sigma: float) -> TrackEstimate:
        if self.x is None:
            self.x = np.array([z, 0.0])
            self.p = np.diag([sigma**2, self.init_speed_var])
            self.n_updates = 1
        else:
            dt = t - self.t
            f = np.array([[1.0, dt], [0.0, 1.0]])
            self.x = f @ self.x
            self.p = f @ self.p @ f.T + white_accel_cov(dt, self.q2)
            if z is not None:
                r = sigma**2
                gain = self.p[:, 0] / (self.p[0, 0] + r)
                self.x = self.x + gain * (z - self.x[0])
                # Joseph form: keeps P symmetric positive semi-definite.
                a = np.eye(2) - np.outer(gain, [1.0, 0.0])
                self.p = a @ self.p @ a.T + r * np.outer(gain, gain)
                self.n_updates += 1
        self.t = t
        return TrackEstimate(
            float(self.x[0]),
            math.sqrt(self.p[0, 0]),
            float(self.x[1]),
            math.sqrt(self.p[1, 1]),
            self.n_updates,
        )


class EmaFilter(TemporalFilter):
    """Exponential moving average with alpha = 1 - exp(-dt / tau_s): the lag in
    seconds does not depend on the frame rate or stride. A missing measurement
    keeps the value."""

    def __init__(self, cfg: SmoothingConfig) -> None:
        self.tau_s = cfg.tau_s
        self.value: float | None = None
        self.t = 0.0
        self.n_updates = 0

    def update(self, t: float, z: float | None, sigma: float) -> TrackEstimate:
        if z is not None:
            if self.value is None:
                self.value = z
            else:
                alpha = 1.0 - math.exp(-(t - self.t) / self.tau_s)
                self.value += alpha * (z - self.value)
            self.t = t
            self.n_updates += 1
        return TrackEstimate(self.value, math.nan, math.nan, math.nan, self.n_updates)


class MedianFilter(TemporalFilter):
    """Causal median of the measurements with t_i >= t - window_s. A missing
    measurement keeps the value."""

    def __init__(self, cfg: SmoothingConfig) -> None:
        self.window_s = cfg.window_s
        self.samples: deque[tuple[float, float]] = deque()
        self.value = math.nan
        self.n_updates = 0

    def update(self, t: float, z: float | None, sigma: float) -> TrackEstimate:
        if z is not None:
            self.samples.append((t, z))
            while self.samples[0][0] < t - self.window_s - FRAME_TIME_EPS_S:
                self.samples.popleft()
            self.value = float(np.median([v for _, v in self.samples]))
            self.n_updates += 1
        return TrackEstimate(self.value, math.nan, math.nan, math.nan, self.n_updates)


# Filter name -> constructor; a new filter is a TemporalFilter subclass plus
# an entry here (and in FilterName).
FILTERS: dict[str, Callable[[SmoothingConfig], TemporalFilter]] = {
    "kalman": KalmanFilter1D,
    "ema": EmaFilter,
    "median": MedianFilter,
}


class TrackSmoother:
    """One TemporalFilter per track id. State of a track without a measurement
    for more than reset_after_s is dropped, so a reused or returning id starts a
    fresh filter. Independent of the tracker: it only needs integer ids."""

    def __init__(self, cfg: SmoothingConfig) -> None:
        self.cfg = cfg
        # track id -> (filter, time of its last measurement)
        self._filters: dict[int, tuple[TemporalFilter, float]] = {}

    def reset(self) -> None:
        """Forget every track's state."""
        self._filters.clear()

    def update(
        self, t: float, track_ids: np.ndarray | None, stats: Sequence[BoxDepthStats]
    ) -> tuple[TrackEstimate | None, ...]:
        """Estimates aligned with `stats` at time `t` (seconds).

        `track_ids` None (no tracker output) gives all None; so does an instance
        whose track has had no distance yet (or none for reset_after_s).
        """
        if track_ids is None:
            return (None,) * len(stats)
        limit = t - self.cfg.reset_after_s - FRAME_TIME_EPS_S  # inclusive
        self._filters = {
            tid: entry for tid, entry in self._filters.items() if entry[1] >= limit
        }
        out: list[TrackEstimate | None] = []
        for tid, s in zip(track_ids.tolist(), stats, strict=True):
            entry = self._filters.get(tid)
            z = s.distance
            if z is None:
                out.append(
                    None if entry is None else entry[0].update(t, None, math.nan)
                )
                continue
            filt = entry[0] if entry else FILTERS[self.cfg.filter](self.cfg)
            out.append(filt.update(t, z, measurement_sigma(z, s.std, self.cfg)))
            self._filters[tid] = (filt, t)
        return tuple(out)
