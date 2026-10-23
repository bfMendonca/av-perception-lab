"""Optional ego-motion (registered as family "ego_motion").

A provider gives the vehicle pose in a world frame (REP-103 2D) at a frame
time; with the camera mount it yields the camera -> world transform, so the
estimator filters in the world (absolute motion) and reports in the current
camera frame. "csv" reads t, x, y, yaw (s, m, m, rad) sampled on the video
clock (+ time_offset_s), interpolated linearly with unwrapped yaw; frames
outside the file's time range hold the edge pose with zero velocity (one
[warn] on stderr, counted in describe()["clamped_frames"]). Nothing is
extrapolated.
"""

import csv
import math
import sys
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, ClassVar

import numpy as np

from det_depth_distance.core.options import ConfigError
from det_depth_distance.core.registry import register
from det_depth_distance.motion.config import CameraMount

# Camera plane (x right, z forward) -> vehicle frame (x forward, y left).
_VEHICLE_FROM_CAMERA = np.array([[0.0, 1.0], [-1.0, 0.0]])
# d/dt of a CCW rotation by theta: J R theta_dot.
_J = np.array([[0.0, -1.0], [1.0, 0.0]])
_CSV_COLUMNS: tuple[str, ...] = ("t", "x", "y", "yaw")
_TIME_EPS_S: float = 1e-6  # frames this close to the data's ends are inside


@dataclass(frozen=True, slots=True)
class EgoPose:
    """Vehicle pose and velocity in the world frame (REP-103 2D)."""

    x: float  # m
    y: float  # m
    yaw: float  # rad, CCW
    vx: float  # m/s
    vy: float  # m/s
    yaw_rate: float  # rad/s, CCW


def _rotation(theta: float) -> np.ndarray:
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s], [s, c]])


@dataclass(frozen=True, slots=True)
class PlanarTransform:
    """Rigid 2D transform p_to = r @ p_from + t."""

    r: np.ndarray  # (2, 2) proper rotation
    t: np.ndarray  # (2,)

    @classmethod
    def identity(cls) -> "PlanarTransform":
        return cls(np.eye(2), np.zeros(2))

    def apply(self, p: np.ndarray) -> np.ndarray:
        """Points (..., 2)."""
        return np.asarray(p) @ self.r.T + self.t

    def rotate_cov(self, cov: np.ndarray) -> np.ndarray:
        """Covariances (..., 2, 2) (a translation does not change them)."""
        return self.r @ np.asarray(cov) @ self.r.T

    def inverse(self) -> "PlanarTransform":
        return PlanarTransform(self.r.T, -self.r.T @ self.t)


def camera_to_world(pose: EgoPose, mount: CameraMount) -> PlanarTransform:
    """Camera plane (x, z) -> world, composing the vehicle pose and the mount."""
    r_wv = _rotation(pose.yaw)
    r_vc = _rotation(math.radians(mount.yaw_deg)) @ _VEHICLE_FROM_CAMERA
    t_vc = np.array([mount.x_m, mount.y_m])
    return PlanarTransform(r_wv @ r_vc, r_wv @ t_vc + np.array([pose.x, pose.y]))


def camera_velocity(pose: EgoPose, mount: CameraMount) -> np.ndarray:
    """(2,) world velocity of the camera origin (vehicle velocity plus the
    rotation of the mount's lever arm)."""
    lever = _rotation(pose.yaw) @ np.array([mount.x_m, mount.y_m])
    return np.array([pose.vx, pose.vy]) + pose.yaw_rate * (_J @ lever)


class EgoMotionProvider(ABC):
    """Vehicle pose over time; built from a spec, `open` reads its data."""

    type_name: ClassVar[str]  # set by @register
    Options: ClassVar[type]

    def __init__(self, options: Any) -> None:
        self.options = options

    def open(self) -> None:
        """Read the data (called by build_motion, before any model loads).

        Raises:
            ConfigError: unusable data; the message starts with the option name.
        """

    @abstractmethod
    def pose_at(self, t: float) -> EgoPose:
        """Pose at frame time `t` (Frame.t, seconds)."""

    def describe(self) -> dict[str, Any]:
        """JSON-ready description for metrics.json."""
        return {"type": self.type_name, "options": _plain(asdict(self.options))}


def _plain(options: dict[str, Any]) -> dict[str, Any]:
    return {k: str(v) if isinstance(v, Path) else v for k, v in options.items()}


@register("ego_motion", "csv")
class CsvEgoMotion(EgoMotionProvider):
    """Poses from a CSV with header t, x, y, yaw (s, m, m, rad CCW), t strictly
    increasing on the video clock: CSV time = Frame.t + time_offset_s."""

    @dataclass(frozen=True, slots=True)
    class Options:
        path: Path
        time_offset_s: float = 0.0

    def __init__(self, options: Any) -> None:
        super().__init__(options)
        self.clamped_frames = 0
        self._data: np.ndarray | None = None  # (N, 4) t, x, y, unwrapped yaw

    def open(self) -> None:
        path = self.options.path
        if not path.is_file():
            raise ConfigError(f"path: {path}: file not found")
        with open(path, newline="") as fh:
            reader = csv.DictReader(fh)
            missing = [c for c in _CSV_COLUMNS if c not in (reader.fieldnames or [])]
            if missing:
                raise ConfigError(f"path: {path}: missing columns {', '.join(missing)}")
            try:
                rows = [[float(row[c]) for c in _CSV_COLUMNS] for row in reader]
            except (TypeError, ValueError) as exc:
                raise ConfigError(f"path: {path}: non-numeric value ({exc})") from None
        data = np.array(rows, dtype=np.float64).reshape(-1, 4)
        if len(data) < 2:
            raise ConfigError(f"path: {path}: needs at least 2 samples")
        if not np.isfinite(data).all():
            raise ConfigError(f"path: {path}: non-finite value")
        if (np.diff(data[:, 0]) <= 0).any():
            raise ConfigError(f"path: {path}: t must be strictly increasing")
        data[:, 3] = np.unwrap(data[:, 3])
        self._data = data

    def pose_at(self, t: float) -> EgoPose:
        data = self._data
        tc = t + self.options.time_offset_s
        t0, t1 = data[0, 0], data[-1, 0]
        if not t0 - _TIME_EPS_S <= tc <= t1 + _TIME_EPS_S:
            if self.clamped_frames == 0:
                print(
                    f"[warn] motion.ego_motion: t={tc:.3f} s outside "
                    f"[{t0:g}, {t1:g}] s, holding the edge pose",
                    file=sys.stderr,
                )
            self.clamped_frames += 1
            _, x, y, yaw = (data[0] if tc < t0 else data[-1]).tolist()
            return EgoPose(x, y, yaw, 0.0, 0.0, 0.0)
        i = int(np.clip(np.searchsorted(data[:, 0], tc) - 1, 0, len(data) - 2))
        (ta, *a), (tb, *b) = data[i], data[i + 1]
        frac = (tc - ta) / (tb - ta)
        x, y, yaw = (pa + frac * (pb - pa) for pa, pb in zip(a, b, strict=True))
        vx, vy, yaw_rate = ((pb - pa) / (tb - ta) for pa, pb in zip(a, b, strict=True))
        return EgoPose(
            float(x), float(y), float(yaw), float(vx), float(vy), float(yaw_rate)
        )

    def describe(self) -> dict[str, Any]:
        return {**super().describe(), "clamped_frames": self.clamped_frames}
