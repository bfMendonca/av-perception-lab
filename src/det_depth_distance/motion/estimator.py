"""One motion filter per track id (the TrackSmoother idiom, see temporal.py).

The model is fixed at the track's birth from its canonical label (classes,
then default; None = no estimate). A track without a distance this frame is
only predicted; a measurement whose NIS exceeds nis_gate restarts the filter
(id switch). State is dropped after reset_after_s without a measurement
(inclusive, FRAME_TIME_EPS_S). Outputs and trajectories are in the camera
frame of the current frame; abs_speed only with ego-motion.
"""

import math
import sys
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from det_depth_distance.core import registry
from det_depth_distance.core.options import ConfigError
from det_depth_distance.core.types import (
    FRAME_TIME_EPS_S,
    DepthResult,
    Instances,
    MotionEstimate,
    Trajectory,
)
from det_depth_distance.distance import BoxDepthStats, DistanceType
from det_depth_distance.motion.config import MotionConfig
from det_depth_distance.motion.ego import (
    EgoMotionProvider,
    PlanarTransform,
    camera_to_world,
    camera_velocity,
)
from det_depth_distance.motion.measurement import measure
from det_depth_distance.motion.models import Kinematics, MotionFilter, MotionModel

_J = np.array([[0.0, -1.0], [1.0, 0.0]])


@dataclass(slots=True)
class _Track:
    """State of one track id."""

    model: str
    filter: MotionFilter
    t: float  # time the filter was last advanced to
    last_t: float  # time of its last measurement
    y: float  # m, camera y of the last measured reference point
    restarts: int = 0


@dataclass(frozen=True, slots=True)
class _Frame:
    """Filter frame of one processed frame: camera -> filter transform and,
    with ego-motion, the camera's world velocity and yaw rate."""

    to_filter: PlanarTransform
    camera_velocity: np.ndarray | None  # None = filtering in the camera frame
    yaw_rate: float


class MotionEstimator:
    """Per-track motion estimates aligned with the instances of each frame."""

    def __init__(
        self,
        cfg: MotionConfig,
        models: Mapping[str, MotionModel],
        *,
        distance_type: DistanceType,
        ego: EgoMotionProvider | None = None,
    ) -> None:
        self.cfg = cfg
        self.models = dict(models)
        self.distance_type = distance_type
        self.ego = ego
        self._tracks: dict[int, _Track] = {}

    def reset(self) -> None:
        """Forget every track's state."""
        self._tracks.clear()

    def update(
        self,
        t: float,
        instances: Instances,
        stats: Sequence[BoxDepthStats],
        depth: DepthResult,
    ) -> tuple[MotionEstimate | None, ...]:
        """Estimates aligned with `instances` at time `t` (seconds).

        `instances.track_id` None (no tracker output) gives all None; so does an
        instance whose class has no model or whose track has had no distance
        yet (or none for reset_after_s).
        """
        frame = self._frame(t)
        if instances.track_id is None:
            return (None,) * len(instances)
        limit = t - self.cfg.reset_after_s - FRAME_TIME_EPS_S  # inclusive
        self._tracks = {
            tid: track for tid, track in self._tracks.items() if track.last_t >= limit
        }
        out: list[MotionEstimate | None] = []
        for i, tid in enumerate(instances.track_id.tolist()):
            track = self._tracks.get(tid)
            m = measure(
                instances.xyxy[i],
                stats[i],
                depth.intrinsics,
                depth.geometry,
                self.cfg.measurement,
                self.distance_type,
            )
            if m is None:
                if track is None:
                    out.append(None)
                    continue
                track.filter.predict(t - track.t)
                track.t = t
                out.append(self._estimate(track, frame, math.nan, restarted=False))
                continue
            z = frame.to_filter.apply(m.xz)
            r = frame.to_filter.rotate_cov(m.cov)
            nis, restarted = math.nan, False
            if track is None:
                name = self.cfg.classes.get(instances.label[i], self.cfg.default)
                if name is None:
                    out.append(None)
                    continue
                track = self._tracks[tid] = _Track(
                    name, self.models[name].new_filter(z, r), t, t, m.y
                )
            else:
                track.filter.predict(t - track.t)
                nis = track.filter.nis(z, r)
                if nis > self.cfg.nis_gate:  # id switch or jump: start over here
                    track.filter = self.models[track.model].new_filter(z, r)
                    track.restarts += 1
                    restarted = True
                else:
                    track.filter.update(z, r)
                track.t = track.last_t = t
                track.y = m.y
            out.append(self._estimate(track, frame, nis, restarted=restarted))
        return tuple(out)

    def _frame(self, t: float) -> _Frame:
        if self.ego is None:
            return _Frame(PlanarTransform.identity(), None, 0.0)
        pose = self.ego.pose_at(t)
        mount = self.cfg.camera_mount
        return _Frame(
            camera_to_world(pose, mount), camera_velocity(pose, mount), pose.yaw_rate
        )

    def _estimate(
        self, track: _Track, frame: _Frame, nis: float, *, restarted: bool
    ) -> MotionEstimate:
        """The track's filtered state, back in the current camera frame."""
        k: Kinematics = track.filter.kinematics()
        to_camera = frame.to_filter.inverse()
        pos = to_camera.apply(k.pos)
        pos_cov = to_camera.rotate_cov(k.pos_cov)
        vel_cov = to_camera.rotate_cov(k.vel_cov)
        heading = k.heading
        if frame.camera_velocity is None:
            vel, abs_speed = k.vel, math.nan
        else:
            # d/dt of the position seen from the moving, rotating camera.
            vel = to_camera.r @ (k.vel - frame.camera_velocity)
            vel = vel - frame.yaw_rate * (_J @ pos)
            abs_speed = float(np.hypot(*k.vel))
            if math.isfinite(heading):
                sx, sz = to_camera.r @ np.array([math.sin(heading), math.cos(heading)])
                heading = math.atan2(sx, sz)
        speed = float(np.hypot(*vel))
        grad = vel / speed if speed > 0 else np.zeros(2)
        t, xz, cov = track.filter.rollout(
            self.cfg.prediction.horizon_s, self.cfg.prediction.step_s
        )
        return MotionEstimate(
            model=track.model,
            x=float(pos[0]),
            z=float(pos[1]),
            x_std=math.sqrt(pos_cov[0, 0]),
            z_std=math.sqrt(pos_cov[1, 1]),
            vx=float(vel[0]),
            vz=float(vel[1]),
            speed=speed,
            speed_std=math.sqrt(max(grad @ vel_cov @ grad, 0.0)),
            heading_deg=math.degrees(heading),
            heading_std_deg=math.degrees(k.heading_std),
            yaw_rate_dps=math.degrees(k.yaw_rate),
            yaw_rate_std_dps=math.degrees(k.yaw_rate_std),
            abs_speed=abs_speed,
            nis=nis,
            n_updates=track.filter.n_updates,
            restarted=restarted,
            trajectory=Trajectory(
                t, to_camera.apply(xz), to_camera.rotate_cov(cov), track.y
            ),
        )

    def describe(self) -> dict[str, Any]:
        """JSON-ready description for metrics.json["pipeline"]["motion"]."""
        cfg = self.cfg
        return {
            "models": {name: m.describe() for name, m in self.models.items()},
            "classes": dict(cfg.classes),
            "default": cfg.default,
            "reset_after_s": cfg.reset_after_s,
            "nis_gate": cfg.nis_gate,
            "measurement": asdict(cfg.measurement),
            "prediction": asdict(cfg.prediction),
            "ego_motion": self.ego.describe() if self.ego is not None else None,
            "camera_mount": asdict(cfg.camera_mount),
        }


def build_motion(cfg: MotionConfig, *, distance_type: DistanceType) -> MotionEstimator:
    """Build the estimator from the YAML section: every model through
    registry.build("motion", ...) and the ego provider through
    registry.build("ego_motion", ...) + open().

    Warns once on stderr per model that needs ego-motion (ctrv) configured
    without it: on the camera plane the own vehicle's turns read as the object's.

    Raises:
        ConfigError: invalid model or ego-motion spec, or unusable ego data
            (dotted path), before any model is loaded.
    """
    models = {
        name: registry.build("motion", spec, path=f"motion.models.{name}")
        for name, spec in cfg.models.items()
    }
    ego = None
    if cfg.ego_motion is not None:
        ego = registry.build("ego_motion", cfg.ego_motion, path="motion.ego_motion")
        try:
            ego.open()
        except ConfigError as exc:
            raise ConfigError(f"motion.ego_motion.{exc}") from None
    else:
        for name, model in models.items():
            if model.needs_ego_motion:
                print(
                    f"[warn] motion.models.{name}: {model.type_name} without "
                    "ego_motion reads the own vehicle's turns as the object's; "
                    "prefer cv",
                    file=sys.stderr,
                )
    return MotionEstimator(cfg, models, distance_type=distance_type, ego=ego)
