"""Motion models (registered as family "motion") and their per-track filters.

A MotionModel is stateless, built once from its spec {type, **options} and
shared by every track; `new_filter` creates the per-track MotionFilter at
the track's first measurement (the temporal.TemporalFilter idiom). Every
filter measures the position (x, z) only, so the update is linear (Joseph
form); "cv" is a linear Kalman filter on [x, z, vx, vz] with white
acceleration per axis, "ctrv" an EKF on [x, z, v, psi, omega] (constant
turn rate and velocity) that runs a cv filter for warmup_s first, because
at v ~ 0 the CTRV Jacobian cannot see lateral motion. Below
min_speed_for_heading the heading is not observable: ctrv freezes psi and
omega and the kinematics report NaN. A new model is a registered subclass
plus its filter; the estimator does not change.

Filters work in the estimator's frame (the camera plane, or the world with
ego-motion) and only assume that it is a proper 2D rotation of the camera
plane (x, z): the heading psi is measured from the second axis toward the
first, velocity = v (sin psi, cos psi), omega > 0 turns from the second axis
toward the first (to the right on the camera plane).
"""

import copy
import math
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from typing import Any, ClassVar

import numpy as np

from det_depth_distance.core.registry import register
from det_depth_distance.temporal import white_accel_cov

# Below this |omega| (rad/s) the CTRV motion uses its straight-line limit.
MIN_YAW_RATE: float = 1e-4
# Time steps of a rollout that land this close to horizon_s still count.
_STEP_EPS: float = 1e-9


@dataclass(frozen=True, slots=True, kw_only=True)
class MotionModelOptions:
    """Options shared by every motion model (same names, so --set works across types)."""

    init_speed_std: float = 5.0  # m/s, speed std at the first measurement
    min_speed_for_heading: float = 1.0  # m/s; below it the heading is unobservable

    def __post_init__(self) -> None:
        if self.init_speed_std <= 0:
            raise ValueError(f"init_speed_std must be > 0, got {self.init_speed_std}")
        if self.min_speed_for_heading < 0:
            raise ValueError(
                f"min_speed_for_heading must be >= 0, got {self.min_speed_for_heading}"
            )


@dataclass(frozen=True, slots=True)
class Kinematics:
    """Filtered state of one track in the filter's frame."""

    pos: np.ndarray  # (2,) m
    pos_cov: np.ndarray  # (2, 2) m^2
    vel: np.ndarray  # (2,) m/s
    vel_cov: np.ndarray  # (2, 2) (m/s)^2
    heading: float  # rad, psi; NaN below min_speed_for_heading
    heading_std: float
    yaw_rate: float  # rad/s, + = turning right; NaN without a turn-rate state
    yaw_rate_std: float


class MotionFilter(ABC):
    """Filter of one track's position on the plane (one instance per track)."""

    n_updates: int  # measurements since the filter started

    @abstractmethod
    def predict(self, dt: float) -> None:
        """Advance the state by `dt` seconds (>= 0) without a measurement."""

    @abstractmethod
    def nis(self, z: np.ndarray, r: np.ndarray) -> float:
        """Normalised innovation squared of `z` against the current prediction,
        without updating (gating)."""

    @abstractmethod
    def update(self, z: np.ndarray, r: np.ndarray) -> None:
        """Incorporate the position measurement `z` (2,) with covariance `r` (2, 2)."""

    @abstractmethod
    def kinematics(self) -> Kinematics:
        """Current position, velocity, heading and turn rate with their uncertainties."""

    @abstractmethod
    def rollout(
        self, horizon_s: float, step_s: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Mean and covariance of the position every `step_s` up to `horizon_s`,
        propagated without measurements on a copy; the filter is unchanged.

        Returns:
            (t (K,), pos (K, 2), pos_cov (K, 2, 2)), t in seconds from now.
        """


def _velocity_heading(
    vel: np.ndarray, vel_cov: np.ndarray, min_speed: float
) -> tuple[float, float]:
    """(heading, std) of a velocity vector, NaN below `min_speed`."""
    vx, vz = vel
    s2 = vx**2 + vz**2
    if math.sqrt(s2) < min_speed or s2 == 0:
        return math.nan, math.nan
    grad = np.array([vz, -vx]) / s2  # d atan2(vx, vz) / d(vx, vz)
    return math.atan2(vx, vz), math.sqrt(max(grad @ vel_cov @ grad, 0.0))


def _wrap(angle: float) -> float:
    """Angle in (-pi, pi]."""
    return math.pi - (math.pi - angle) % (2 * math.pi)


class _GaussianFilter(MotionFilter):
    """Mean/covariance filter measuring the first two state components
    (H = [I2 0]): Joseph-form update and NIS here, the motion in
    `_transition`; predict and rollout are template methods over it."""

    def __init__(self, x: np.ndarray, p: np.ndarray) -> None:
        self.x = x
        self.p = p
        self.n_updates = 1

    @abstractmethod
    def _transition(
        self, x: np.ndarray, dt: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(predicted mean, Jacobian F, process noise Q) of a step of `dt` from `x`."""

    def _constrain(self) -> None:
        """Bring the state back to its canonical form (after predict and update)."""

    def _gain_mask(self) -> np.ndarray | None:
        """Rows of the gain to zero (frozen components), or None."""
        return None

    def predict(self, dt: float) -> None:
        self.x, f, q = self._transition(self.x, dt)
        self.p = f @ self.p @ f.T + q
        self._constrain()

    def nis(self, z: np.ndarray, r: np.ndarray) -> float:
        innovation = z - self.x[:2]
        return float(innovation @ np.linalg.solve(self.p[:2, :2] + r, innovation))

    def update(self, z: np.ndarray, r: np.ndarray) -> None:
        s = self.p[:2, :2] + r
        gain = np.linalg.solve(s, self.p[:2, :]).T  # P H^T S^-1 (S symmetric)
        frozen = self._gain_mask()
        if frozen is not None:
            gain[frozen] = 0.0
        self.x = self.x + gain @ (z - self.x[:2])
        # Joseph form: P stays symmetric PSD for any gain (also the masked one).
        a = np.eye(len(self.x))
        a[:, :2] -= gain
        self.p = a @ self.p @ a.T + gain @ r @ gain.T
        self.n_updates += 1
        self._constrain()

    def rollout(
        self, horizon_s: float, step_s: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        twin = copy.deepcopy(self)
        t = step_s * np.arange(1, int(horizon_s / step_s + _STEP_EPS) + 1)
        pos, cov = [], []
        for _ in t:
            twin.predict(step_s)
            pos.append(twin.x[:2].copy())
            cov.append(twin.p[:2, :2].copy())
        return t, np.array(pos).reshape(-1, 2), np.array(cov).reshape(-1, 2, 2)


class _CvFilter(_GaussianFilter):
    """Constant velocity on [x, z, vx, vz], white acceleration of variance
    accel_var per axis; starts at rest +- init_speed_std."""

    def __init__(
        self,
        z: np.ndarray,
        r: np.ndarray,
        *,
        accel_var: float,
        init_speed_var: float,
        min_speed: float,
    ) -> None:
        p = np.zeros((4, 4))
        p[:2, :2] = r
        p[2:, 2:] = init_speed_var * np.eye(2)
        super().__init__(np.array([*z, 0.0, 0.0]), p)
        self.accel_var = accel_var
        self.min_speed = min_speed

    def _transition(
        self, x: np.ndarray, dt: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        f = np.eye(4)
        f[0, 2] = f[1, 3] = dt
        axis = white_accel_cov(dt, self.accel_var)  # [pos, vel] of one axis
        q = np.zeros((4, 4))
        for i in (0, 1):
            q[np.ix_((i, i + 2), (i, i + 2))] = axis
        return f @ x, f, q

    def kinematics(self) -> Kinematics:
        vel, vel_cov = self.x[2:].copy(), self.p[2:, 2:].copy()
        heading, heading_std = _velocity_heading(vel, vel_cov, self.min_speed)
        return Kinematics(
            self.x[:2].copy(),
            self.p[:2, :2].copy(),
            vel,
            vel_cov,
            heading,
            heading_std,
            math.nan,
            math.nan,
        )


class _CtrvEkf(_GaussianFilter):
    """EKF on [x, z, v, psi, omega]: constant speed v >= 0 and turn rate omega,
    white longitudinal and yaw accelerations. Below min_speed the gain rows of
    psi and omega are zeroed (heading frozen)."""

    def __init__(
        self,
        x: np.ndarray,
        p: np.ndarray,
        *,
        accel_var: float,
        yaw_accel_var: float,
        min_speed: float,
    ) -> None:
        super().__init__(x, p)
        self.accel_var = accel_var
        self.yaw_accel_var = yaw_accel_var
        self.min_speed = min_speed
        self._constrain()

    def _transition(
        self, x: np.ndarray, dt: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        px, pz, v, psi, omega = x
        s0, c0 = math.sin(psi), math.cos(psi)
        f = np.eye(5)
        if abs(omega) < MIN_YAW_RATE:
            # First-order expansion in omega (error O(v omega^2 dt^3)):
            # continuous with the turning branch, and F stays the exact
            # Jacobian of this mean (incl. d/d omega, which the plain
            # straight line lacks).
            h = omega * dt / 2
            ds, dc = dt * (s0 + c0 * h), dt * (c0 - s0 * h)
            dx, dz = v * ds, v * dc
            f[0, 2:] = ds, v * dc, v * c0 * dt**2 / 2
            f[1, 2:] = dc, -v * ds, -v * s0 * dt**2 / 2
        else:
            s1, c1 = math.sin(psi + omega * dt), math.cos(psi + omega * dt)
            dx, dz = v / omega * (c0 - c1), v / omega * (s1 - s0)
            f[0, 2:] = (
                (c0 - c1) / omega,
                v / omega * (s1 - s0),
                -dx / omega + v / omega * s1 * dt,
            )
            f[1, 2:] = (
                (s1 - s0) / omega,
                v / omega * (c1 - c0),
                -dz / omega + v / omega * c1 * dt,
            )
        f[3, 4] = dt
        g = np.array(
            [
                [dt**2 / 2 * s0, 0.0],
                [dt**2 / 2 * c0, 0.0],
                [dt, 0.0],
                [0.0, dt**2 / 2],
                [0.0, dt],
            ]
        )
        q = g @ np.diag([self.accel_var, self.yaw_accel_var]) @ g.T
        return np.array([px + dx, pz + dz, v, psi + omega * dt, omega]), f, q

    def _constrain(self) -> None:
        if self.x[2] < 0:  # moving backwards = forwards with the opposite heading
            self.x[2] = -self.x[2]
            self.x[3] += math.pi
            flip = np.diag([1.0, 1.0, -1.0, 1.0, 1.0])
            self.p = flip @ self.p @ flip
        self.x[3] = _wrap(self.x[3])

    def _gain_mask(self) -> np.ndarray | None:
        return np.array([3, 4]) if self.x[2] < self.min_speed else None

    def kinematics(self) -> Kinematics:
        _, _, v, psi, omega = self.x
        s, c = math.sin(psi), math.cos(psi)
        jac = np.array([[s, v * c], [c, -v * s]])  # (vx, vz) / (v, psi)
        slow = v < self.min_speed
        return Kinematics(
            self.x[:2].copy(),
            self.p[:2, :2].copy(),
            np.array([v * s, v * c]),
            jac @ self.p[2:4, 2:4] @ jac.T,
            math.nan if slow else float(psi),
            math.nan if slow else math.sqrt(self.p[3, 3]),
            math.nan if slow else float(omega),
            math.nan if slow else math.sqrt(self.p[4, 4]),
        )


class _CtrvFilter(MotionFilter):
    """CTRV with a constant-velocity warm-up: a _CvFilter for the first
    warmup_s seconds, then an EKF started from its state (v = |v|,
    psi = atan2(vx, vz), covariance through the Jacobian, omega = 0 +-
    init_yaw_rate_std)."""

    def __init__(self, z: np.ndarray, r: np.ndarray, options: "Ctrv.Options") -> None:
        self.options = options
        self._age = 0.0
        self._inner: _GaussianFilter = _CvFilter(
            z,
            r,
            accel_var=options.accel_std**2,
            init_speed_var=options.init_speed_std**2,
            min_speed=options.min_speed_for_heading,
        )
        if options.warmup_s == 0:
            self._switch()

    @property
    def n_updates(self) -> int:
        return self._inner.n_updates

    @property
    def warming_up(self) -> bool:
        return isinstance(self._inner, _CvFilter)

    def _switch(self) -> None:
        cv, o = self._inner, self.options
        vx, vz = cv.x[2:]
        speed = math.hypot(vx, vz)
        t = np.zeros((5, 4))  # Jacobian of [x, z, v, psi, omega] / [x, z, vx, vz]
        t[0, 0] = t[1, 1] = 1.0
        if speed > 1e-6:
            t[2, 2:] = vx / speed, vz / speed
            t[3, 2:] = vz / speed**2, -vx / speed**2
        p = t @ cv.p @ t.T
        if speed <= 1e-6:  # direction unknown: any heading, speed as at the start
            p[2, 2] = o.init_speed_std**2
            p[3, 3] = math.pi**2 / 3
        p[4, 4] = o.init_yaw_rate_std**2
        self._inner = _CtrvEkf(
            np.array([cv.x[0], cv.x[1], speed, math.atan2(vx, vz), 0.0]),
            p,
            accel_var=o.accel_std**2,
            yaw_accel_var=o.yaw_accel_std**2,
            min_speed=o.min_speed_for_heading,
        )
        self._inner.n_updates = cv.n_updates

    def predict(self, dt: float) -> None:
        self._inner.predict(dt)
        self._age += dt

    def nis(self, z: np.ndarray, r: np.ndarray) -> float:
        return self._inner.nis(z, r)

    def update(self, z: np.ndarray, r: np.ndarray) -> None:
        self._inner.update(z, r)
        if self.warming_up and self._age >= self.options.warmup_s - 1e-9:
            self._switch()

    def kinematics(self) -> Kinematics:
        return self._inner.kinematics()

    def rollout(
        self, horizon_s: float, step_s: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self._inner.rollout(horizon_s, step_s)


class MotionModel(ABC):
    """Stateless motion model, registered and shared by every track (like
    core.aggregate.Aggregator: built from a spec, nothing to load)."""

    type_name: ClassVar[str]  # set by @register
    Options: ClassVar[type]
    # True when the model's constraints (heading, turn radius) only hold for
    # motion over ground, i.e. it needs ego-motion to be meaningful.
    needs_ego_motion: ClassVar[bool] = False

    def __init__(self, options: Any) -> None:
        self.options = options

    @abstractmethod
    def new_filter(self, z: np.ndarray, r: np.ndarray) -> MotionFilter:
        """A track's filter, initialised at its first measurement (z, r)."""

    def describe(self) -> dict[str, Any]:
        """JSON-ready description for metrics.json."""
        return {"type": self.type_name, "options": asdict(self.options)}


@register("motion", "cv")
class ConstantVelocity(MotionModel):
    """Constant velocity: people, and the default without ego-motion."""

    @dataclass(frozen=True, slots=True, kw_only=True)
    class Options(MotionModelOptions):
        accel_std: float = 2.0  # m/s^2 per axis

        def __post_init__(self) -> None:
            MotionModelOptions.__post_init__(self)
            if self.accel_std <= 0:
                raise ValueError(f"accel_std must be > 0, got {self.accel_std}")

    def new_filter(self, z: np.ndarray, r: np.ndarray) -> MotionFilter:
        o = self.options
        return _CvFilter(
            z,
            r,
            accel_var=o.accel_std**2,
            init_speed_var=o.init_speed_std**2,
            min_speed=o.min_speed_for_heading,
        )


@register("motion", "ctrv")
class Ctrv(MotionModel):
    """Constant turn rate and velocity: vehicles, with ego-motion."""

    needs_ego_motion = True

    @dataclass(frozen=True, slots=True, kw_only=True)
    class Options(MotionModelOptions):
        accel_std: float = 3.0  # m/s^2, longitudinal
        yaw_accel_std: float = 1.0  # rad/s^2
        init_yaw_rate_std: float = 0.5  # rad/s
        warmup_s: float = 0.5  # s of constant velocity before switching to ctrv

        def __post_init__(self) -> None:
            MotionModelOptions.__post_init__(self)
            for name in ("accel_std", "yaw_accel_std", "init_yaw_rate_std"):
                if getattr(self, name) <= 0:
                    raise ValueError(f"{name} must be > 0, got {getattr(self, name)}")
            if self.warmup_s < 0:
                raise ValueError(f"warmup_s must be >= 0, got {self.warmup_s}")

    def new_filter(self, z: np.ndarray, r: np.ndarray) -> MotionFilter:
        return _CtrvFilter(z, r, self.options)
