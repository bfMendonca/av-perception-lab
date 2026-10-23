"""Motion models of task_0009 (motion/models.py): consistency of the cv
filter (NEES/NIS on its own process model), the ctrv EKF (constant turn,
Jacobian, omega -> 0 limit, warm-up), heading observability, stride
invariance, rollout and covariance hygiene."""

import math
import subprocess
import sys

import numpy as np
import pytest

from det_depth_distance.core.options import ConfigError
from det_depth_distance.core.registry import build
from det_depth_distance.motion.models import (
    MIN_YAW_RATE,
    ConstantVelocity,
    Ctrv,
    _CtrvEkf,
    _CvFilter,
)


def cv(**options):
    return ConstantVelocity(ConstantVelocity.Options(**options))


def ctrv(**options):
    return Ctrv(Ctrv.Options(**options))


def r_of(sigma):
    return sigma**2 * np.eye(2)


def turn_truth(t, *, x0=0.0, z0=20.0, v=10.0, psi0=0.0, omega=0.2):
    """Position on a constant turn (omega > 0 = to the right, toward +x)."""
    psi = psi0 + omega * t
    return np.array(
        [
            x0 + v / omega * (math.cos(psi0) - math.cos(psi)),
            z0 + v / omega * (math.sin(psi) - math.sin(psi0)),
        ]
    )


def run(model, truth, *, fps, duration_s, sigma, rng):
    """Filter a noisy position series; returns the filter and its kinematics."""
    times = np.arange(0.0, duration_s, 1.0 / fps)
    r = r_of(sigma)
    filt, out = None, []
    for k, t in enumerate(times):
        z = truth(t) + rng.normal(0.0, sigma, 2)
        if filt is None:
            filt = model.new_filter(z, r)
        else:
            filt.predict(t - times[k - 1])
            filt.update(z, r)
        out.append(filt.kinematics())
    return filt, out


def test_cv_nees_and_nis_are_consistent():
    # The truth follows the filter's own process model (white piecewise
    # acceleration), so the mean NEES (4 dof) and NIS (2 dof) match their dof.
    accel_std, speed_std, sigma, dt = 1.5, 2.0, 0.3, 0.1
    model = cv(accel_std=accel_std, init_speed_std=speed_std)
    rng = np.random.default_rng(0)
    g = np.array([dt**2 / 2, dt])
    nees, nis = [], []
    for _ in range(50):
        state = np.array([0.0, 20.0, *rng.normal(0.0, speed_std, 2)])
        filt = model.new_filter(state[:2] + rng.normal(0, sigma, 2), r_of(sigma))
        for k in range(1, 60):
            a = rng.normal(0.0, accel_std, 2)
            state = np.array(
                [
                    state[0] + state[2] * dt + g[0] * a[0],
                    state[1] + state[3] * dt + g[0] * a[1],
                    state[2] + g[1] * a[0],
                    state[3] + g[1] * a[1],
                ]
            )
            z = state[:2] + rng.normal(0, sigma, 2)
            filt.predict(dt)
            if k > 10:
                nis.append(filt.nis(z, r_of(sigma)))
            filt.update(z, r_of(sigma))
            if k > 10:
                e = state - filt.x
                nees.append(e @ np.linalg.solve(filt.p, e))
    assert 0.85 * 4 < np.mean(nees) < 1.15 * 4
    assert 0.85 * 2 < np.mean(nis) < 1.15 * 2


def test_cv_follows_a_pedestrian_turning_back():
    # 1.5 m/s to the right for 3 s, then 1.5 m/s to the left.
    def truth(t):
        x = 1.5 * t if t < 3.0 else 4.5 - 1.5 * (t - 3.0)
        return np.array([x, 12.0])

    _, out = run(
        cv(accel_std=1.0, init_speed_std=2.0, min_speed_for_heading=0.3),
        truth,
        fps=10,
        duration_s=6.0,
        sigma=0.1,
        rng=np.random.default_rng(1),
    )
    before, after = out[29], out[-1]  # t = 2.9 s and 5.9 s
    assert before.vel[0] == pytest.approx(1.5, abs=0.3)
    assert after.vel[0] == pytest.approx(-1.5, abs=0.3)
    assert math.degrees(after.heading) == pytest.approx(-90.0, abs=15.0)


def test_ctrv_tracks_a_constant_turn():
    filt, out = run(
        ctrv(warmup_s=0.5),
        turn_truth,
        fps=10,
        duration_s=6.0,
        sigma=0.2,
        rng=np.random.default_rng(2),
    )
    assert not filt.warming_up
    k = out[-1]
    t = 5.9
    assert np.hypot(*k.vel) == pytest.approx(10.0, abs=0.5)
    # The turn rate is a second derivative of noisy positions: one frame is
    # within its reported std, the mean over the last 3 s close to the truth.
    assert abs(k.yaw_rate - 0.2) < 2 * k.yaw_rate_std
    assert np.mean([e.yaw_rate for e in out[30:]]) == pytest.approx(0.2, abs=0.03)
    assert abs(k.heading - 0.2 * t) < 3 * k.heading_std
    np.testing.assert_allclose(k.pos, turn_truth(t), atol=0.3)
    # The rollout keeps turning: 1 s ahead, the true position lies inside the
    # predicted covariance (chi2(2) 99 %).
    times, pos, cov = filt.rollout(1.0, 0.5)
    np.testing.assert_allclose(times, [0.5, 1.0])
    e = turn_truth(t + 1.0) - pos[-1]
    assert e @ np.linalg.solve(cov[-1], e) < 9.21


@pytest.mark.parametrize("omega", [0.3, 0.0])
def test_ctrv_jacobian_matches_finite_differences(omega):
    ekf = _CtrvEkf(
        np.zeros(5), np.eye(5), accel_var=1.0, yaw_accel_var=1.0, min_speed=1.0
    )
    x = np.array([1.0, 20.0, 8.0, 0.4, omega])
    dt, h = 0.1, 1e-6
    _, f, _ = ekf._transition(x, dt)
    numeric = np.empty((5, 5))
    for j in range(5):
        step = np.zeros(5)
        step[j] = h
        numeric[:, j] = (
            ekf._transition(x + step, dt)[0] - ekf._transition(x - step, dt)[0]
        ) / (2 * h)
    np.testing.assert_allclose(f, numeric, atol=1e-4)


def test_ctrv_zero_yaw_rate_is_continuous():
    ekf = _CtrvEkf(
        np.zeros(5), np.eye(5), accel_var=1.0, yaw_accel_var=1.0, min_speed=1.0
    )
    x = np.array([1.0, 20.0, 8.0, 0.4, 0.0])
    below, above = x.copy(), x.copy()
    below[4], above[4] = 0.99 * MIN_YAW_RATE, 1.01 * MIN_YAW_RATE
    xb, fb, _ = ekf._transition(below, 0.5)
    xa, fa, _ = ekf._transition(above, 0.5)
    np.testing.assert_allclose(xb[:2], xa[:2], atol=1e-5)
    np.testing.assert_allclose(fb, fa, atol=1e-4)


def test_ctrv_warmup_switches_from_cv():
    filt, out = run(
        ctrv(warmup_s=0.5),
        lambda t: np.array([0.0, 20.0 + 8.0 * t]),
        fps=10,
        duration_s=1.0,
        sigma=0.1,
        rng=np.random.default_rng(3),
    )
    # Six measurements (0.5 s of predictions) run the cv filter; then the EKF.
    assert all(math.isnan(k.yaw_rate) for k in out[:5])
    assert all(math.isfinite(k.yaw_rate) for k in out[5:])
    assert not filt.warming_up and filt.n_updates == 10
    # Converted state: speed and heading of the cv velocity (moving away).
    assert np.hypot(*out[-1].vel) == pytest.approx(8.0, abs=0.5)
    assert abs(out[-1].heading) < 3 * out[-1].heading_std
    assert ctrv(warmup_s=0.0).new_filter(np.zeros(2), r_of(0.1)).warming_up is False


@pytest.mark.parametrize("model", [cv(), ctrv(warmup_s=0.0)], ids=["cv", "ctrv"])
def test_heading_is_nan_below_min_speed(model):
    _, out = run(
        model,
        lambda t: np.array([2.0, 15.0]),
        fps=10,
        duration_s=2.0,
        sigma=0.05,
        rng=np.random.default_rng(4),
    )
    k = out[-1]
    assert np.hypot(*k.vel) < 1.0
    assert math.isnan(k.heading) and math.isnan(k.heading_std)
    assert math.isnan(k.yaw_rate)


def test_speed_is_stride_invariant():
    speeds = []
    for fps in (30, 10):
        _, out = run(
            cv(),
            lambda t: np.array([3.0 * t, 20.0 - 4.0 * t]),
            fps=fps,
            duration_s=3.0,
            sigma=1e-3,
            rng=np.random.default_rng(5),
        )
        speeds.append(np.hypot(*out[-1].vel))
    assert speeds[0] == pytest.approx(5.0, abs=0.05)
    assert speeds[1] == pytest.approx(speeds[0], abs=0.05)


@pytest.mark.parametrize("model", [cv(), ctrv(warmup_s=0.2)], ids=["cv", "ctrv"])
def test_rollout_does_not_change_the_filter(model):
    filt, _ = run(
        model,
        lambda t: np.array([1.0 * t, 20.0 + 5.0 * t]),
        fps=10,
        duration_s=2.0,
        sigma=0.1,
        rng=np.random.default_rng(6),
    )
    inner = filt if isinstance(filt, _CvFilter) else filt._inner
    x, p, n = inner.x.copy(), inner.p.copy(), filt.n_updates
    t, pos, cov = filt.rollout(3.0, 0.25)
    np.testing.assert_array_equal(inner.x, x)
    np.testing.assert_array_equal(inner.p, p)
    assert filt.n_updates == n
    assert t.shape == (12,) and pos.shape == (12, 2) and cov.shape == (12, 2, 2)
    np.testing.assert_allclose(t[[0, -1]], [0.25, 3.0])
    # The straight truth 3 s ahead lies inside the predicted covariance
    # (chi2(2) 99 %); the cv mean is exactly the constant-velocity line.
    e = np.array([1.0 * 4.9, 20.0 + 5.0 * 4.9]) - pos[-1]
    assert e @ np.linalg.solve(cov[-1], e) < 9.21
    k = filt.kinematics()
    if isinstance(filt, _CvFilter):
        np.testing.assert_allclose(pos[-1], k.pos + 3.0 * k.vel)
    traces = np.trace(cov, axis1=1, axis2=2)
    assert (np.diff(traces) > 0).all()


@pytest.mark.parametrize("model", [cv(), ctrv(warmup_s=0.3)], ids=["cv", "ctrv"])
def test_covariances_are_symmetric_psd(model):
    filt, out = run(
        model,
        turn_truth,
        fps=10,
        duration_s=4.0,
        sigma=0.3,
        rng=np.random.default_rng(7),
    )
    inner = filt if isinstance(filt, _CvFilter) else filt._inner
    _, _, rollout_cov = filt.rollout(3.0, 0.5)
    for c in [
        inner.p,
        *(k.pos_cov for k in out),
        *(k.vel_cov for k in out),
        *rollout_cov,
    ]:
        np.testing.assert_allclose(c, c.T, atol=1e-9)
        assert np.linalg.eigvalsh(c).min() > -1e-9


def test_ctrv_negative_speed_flips_the_heading():
    ekf = _CtrvEkf(
        np.array([0.0, 10.0, -2.0, 0.5, 0.0]),
        np.diag([1.0, 1.0, 1.0, 0.1, 0.1]) + 0.05,
        accel_var=1.0,
        yaw_accel_var=1.0,
        min_speed=1.0,
    )
    assert ekf.x[2] == pytest.approx(2.0)
    assert ekf.x[3] == pytest.approx(0.5 - math.pi)
    assert ekf.p[2, 3] == pytest.approx(-0.05)  # sign of the v cross terms flipped


def test_options_are_validated_with_their_path():
    assert isinstance(build("motion", {"type": "cv"}, path="m"), ConstantVelocity)
    with pytest.raises(ConfigError, match=r"^motion\.models\.v: accel_std must be > 0"):
        build("motion", {"type": "cv", "accel_std": 0}, path="motion.models.v")
    with pytest.raises(ConfigError, match=r"warmup_s must be >= 0"):
        build("motion", {"type": "ctrv", "warmup_s": -1}, path="m")
    with pytest.raises(ConfigError, match=r"init_speed_std must be > 0"):
        build("motion", {"type": "ctrv", "init_speed_std": 0}, path="m")
    with pytest.raises(ConfigError, match=r"^m\.yaw_accel_std: unknown option"):
        build("motion", {"type": "cv", "yaw_accel_std": 1.0}, path="m")
    assert Ctrv.needs_ego_motion and not ConstantVelocity.needs_ego_motion


def test_motion_package_imports_no_framework():
    # Like temporal/tracking (test_temporal.py): numpy only, no GPU stack,
    # no scipy, not even OpenCV/PIL.
    code = (
        "import sys, det_depth_distance.motion, det_depth_distance.motion.models, "
        "det_depth_distance.motion.ego; "
        "print(sorted({'torch', 'ultralytics', 'scipy', 'cv2', 'PIL'} "
        "& set(sys.modules)))"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], check=False, capture_output=True, text=True
    )
    assert out.stdout.strip() == "[]", out.stderr
