import math
import subprocess
import sys

import numpy as np
import pytest

from det_depth_distance.core.options import ConfigError, from_mapping
from det_depth_distance.distance import BoxDepthStats
from det_depth_distance.temporal import (
    FILTERS,
    EmaFilter,
    KalmanFilter1D,
    MedianFilter,
    SmoothingConfig,
    TrackSmoother,
    measurement_sigma,
)

CFG = SmoothingConfig()


def run(filt, t, z, sigma=0.2):
    """Feed the series (z None = no measurement) and return the estimates."""
    return [filt.update(ti, zi, sigma) for ti, zi in zip(t, z, strict=True)]


def stats(distance: float | None, std: float = 0.0) -> BoxDepthStats:
    nan = math.nan
    return BoxDepthStats(
        (0, 0, 1, 1), 1, 1, 1, 1.0, nan, nan, std, nan, nan, nan, nan, nan, nan, nan,
        distance,
    )  # fmt: skip


def test_kalman_and_ema_reduce_white_noise():
    # task_0006 Fact 8: Kalman q 2 leaves 0.33 sigma of white noise.
    rng = np.random.default_rng(0)
    t = np.arange(300) / 30
    z = 20.0 + rng.normal(0.0, 0.2, size=300)
    kalman = run(KalmanFilter1D(CFG), t, z)
    ema = run(EmaFilter(CFG), t, z)
    settled = t >= 1.0
    k_out = np.array([e.distance for e in kalman])[settled]
    e_out = np.array([e.distance for e in ema])[settled]
    assert k_out.std() <= 0.4 * 0.2
    assert e_out.std() <= 0.7 * 0.2
    assert abs(np.mean([e.rel_speed for e in kalman][30:])) < 0.1
    assert all(math.isnan(e.rel_speed) and math.isnan(e.distance_std) for e in ema)


def _speed_step(fps: float, duration: float = 5.0):
    """20 m for 2 s, then approaching at 3 m/s (no noise)."""
    t = np.arange(round(duration * fps)) / fps
    return t, np.where(t < 2.0, 20.0, 20.0 - 3.0 * (t - 2.0))


def test_kalman_lag_after_a_speed_change_is_bounded():
    # Measured 0.401 m / 0.83 s (Fact 8); the bounds keep a margin.
    t, z = _speed_step(30.0)
    out = run(KalmanFilter1D(CFG), t, z)
    error = np.abs(np.array([e.distance for e in out]) - z)
    speed = np.array([e.rel_speed for e in out])
    assert error.max() <= 0.45
    converged = (t >= 2.0) & (np.abs(speed + 3.0) < 0.3)
    assert t[np.argmax(converged)] - 2.0 <= 1.0
    assert np.all(np.abs(speed[t >= 3.0] + 3.0) < 0.3)  # and it stays there
    assert error[-1] < 0.01  # steady ramp: no lag
    assert out[-1].rel_speed == pytest.approx(-3.0, abs=0.01)
    assert out[-1].distance_std > 0 and out[-1].rel_speed_std > 0


def test_kalman_speed_is_stride_invariant():
    finals = [run(KalmanFilter1D(CFG), *_speed_step(fps))[-1] for fps in (30.0, 10.0)]
    assert finals[0].rel_speed == pytest.approx(finals[1].rel_speed, abs=0.1)


@pytest.mark.parametrize("fps", [30.0, 10.0])
def test_ema_ramp_lag_is_exact_and_in_seconds(fps):
    t = np.arange(round(3 * fps)) / fps
    z = 20.0 - 3.0 * t
    out = run(EmaFilter(CFG), t, z)
    dt = 1 / fps
    alpha = 1 - math.exp(-dt / CFG.tau_s)
    lag_m = out[-1].distance - z[-1]
    assert lag_m == pytest.approx(3.0 * dt * (1 - alpha) / alpha, rel=1e-6)
    # In seconds the lag stays near tau_s - dt / 2 (0.085 s at 30 Hz, 0.058 s
    # at 10 Hz); a fixed alpha of 0.28 would lag 0.25 s at 10 Hz.
    assert lag_m / 3.0 == pytest.approx(CFG.tau_s - dt / 2, abs=0.01)


def test_median_drops_an_isolated_spike():
    t = np.arange(30) / 30
    z = np.full(30, 20.0)
    z[15] = 25.0
    out = run(MedianFilter(CFG), t, z)
    assert [e.distance for e in out] == [20.0] * 30
    # Default window (0.15 s): the last 5 samples at 30 fps.
    median = MedianFilter(CFG)
    run(median, t[:10], z[:10])
    assert len(median.samples) == 5


def test_kalman_gap_predicts_and_resumes_without_a_jump():
    # Approaching at 3 m/s; no measurement for 0.3 s.
    t = np.arange(90) / 30
    z = 20.0 - 3.0 * t
    gap = (t >= 1.5) & (t < 1.8)
    out = run(KalmanFilter1D(CFG), t, [None if g else v for g, v in zip(gap, z)])
    d = np.array([e.distance for e in out])
    std = np.array([e.distance_std for e in out])
    np.testing.assert_allclose(d[gap], z[gap], atol=0.05)  # extrapolated
    assert np.all(np.diff(std[gap]) > 0)  # uncertainty grows
    assert std[np.argmax(gap) - 1] < std[gap][-1]
    assert np.abs(np.diff(d)).max() < 0.2  # 0.1 m per frame, no jump at resume
    n = [e.n_updates for e in out]
    assert n[-1] == 90 - gap.sum()  # missing frames add no update


@pytest.mark.parametrize("name", ["ema", "median"])
def test_ema_and_median_hold_the_value_without_measurement(name):
    filt = FILTERS[name](CFG)
    first = filt.update(0.0, 10.0, 0.1)
    held = filt.update(0.1, None, math.nan)
    assert held.distance == first.distance and held.n_updates == 1


def test_measurement_sigma():
    assert measurement_sigma(20.0, 1.0, CFG) == pytest.approx(0.5)  # 0.5 * std
    assert measurement_sigma(20.0, 0.1, CFG) == pytest.approx(0.2)  # 1 % floor
    assert measurement_sigma(20.0, math.nan, CFG) == pytest.approx(0.2)
    relative = SmoothingConfig(std_scale=0.0)
    assert measurement_sigma(20.0, 9.0, relative) == pytest.approx(0.2)


def test_smoother_keeps_independent_states_per_track():
    smoother = TrackSmoother(SmoothingConfig(filter="ema"))
    ids = np.array([3, 7])
    smoother.update(0.0, ids, [stats(10.0), stats(30.0)])
    out = smoother.update(0.1, ids[::-1], [stats(31.0), stats(11.0)])
    assert out[0].distance == pytest.approx(30.0 + (1 - math.exp(-1)) * 1.0)
    assert out[1].distance == pytest.approx(10.0 + (1 - math.exp(-1)) * 1.0)
    assert smoother.update(0.2, None, [stats(1.0)]) == (None,)


def test_smoother_na_distance_and_track_without_distance():
    smoother = TrackSmoother(CFG)
    ids = np.array([1, 2])
    smoother.update(0.0, ids, [stats(20.0), stats(None)])
    first, second = smoother.update(1 / 30, ids, [stats(None), stats(None)])
    assert first.n_updates == 1 and first.distance == pytest.approx(20.0)  # predicted
    assert second is None  # never measured: no estimate


def test_smoother_drops_stale_state_and_restarts_returning_ids():
    smoother = TrackSmoother(CFG)
    ids = np.array([1])
    for k in range(10):
        (est,) = smoother.update(k / 30, ids, [stats(20.0)])
    assert est.n_updates == 10
    # 0.9 s later (< reset_after_s): same filter.
    (est,) = smoother.update(9 / 30 + 0.9, ids, [stats(20.0)])
    assert est.n_updates == 11
    # 1.2 s without a measurement (track absent or n/a): fresh filter.
    (est,) = smoother.update(9 / 30 + 2.1, ids, [stats(25.0)])
    assert est.n_updates == 1 and est.distance == 25.0
    smoother.reset()
    (est,) = smoother.update(5.0, ids, [stats(30.0)])
    assert est.n_updates == 1


def test_smoother_reset_limit_is_inclusive():
    # Measured at 32/30 s and again exactly reset_after_s (1.0 s, 30 frames)
    # later; 62/30 - 32/30 is 1.0000000000000002 in floating point.
    smoother = TrackSmoother(CFG)
    ids = np.array([1])
    smoother.update(32 / 30, ids, [stats(20.0)])
    assert (62 / 30 - 32 / 30) > 1.0
    (est,) = smoother.update(62 / 30, ids, [stats(20.0)])
    assert est.n_updates == 2


def test_numeric_core_imports_no_framework():
    # task_0006 Verification: tracking and smoothing stay numpy-only, like
    # distance.py (task_0004): no GPU stack, no lap/scipy, not even OpenCV/PIL.
    code = (
        "import sys, det_depth_distance.temporal, det_depth_distance.core.tracking; "
        "print(sorted({'torch', 'ultralytics', 'lap', 'scipy', 'cv2', 'PIL'} "
        "& set(sys.modules)))"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], check=False, capture_output=True, text=True
    )
    assert out.stdout.strip() == "[]", out.stderr


def test_smoother_is_deterministic():
    rng = np.random.default_rng(3)
    z = 20.0 + rng.normal(0, 0.3, size=60)
    runs = []
    for _ in range(2):
        smoother = TrackSmoother(CFG)
        runs.append(
            [
                smoother.update(k / 30, np.array([1]), [stats(v, 0.4)])
                for k, v in enumerate(z)
            ]
        )
    assert runs[0] == runs[1]


@pytest.mark.parametrize(
    ("data", "match"),
    [
        ({"filter": "kalmann"}, r"^smoothing\.filter: 'kalmann' is not one of"),
        ({"accel_std": 0}, r"^smoothing: accel_std must be > 0"),
        ({"window_s": -1}, r"^smoothing: window_s must be > 0"),
        ({"std_scale": -0.1}, r"^smoothing: std_scale and rel_floor must be >= 0"),
        ({"std_scale": 0, "rel_floor": 0}, r"^smoothing: .*cannot both be 0"),
        ({"tau": 0.1}, r"^smoothing\.tau: unknown option"),
    ],
)
def test_config_is_validated(data, match):
    with pytest.raises(ConfigError, match=match):
        from_mapping(SmoothingConfig, data, "smoothing")


def test_config_filter_must_be_registered():
    with pytest.raises(ValueError, match="filter must be one of"):
        SmoothingConfig(filter="kalmann")
