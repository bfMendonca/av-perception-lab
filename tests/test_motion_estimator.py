"""Per-track motion estimator of task_0009 (motion/estimator.py): model by
class at birth, prediction without a distance, reset by age, restart by the
NIS gate, alignment with the instances, describe and build_motion."""

import math

import numpy as np
import pytest

from det_depth_distance.core.options import ConfigError
from det_depth_distance.core.types import CameraIntrinsics, DepthResult, MapGeometry
from det_depth_distance.distance import BoxDepthStats
from det_depth_distance.motion import MotionConfig, MotionEstimator, build_motion
from det_depth_distance.motion.config import MeasurementConfig

FRAME_W, FRAME_H = 1000, 600
GEOMETRY = MapGeometry(FRAME_W, FRAME_H, FRAME_W, FRAME_H, 1.0, 1.0)
K = CameraIntrinsics(500.0, 500.0, 500.0, 300.0)
DEPTH = DepthResult(np.zeros((2, 2), np.float32), K, None, GEOMETRY)
# Nearly exact measurements, so the tests see the filters, not the noise.
PRECISE = MeasurementConfig(std_scale=0.0, rel_floor=1e-4, pixel_std=0.01)
CAMERA_HEIGHT_M = 1.2  # reference points on the ground, below the camera


def box_stats(distance: float | None) -> BoxDepthStats:
    nan = math.nan
    return BoxDepthStats(
        (0, 0, 1, 1), 1, 1, 1, 1.0, nan, nan, 0.0, nan, nan, nan, nan, nan, nan, nan,
        distance,
    )  # fmt: skip


def observe(make_instances, points, *, ids, labels=None, distances=None):
    """Instances whose bottom-centre reference points see the camera-plane
    points (x, z) on the ground, with their z as distance (None = n/a)."""
    boxes = []
    for x, z in points:
        u = K.fx * x / z + K.cx
        v = K.fy * CAMERA_HEIGHT_M / z + K.cy
        boxes.append([u - 10.0, v - 30.0, u + 10.0, v])
    inst = make_instances(
        boxes,
        labels=labels or ["car"] * len(points),
        track_id=np.array(ids, dtype=np.int64),
    )
    zs = distances if distances is not None else [z for _, z in points]
    return inst, tuple(box_stats(z) for z in zs)


def estimator(**cfg) -> MotionEstimator:
    return build_motion(
        MotionConfig(**{"measurement": PRECISE, **cfg}), distance_type="z"
    )


def test_lateral_motion_gives_the_velocity_and_heading(make_instances):
    est = estimator()
    for k in range(30):  # 2 m/s to the right at 10 m, 10 fps
        inst, stats = observe(make_instances, [(-3.0 + 0.2 * k, 10.0)], ids=[1])
        (m,) = est.update(k / 10, inst, stats, DEPTH)
    assert m.model == "cv" and m.n_updates == 30
    assert (m.vx, m.vz) == (pytest.approx(2.0, abs=0.05), pytest.approx(0.0, abs=0.05))
    assert m.speed == pytest.approx(2.0, abs=0.05)
    assert m.heading_deg == pytest.approx(90.0, abs=2.0)  # moving right
    assert math.isnan(m.yaw_rate_dps) and math.isnan(m.abs_speed)
    assert (m.x, m.z) == (pytest.approx(2.8, abs=0.01), pytest.approx(10.0, abs=0.01))
    traj = m.trajectory
    assert traj.t.shape == (12,) and traj.y == pytest.approx(CAMERA_HEIGHT_M)
    np.testing.assert_allclose(traj.xz[-1], [2.8 + 2.0 * 3.0, 10.0], atol=0.2)


def test_model_is_chosen_by_class_at_birth(make_instances):
    est = estimator(
        models={"cv": {"type": "cv"}, "walk": {"type": "cv", "accel_std": 1.0}},
        classes={"person": "walk"},
        default=None,
    )
    inst, stats = observe(
        make_instances,
        [(0.0, 10.0), (2.0, 12.0), (-2.0, 8.0)],
        ids=[1, 2, 3],
        labels=["person", "car", "person"],
    )
    a, b, c = est.update(0.0, inst, stats, DEPTH)
    assert (a.model, b, c.model) == ("walk", None, "walk")  # default None: car skipped
    # A label flip later does not change the model of an existing track.
    inst, stats = observe(make_instances, [(0.0, 10.0)], ids=[1], labels=["car"])
    (a,) = est.update(0.1, inst, stats, DEPTH)
    assert a.model == "walk" and a.n_updates == 2


def test_without_distance_the_track_is_only_predicted(make_instances):
    est = estimator()
    for k in range(10):
        inst, stats = observe(make_instances, [(0.0, 10.0 + k * 0.5)], ids=[1])
        est.update(k / 10, inst, stats, DEPTH)
    inst, stats = observe(
        make_instances, [(0.0, 15.0), (3.0, 9.0)], ids=[1, 2], distances=[None, None]
    )
    old, new = est.update(1.0, inst, stats, DEPTH)
    assert new is None  # no distance yet: no filter
    assert old.n_updates == 10 and math.isnan(old.nis) and not old.restarted
    assert old.z == pytest.approx(10.0 + 9 * 0.5 + 0.5, abs=0.05)  # predicted 0.1 s on


def test_track_ids_none_gives_all_none(make_instances):
    est = estimator()
    inst, stats = observe(make_instances, [(0.0, 10.0), (1.0, 12.0)], ids=[1, 2])
    plain = make_instances(inst.xyxy.tolist(), labels=["car", "car"])
    assert est.update(0.0, plain, stats, DEPTH) == (None, None)


def test_state_is_dropped_after_reset_after_s(make_instances):
    est = estimator(reset_after_s=0.5)
    inst, stats = observe(make_instances, [(0.0, 10.0)], ids=[1])
    est.update(0.0, inst, stats, DEPTH)
    (m,) = est.update(0.5, inst, stats, DEPTH)  # exactly the limit: kept
    assert m.n_updates == 2
    (m,) = est.update(1.0 + 1e-3, inst, stats, DEPTH)  # beyond: a fresh filter
    assert m.n_updates == 1
    est.reset()
    (m,) = est.update(1.1, inst, stats, DEPTH)
    assert m.n_updates == 1


def test_a_jump_restarts_the_filter(make_instances):
    est = estimator()
    for k in range(5):
        inst, stats = observe(make_instances, [(0.0, 10.0)], ids=[7])
        (m,) = est.update(k / 10, inst, stats, DEPTH)
    assert m.nis < 30.0 and not m.restarted
    inst, stats = observe(make_instances, [(4.0, 25.0)], ids=[7])  # id switch
    (m,) = est.update(0.5, inst, stats, DEPTH)
    assert m.restarted and m.nis > 30.0 and m.n_updates == 1
    assert (m.x, m.z) == (pytest.approx(4.0, abs=1e-3), pytest.approx(25.0, abs=1e-3))


def test_estimates_are_aligned_with_the_instances(make_instances):
    est = estimator()
    inst, stats = observe(
        make_instances, [(-2.0, 8.0), (3.0, 20.0)], ids=[5, 9], distances=[8.0, None]
    )
    a, b = est.update(0.0, inst, stats, DEPTH)
    assert a.x == pytest.approx(-2.0, abs=1e-3) and b is None
    inst, stats = observe(make_instances, [(3.0, 20.0), (-2.0, 8.0)], ids=[9, 5])
    b, a = est.update(0.1, inst, stats, DEPTH)
    assert b.n_updates == 1 and a.n_updates == 2


def test_describe():
    desc = estimator(classes={"person": "cv"}).describe()
    assert desc == {
        "models": {
            "cv": {
                "type": "cv",
                "options": {
                    "init_speed_std": 5.0,
                    "min_speed_for_heading": 1.0,
                    "accel_std": 2.0,
                },
            }
        },
        "classes": {"person": "cv"},
        "default": "cv",
        "reset_after_s": 1.0,
        "nis_gate": 30.0,
        "measurement": {
            "reference_point": "bottom_center",
            "std_scale": 0.0,
            "rel_floor": 1e-4,
            "pixel_std": 0.01,
        },
        "prediction": {"horizon_s": 3.0, "step_s": 0.25, "n_sigma": 2.0},
        "ego_motion": None,
        "camera_mount": {"x_m": 0.0, "y_m": 0.0, "yaw_deg": 0.0},
    }


def test_build_motion_errors_have_the_dotted_path(tmp_path):
    with pytest.raises(
        ConfigError, match=r"^motion\.models\.car\.type: unknown motion"
    ):
        build_motion(
            MotionConfig(models={"car": {"type": "kalman"}}, default="car"),
            distance_type="z",
        )
    with pytest.raises(
        ConfigError, match=r"^motion\.models\.cv\.accel_std: expected float"
    ):
        build_motion(
            MotionConfig(models={"cv": {"type": "cv", "accel_std": "x"}}),
            distance_type="z",
        )
    missing = tmp_path / "ego.csv"
    with pytest.raises(
        ConfigError, match=r"^motion\.ego_motion\.path: .*file not found"
    ):
        build_motion(
            MotionConfig(ego_motion={"type": "csv", "path": str(missing)}),
            distance_type="z",
        )


def test_ctrv_without_ego_motion_warns(tmp_path, capsys):
    models = {"cv": {"type": "cv"}, "car": {"type": "ctrv"}}
    build_motion(MotionConfig(models=models), distance_type="z")
    err = capsys.readouterr().err
    assert err.count("[warn]") == 1
    assert "[warn] motion.models.car: ctrv without ego_motion" in err
    ego = tmp_path / "ego.csv"
    ego.write_text("t,x,y,yaw\n0,0,0,0\n10,100,0,0\n")
    build_motion(
        MotionConfig(models=models, ego_motion={"type": "csv", "path": str(ego)}),
        distance_type="z",
    )
    assert capsys.readouterr().err == ""
