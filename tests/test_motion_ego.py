"""Optional ego-motion of task_0009 (motion/ego.py): the CSV provider
(interpolation, unwrapped yaw, clamping outside the data, malformed files),
planar transforms, and the estimator filtering in the world: a static object
seen from a turning vehicle has no speed over ground and a relative velocity
equal to the derivative of its camera position."""

import math

import numpy as np
import pytest
from test_motion_estimator import DEPTH, PRECISE, observe

from av_perception_lab.core.options import ConfigError
from av_perception_lab.core.registry import build
from av_perception_lab.motion import MotionConfig, MotionEstimator
from av_perception_lab.motion.config import CameraMount
from av_perception_lab.motion.ego import (
    CsvEgoMotion,
    EgoMotionProvider,
    EgoPose,
    PlanarTransform,
    camera_to_world,
    camera_velocity,
)
from av_perception_lab.motion.models import ConstantVelocity


def csv_provider(tmp_path, text, **options) -> CsvEgoMotion:
    path = tmp_path / "ego.csv"
    path.write_text(text)
    ego = build("ego_motion", {"type": "csv", "path": str(path), **options}, path="e")
    ego.open()
    return ego


def test_csv_interpolates_and_unwraps_the_yaw(tmp_path):
    # yaw crosses +-pi between the last two samples: 3.0 -> -3.0 is +0.283 rad.
    ego = csv_provider(
        tmp_path, "t,x,y,yaw\n0,0,0,0\n1,10,0,3.0\n2,20,4,-3.0\n", time_offset_s=0.5
    )
    p = ego.pose_at(0.0)  # CSV time 0.5
    assert (p.x, p.y, p.yaw) == pytest.approx((5.0, 0.0, 1.5))
    assert (p.vx, p.vy, p.yaw_rate) == pytest.approx((10.0, 0.0, 3.0))
    p = ego.pose_at(1.0)  # CSV time 1.5
    step = 2 * math.pi - 6.0
    assert (p.x, p.y) == pytest.approx((15.0, 2.0))
    assert p.yaw == pytest.approx(3.0 + step / 2)
    assert p.yaw_rate == pytest.approx(step)
    assert ego.clamped_frames == 0


def test_csv_holds_the_edge_pose_outside_the_data(tmp_path, capsys):
    ego = csv_provider(tmp_path, "t,x,y,yaw\n1,0,0,0\n2,10,0,0.5\n")
    before = ego.pose_at(0.5)
    assert before == EgoPose(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    after = ego.pose_at(3.0)
    assert after == EgoPose(10.0, 0.0, 0.5, 0.0, 0.0, 0.0)
    ego.pose_at(1.5)
    err = capsys.readouterr().err
    assert err.count("[warn] motion.ego_motion: t=0.500 s outside [1, 2] s") == 1
    assert "[warn]" not in err.replace("[warn] motion.ego_motion: t=0.500", "", 1)
    assert ego.clamped_frames == 2
    assert ego.describe()["clamped_frames"] == 2
    assert ego.describe()["options"]["path"].endswith("ego.csv")


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("t,x,yaw\n0,0,0\n1,1,0\n", "missing columns y"),
        ("t,x,y,yaw\n0,0,0,0\n", "at least 2 samples"),
        ("t,x,y,yaw\n0,0,0,0\n0,1,0,0\n", "strictly increasing"),
        ("t,x,y,yaw\n0,0,0,0\n1,a,0,0\n", "non-numeric"),
        ("t,x,y,yaw\n0,0,0,0\n1,nan,0,0\n", "non-finite"),
    ],
)
def test_csv_rejects_malformed_files(tmp_path, text, message):
    with pytest.raises(ConfigError, match=rf"^path: .*{message}"):
        csv_provider(tmp_path, text)


def test_planar_transform_round_trip():
    a = PlanarTransform(np.array([[0.0, -1.0], [1.0, 0.0]]), np.array([3.0, -2.0]))
    p = np.array([[1.0, 2.0], [-4.0, 0.5]])
    np.testing.assert_allclose(a.inverse().apply(a.apply(p)), p)
    cov = np.array([[4.0, 0.0], [0.0, 1.0]])
    np.testing.assert_allclose(a.rotate_cov(cov), [[1.0, 0.0], [0.0, 4.0]])
    ident = PlanarTransform.identity()
    np.testing.assert_allclose(ident.apply(p), p)


def test_camera_to_world_axes():
    pose = EgoPose(5.0, 1.0, math.pi / 2, 0.0, 3.0, 0.0)  # heading north (+y)
    mount = CameraMount(x_m=2.0, y_m=0.0, yaw_deg=0.0)
    t = camera_to_world(pose, mount)
    # Camera origin 2 m ahead of the vehicle; z (forward) -> +y, x (right) -> +x.
    np.testing.assert_allclose(t.apply(np.zeros(2)), [5.0, 3.0], atol=1e-12)
    np.testing.assert_allclose(t.apply([0.0, 1.0]), [5.0, 4.0], atol=1e-12)
    np.testing.assert_allclose(t.apply([1.0, 0.0]), [6.0, 3.0], atol=1e-12)
    # Turning, the lever arm adds velocity across it: yaw rate x arm.
    spin = EgoPose(0.0, 0.0, 0.0, 10.0, 0.0, 0.5)
    np.testing.assert_allclose(camera_velocity(spin, mount), [10.0, 1.0])


class TurningEgo(EgoMotionProvider):
    """Vehicle on a circle: speed v, CCW turn rate omega, starting at the
    origin heading +x (an unregistered double: no options)."""

    type_name = "turning"

    def __init__(self, v: float = 5.0, omega: float = 0.1) -> None:
        super().__init__(None)
        self.v, self.omega = v, omega

    def pose_at(self, t: float) -> EgoPose:
        v, w = self.v, self.omega
        yaw = w * t
        return EgoPose(
            v / w * math.sin(yaw),
            v / w * (1 - math.cos(yaw)),
            yaw,
            v * math.cos(yaw),
            v * math.sin(yaw),
            w,
        )

    def describe(self) -> dict:
        return {"type": self.type_name}


def test_static_object_seen_from_a_turning_vehicle(make_instances):
    ego = TurningEgo()
    mount = CameraMount(x_m=1.5, y_m=0.2, yaw_deg=5.0)
    cfg = MotionConfig(measurement=PRECISE, camera_mount=mount)
    model = ConstantVelocity(ConstantVelocity.Options(accel_std=0.5))
    est = MotionEstimator(cfg, {"cv": model}, distance_type="z", ego=ego)
    obj = np.array([40.0, 4.0])  # world, static

    def camera_point(t: float) -> np.ndarray:
        return camera_to_world(ego.pose_at(t), mount).inverse().apply(obj)

    for k in range(31):  # 3 s at 10 fps
        t = k / 10
        inst, stats = observe(make_instances, [camera_point(t)], ids=[1])
        (m,) = est.update(t, inst, stats, DEPTH)
    assert m.abs_speed < 0.2
    h = 1e-4
    derivative = (camera_point(t + h) - camera_point(t - h)) / (2 * h)
    np.testing.assert_allclose([m.vx, m.vz], derivative, atol=0.15)
    # The heading is the one of the motion over ground: none for a static
    # object, although it moves fast relative to the camera.
    assert m.speed > 4.0 and math.isnan(m.heading_deg)
    np.testing.assert_allclose([m.x, m.z], camera_point(t), atol=0.01)
    # Prediction of a static object: it stays where it is (in this camera frame).
    np.testing.assert_allclose(m.trajectory.xz[-1], camera_point(t), atol=0.3)
    assert est.describe()["ego_motion"] == {"type": "turning"}
