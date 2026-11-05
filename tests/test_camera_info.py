"""CameraInfo -> Calibration conversion (task_0033)."""

from dataclasses import replace

import pytest

from av_perception_lab.camera import calibration_from_camera_info, has_distortion
from av_perception_lab.core.options import ConfigError
from av_perception_lab.io.mcap_reader import CameraInfoMsg

K = (1000.0, 0.0, 639.5, 0.0, 1000.0, 359.5, 0.0, 0.0, 1.0)
P = (900.0, 0.0, 600.0, 0.0, 0.0, 910.0, 350.0, 0.0, 0.0, 0.0, 1.0, 0.0)


def _msg(**kw) -> CameraInfoMsg:
    base = CameraInfoMsg(
        width=1280,
        height=720,
        k=K,
        d=(-0.1, 0.05, 0.001, 0.002, 0.0),
        model="plumb_bob",
        p=P,
        roi=(0, 0, 0, 0),
        binning=0,
    )
    return replace(base, **kw)


def test_raw_uses_k_and_d():
    msg = _msg()
    cal = calibration_from_camera_info(msg, (1280, 720))
    assert cal.image_size == (1280, 720)
    assert cal.intrinsics.fx == 1000.0
    assert cal.intrinsics.fy == 1000.0
    assert cal.intrinsics.cx == 640.0  # 639.5 + 0.5 (continuous convention)
    assert cal.intrinsics.cy == 360.0
    assert cal.dist == msg.d
    assert has_distortion(msg)


def test_rational_polynomial_accepted():
    d = (0.1, 0.2, 0.0, 0.0, 0.3, 0.4, 0.5, 0.6)
    cal = calibration_from_camera_info(
        _msg(model="rational_polynomial", d=d), (1280, 720)
    )
    assert cal.dist == d


@pytest.mark.parametrize("d", [(), (0.0, 0.0, 0.0, 0.0, 0.0)])
def test_rectified_uses_p(d):
    msg = _msg(d=d)
    cal = calibration_from_camera_info(msg, (1280, 720))
    assert cal.intrinsics.fx == 900.0
    assert cal.intrinsics.fy == 910.0
    assert cal.intrinsics.cx == 600.5
    assert cal.intrinsics.cy == 350.5
    assert cal.dist == ()
    assert not has_distortion(msg)


def test_size_mismatch_raises():
    with pytest.raises(ConfigError, match="1280x720"):
        calibration_from_camera_info(_msg(), (1920, 1080))


@pytest.mark.parametrize(
    ("kw", "field"),
    [
        ({"model": "equidistant"}, "distortion_model"),
        ({"roi": (10, 0, 100, 100)}, "roi"),
        ({"binning": 2}, "binning"),
    ],
)
def test_equidistant_roi_and_binning_raise_config_error(kw, field):
    with pytest.raises(ConfigError, match=field):
        calibration_from_camera_info(_msg(**kw), (1280, 720))


def test_full_frame_roi_means_no_roi():
    # ROS RegionOfInterest order: x_offset, y_offset, height, width
    cal = calibration_from_camera_info(_msg(roi=(0, 0, 720, 1280)), (1280, 720))
    assert cal.image_size == (1280, 720)
    for roi in [(0, 0, 1280, 720), (0, 0, 719, 1280), (1, 0, 720, 1280)]:
        with pytest.raises(ConfigError, match="roi"):
            calibration_from_camera_info(_msg(roi=roi), (1280, 720))


def test_full_frame_roi_read_from_an_mcap(tmp_path):
    from mcap_fixture import CameraInfoSpec, write_mcap

    from av_perception_lab.io.mcap_reader import read_camera_info

    path = write_mcap(
        tmp_path / "a.mcap",
        camera_infos=[
            CameraInfoSpec(
                topic="/cam/camera_info",
                width=16,
                height=12,
                k=[100, 0, 8, 0, 100, 6, 0, 0, 1],
                roi=(0, 0, 12, 16),
            )
        ],
    )
    msg = read_camera_info(path, "/cam/camera_info")
    assert msg.roi == (0, 0, 12, 16)
    assert calibration_from_camera_info(msg, (16, 12)).image_size == (16, 12)
