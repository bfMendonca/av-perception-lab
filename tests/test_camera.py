import json
from dataclasses import replace

import cv2
import numpy as np
import pytest

from av_perception_lab import camera
from av_perception_lab.calibration import TargetSpec, make_target
from av_perception_lab.camera import (
    CalibratedIntrinsics,
    Calibration,
    EstimatedIntrinsics,
    FixedIntrinsics,
    Undistorter,
    build_camera,
    build_intrinsics_provider,
    check_camera_metadata,
    load_calibration,
    save_calibration,
)
from av_perception_lab.config import CalibrationConfig, CameraConfig, EstimateConfig
from av_perception_lab.core.options import ConfigError
from av_perception_lab.core.types import CameraIntrinsics
from av_perception_lab.io.source import FrameSource


class StillSource(FrameSource):
    frame_size = (1920, 1080)
    name = "still"

    def __iter__(self):
        return iter(())

    def sample(self, n):
        return [np.zeros((1080, 1920, 3), np.uint8)] * n

    def describe(self):
        return {}


def test_fixed():
    result = FixedIntrinsics(1400.0).resolve(StillSource(), device="cpu")
    assert result.intrinsics == CameraIntrinsics(1400.0, 1400.0, 960.0, 540.0)
    assert result.source == "config (camera.focal_px)"
    with pytest.raises(ConfigError):
        FixedIntrinsics(0.0)


def test_estimated(monkeypatch):
    calls = {}

    def fake_estimate(frames, model, **kwargs):
        calls.update(n=len(frames), model=model, **kwargs)
        return 1390.5

    monkeypatch.setattr(camera, "estimate_focal_frame_px", fake_estimate)
    cfg = EstimateConfig(frames=4, process_res=280)
    result = EstimatedIntrinsics(cfg).resolve(StillSource(), device="cuda:1")
    assert result.intrinsics == CameraIntrinsics.centered(1390.5, 1920, 1080)
    assert result.source == "estimated by da3-large-1.1 (median of 4 frames)"
    assert calls == {
        "n": 4,
        "model": "da3-large-1.1",
        "device": "cuda:1",
        "process_res": 280,
        "process_res_method": "upper_bound_resize",
        "cache_dir": None,
    }


def test_precedence():
    assert isinstance(
        build_intrinsics_provider(CameraConfig(focal_px=1000.0)), FixedIntrinsics
    )
    assert isinstance(build_intrinsics_provider(CameraConfig()), EstimatedIntrinsics)
    assert build_intrinsics_provider(CameraConfig(estimate=None)) is None


# --- task_0007: calibration files, undistortion, calibrated provider -------------

LENS = "iPhone 14 Pro Max back triple camera 6.86mm f/1.78"
IPHONE = {"model": "iPhone 14 Pro Max", "lens_model": LENS, "focal_length_35mm": 24}
K_CV = np.array([[1452.3, 0.0, 958.1], [0.0, 1453.0, 541.2], [0.0, 0.0, 1.0]])
DIST = np.array([[0.11, -0.25, 0.0004, -0.0002]])


def calibration(**fields) -> Calibration:
    return Calibration.from_opencv(K_CV, DIST, (1920, 1080), **fields)


def test_opencv_convention_round_trip():
    cal = calibration()
    # Continuous convention: OpenCV's cx/cy + 0.5.
    assert cal.intrinsics == CameraIntrinsics(1452.3, 1453.0, 958.6, 541.7)
    assert np.array_equal(cal.camera_matrix(), K_CV)
    assert cal.dist == pytest.approx(tuple(DIST.ravel()))
    assert cal.dist_array().shape == (1, 4) and cal.has_distortion
    plain = Calibration.from_opencv(K_CV, np.zeros(0), (1920, 1080))
    assert plain.dist == () and not plain.has_distortion
    assert np.array_equal(plain.dist_array(), np.zeros((1, 5)))


def test_ideal_opencv_centre_is_the_project_centre():
    k = np.array([[1400.0, 0, (1920 - 1) / 2], [0, 1400.0, (1080 - 1) / 2], [0, 0, 1]])
    cal = Calibration.from_opencv(k, np.zeros(5), (1920, 1080))
    assert cal.intrinsics == CameraIntrinsics.centered(1400.0, 1920, 1080)


def test_scaled_to():
    uhd = Calibration.from_opencv(K_CV * [[2], [2], [1]], DIST, (3840, 2160))
    hd = uhd.scaled_to(1920, 1080)
    k = uhd.intrinsics
    assert hd.intrinsics == CameraIntrinsics(k.fx / 2, k.fy / 2, k.cx / 2, k.cy / 2)
    assert hd.image_size == (1920, 1080) and hd.dist == uhd.dist
    assert uhd.scaled_to(3840, 2160) is uhd
    with pytest.raises(ConfigError, match="different aspect ratio"):
        uhd.scaled_to(1440, 1080)
    with pytest.raises(ConfigError, match="portrait vs landscape"):
        uhd.scaled_to(1080, 1920)


@pytest.mark.parametrize("suffix", [".yaml", ".json"])
def test_file_round_trip(tmp_path, suffix):
    cal = calibration(
        rms_px=0.21,
        std={"fx": 0.9, "fy": 0.9, "cx": 1.4, "cy": 1.2},
        camera=IPHONE,
        meta={
            "target": {"type": "chessboard", "cols": 9, "rows": 6, "square_mm": 25.0},
            "n_views": 48,
            "coverage": 0.83,
            "recording": {"notes": "stabilisation standard"},
        },
    )
    path = tmp_path / f"cal{suffix}"
    save_calibration(cal, path)
    loaded = load_calibration(path)
    assert loaded.image_size == (1920, 1080)
    assert loaded.intrinsics == pytest.approx(cal.intrinsics)
    assert loaded.dist == pytest.approx(cal.dist)
    assert (loaded.rms_px, dict(loaded.std)) == (0.21, dict(cal.std))
    assert loaded.camera == IPHONE  # 35 mm focal stays an integer
    assert loaded.meta == cal.meta
    if suffix == ".json":
        assert (
            json.loads(path.read_text())["camera_matrix"]["type_id"] == "opencv-matrix"
        )


def write_minimal(path, k=K_CV, dist=DIST, width=1920, height=1080):
    """Calibration file with only the required keys, written by OpenCV itself."""
    fs = cv2.FileStorage(str(path), cv2.FILE_STORAGE_WRITE)
    if width is not None:
        fs.write("image_width", width)
    fs.write("image_height", height)
    if k is not None:
        fs.write("camera_matrix", np.asarray(k, dtype=np.float64))
    fs.write("distortion_coefficients", np.asarray(dist, dtype=np.float64))
    fs.release()
    return path


def test_minimal_file(tmp_path):
    cal = load_calibration(write_minimal(tmp_path / "min.yaml"))
    assert cal.camera_matrix() == pytest.approx(K_CV)
    assert (cal.rms_px, cal.std, cal.camera, cal.meta) == (None, {}, {}, {})


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"k": None}, "missing required key 'camera_matrix'"),
        ({"width": None}, "image_width must be a positive integer"),
        ({"width": 0}, "image_width must be a positive integer"),
        ({"k": np.eye(2)}, "camera_matrix must be 3x3"),
        ({"k": [[1000, 0, 960], [0, 1000, 540], [0, 1, 1]]}, "last row"),
        ({"k": [[1000, 2, 960], [0, 1000, 540], [0, 0, 1]]}, "skew"),
        ({"k": [[-1000, 0, 960], [0, 1000, 540], [0, 0, 1]]}, "fx and fy must be > 0"),
        ({"dist": np.zeros((1, 3))}, "distortion_coefficients must be 1xN"),
        ({"dist": np.zeros((2, 4))}, "distortion_coefficients must be 1xN"),
    ],
)
def test_invalid_files(tmp_path, kwargs, match):
    path = write_minimal(tmp_path / "bad.yaml", **kwargs)
    with pytest.raises(ConfigError, match=match) as exc:
        load_calibration(path)
    assert str(exc.value).startswith(str(path))


def test_unreadable_files(tmp_path):
    with pytest.raises(ConfigError, match="calibration file not found"):
        load_calibration(tmp_path / "missing.yaml")
    plain = tmp_path / "plain.yaml"
    plain.write_text("image_width: 1920\n")  # YAML without the %YAML:1.0 header
    with pytest.raises(ConfigError, match="not an OpenCV FileStorage file"):
        load_calibration(plain)
    future = write_minimal(tmp_path / "v2.yaml")
    future.write_text(future.read_text() + "format_version: 2\n")
    with pytest.raises(ConfigError, match="unsupported format_version 2"):
        load_calibration(future)


@pytest.mark.parametrize(
    ("extra", "match"),
    [
        ("rms_px: abc\n", "rms_px must be a number"),
        ("std_intrinsics: [ 1, 2 ]\n", "std_intrinsics must map names to numbers"),
        ("std_intrinsics: { fx: abc }\n", "std_intrinsics must map names to numbers"),
        ("camera: Apple\n", "camera must be a map"),
    ],
)
def test_malformed_optional_keys(tmp_path, extra, match):
    path = write_minimal(tmp_path / "opt.yaml")
    path.write_text(path.read_text() + extra)
    with pytest.raises(ConfigError, match=match) as exc:
        load_calibration(path)
    assert str(exc.value).startswith(str(path))


def test_calibration_description_round_trip():
    cal = Calibration.from_opencv(K_CV, DIST, (1920, 1080), rms_px=0.2)
    again = Calibration.from_description(json.loads(json.dumps(cal.describe())))
    assert (again.image_size, again.intrinsics, again.dist, again.rms_px) == (
        cal.image_size,
        cal.intrinsics,
        cal.dist,
        cal.rms_px,
    )
    with pytest.raises(ConfigError, match="invalid calibration description"):
        Calibration.from_description({"image_size": [1920, 1080]})


@pytest.mark.parametrize("alpha", [0.0, 1.0])
def test_undistorter_without_distortion_is_the_identity(alpha):
    cal = Calibration.from_opencv(K_CV, np.zeros(5), (1920, 1080))
    undistorter = Undistorter(cal, alpha=alpha)
    assert undistorter.output.intrinsics == pytest.approx(cal.intrinsics)
    assert undistorter.output.dist == ()
    frame = np.random.default_rng(0).integers(0, 256, (1080, 1920, 3), dtype=np.uint8)
    assert np.array_equal(undistorter(frame), frame)
    pts = np.array([[0.0, 0.0], [960.0, 540.0], [1919.5, 1079.5]])
    assert undistorter.undistort_points(pts) == pytest.approx(pts, abs=1e-6)


def test_undistorter_on_a_rendered_board(synthetic_camera, render_board):
    cam = synthetic_camera
    cal = Calibration.from_opencv(cam.k, cam.dist, cam.size, camera=IPHONE)
    undistorter = Undistorter(cal)
    out = undistorter.output
    assert out.dist == () and out.camera == IPHONE
    assert out.intrinsics.fx > cal.intrinsics.fx  # alpha 0 crops: longer focal
    target = make_target(TargetSpec())
    for rvec, tvec in cam.poses(3, seed=3):
        image = render_board(rvec, tvec)
        view = target.detect(undistorter(np.dstack([image] * 3)), 0)
        pinhole, _ = cv2.projectPoints(
            cam.object_points(), rvec, tvec, out.camera_matrix(), None
        )
        pinhole = pinhole.reshape(-1, 2)
        # Board symmetry may flip the corner order: compare with the nearest point.
        gap = np.linalg.norm(view.image_points[:, None] - pinhole[None], axis=2)
        assert gap.min(axis=1).max() <= 0.5
        # Points of the distorted video (continuous convention) -> pinhole + 0.5.
        moved = undistorter.undistort_points(cam.project(rvec, tvec) + 0.5)
        assert moved == pytest.approx(pinhole + 0.5, abs=0.01)
    assert json.loads(json.dumps(undistorter.describe()))["alpha"] == 0.0


def test_check_camera_metadata(capsys):
    video = dict(IPHONE, make="Apple", software="27.0.1")
    ok = check_camera_metadata(IPHONE, video, "error")
    assert ok["status"] == "ok" and set(ok["compared"]) == set(IPHONE)
    other = dict(video, lens_model="iPhone 14 Pro Max back camera 2.22mm f/2.2")
    with pytest.raises(ConfigError, match="recorded with another camera.*lens_model"):
        check_camera_metadata(IPHONE, other, "error")
    warned = check_camera_metadata(IPHONE, other, "warn")
    assert warned["status"] == "mismatch" and list(warned["differences"]) == [
        "lens_model"
    ]
    assert "[warn] camera.calibration: camera mismatch" in capsys.readouterr().err
    assert check_camera_metadata(IPHONE, {}, "error")["status"] == "unavailable"
    assert check_camera_metadata({}, video, "error")["status"] == "unavailable"
    assert check_camera_metadata(IPHONE, other, "off")["status"] == "off"


class SizedSource(StillSource):
    def __init__(self, size):
        self.frame_size = size


def test_calibrated_intrinsics(monkeypatch):
    cal = calibration()
    provider = CalibratedIntrinsics(cal, label="calibration cal.yaml")
    with pytest.raises(ConfigError, match="calibrated for 1920x1080 frames"):
        provider.resolve(SizedSource((1280, 720)), device="cpu")
    result = provider.resolve(StillSource(), device="cpu")
    assert result.intrinsics == cal.intrinsics
    assert (result.source, result.details) == ("calibration cal.yaml", {})

    monkeypatch.setattr(camera, "estimate_focal_frame_px", lambda *a, **k: 1500.0)
    cross = CalibratedIntrinsics(
        cal, label="c", cross_check=EstimatedIntrinsics(EstimateConfig(frames=2))
    )
    details = cross.resolve(StillSource(), device="cpu").details
    assert details["estimated_focal_frame_px"] == 1500.0
    assert details["estimated_source"].startswith("estimated by da3-large-1.1")
    assert details["estimated_over_calibrated"] == pytest.approx(1500.0 / 1452.65)


class FakeVideo:
    """The part of VideoSource that build_camera uses."""

    def __init__(self, size=(1920, 1080), metadata=None, transform=None):
        self.frame_size, self.transform = size, transform
        self._metadata = IPHONE if metadata is None else metadata

    def camera_metadata(self):
        return self._metadata

    def with_transform(self, transform):
        return FakeVideo(self.frame_size, self._metadata, transform)


def test_build_camera_without_calibration_keeps_the_task_0006_path():
    # Regression invariant: camera.calibration null -> same source, same provider.
    video = FakeVideo()
    for cfg, kind in [
        (CameraConfig(focal_px=1000.0), FixedIntrinsics),
        (CameraConfig(), EstimatedIntrinsics),
    ]:
        setup = build_camera(cfg, video)
        assert setup.source is video and setup.details == {}
        assert type(setup.intrinsics) is kind
        assert type(build_intrinsics_provider(cfg)) is kind
    assert build_camera(CameraConfig(estimate=None), video).intrinsics is None


def test_build_camera_with_calibration(tmp_path):
    path = tmp_path / "cal.yaml"
    hd = calibration(camera=IPHONE)
    k = hd.intrinsics
    # The same camera calibrated at 3840x2160: continuous K doubles exactly (the
    # OpenCV cx would be 2 * cx_cv + 0.5, the half-pixel trap).
    uhd = replace(
        hd,
        image_size=(3840, 2160),
        intrinsics=CameraIntrinsics(2 * k.fx, 2 * k.fy, 2 * k.cx, 2 * k.cy),
    )
    assert uhd.camera_matrix()[0, 2] == pytest.approx(2 * K_CV[0, 2] + 0.5)
    save_calibration(uhd, path)
    video = FakeVideo()
    cfg = CameraConfig(calibration=CalibrationConfig(path=path))
    setup = build_camera(cfg, video)
    assert setup.source is video
    assert isinstance(setup.intrinsics, CalibratedIntrinsics)
    assert setup.intrinsics.cross_check is None
    assert setup.intrinsics.calibration.intrinsics == pytest.approx(
        calibration().intrinsics
    )
    details = setup.details
    assert details["scaled_from"] == [3840, 2160]
    assert details["lens_check"]["status"] == "ok"
    assert details["calibration"]["path"] == str(path)
    assert details["undistort"] is None
    json.dumps(details)  # metrics.json-ready

    undistorted = build_camera(
        replace(
            cfg,
            calibration=CalibrationConfig(path=path, undistort=True, cross_check=True),
        ),
        video,
    )
    assert isinstance(undistorted.source.transform, Undistorter)
    provider = undistorted.intrinsics
    assert provider.calibration == undistorted.source.transform.output
    assert provider.label.endswith("(undistorted)")
    assert isinstance(provider.cross_check, EstimatedIntrinsics)
    assert undistorted.details["undistort"]["alpha"] == 0.0


def test_build_camera_rejects_another_lens_before_any_model(tmp_path):
    path = tmp_path / "cal.yaml"
    save_calibration(calibration(camera=IPHONE), path)
    wide = FakeVideo(metadata=dict(IPHONE, focal_length_35mm=13))
    cfg = CameraConfig(calibration=CalibrationConfig(path=path))
    with pytest.raises(ConfigError, match="focal_length_35mm"):
        build_camera(cfg, wide)
    lenient = CameraConfig(calibration=CalibrationConfig(path=path, check_lens="warn"))
    assert build_camera(lenient, wide).details["lens_check"]["status"] == "mismatch"
    with pytest.raises(ConfigError, match="calibration file not found"):
        build_camera(
            CameraConfig(calibration=CalibrationConfig(path=tmp_path / "x.yaml")), wide
        )
