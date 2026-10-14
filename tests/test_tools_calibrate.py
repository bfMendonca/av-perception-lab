import re
import shutil
import subprocess
from dataclasses import replace

import cv2
import numpy as np
import pytest
from PIL import Image

from det_depth_distance.calibration import CalibrationReport
from det_depth_distance.camera import Calibration, load_calibration
from det_depth_distance.io.video import VideoWriter
from det_depth_distance.tools.calibrate import _acceptance, build_parser, main

needs_ffmpeg = pytest.mark.skipif(
    shutil.which("ffmpeg") is None, reason="needs ffmpeg/ffprobe"
)
LENS = "iPhone 14 Pro Max back triple camera 6.86mm f/1.78"


def test_parser():
    args = build_parser().parse_args(["calibrate", "v.mov", "--out", "c.yaml"])
    assert (args.every_s, args.max_views, args.min_views) == (0.5, 60, 12)
    assert args.target == "chessboard" and not args.no_fix_k3
    with pytest.raises(SystemExit):
        build_parser().parse_args([])  # a subcommand is required


def test_calibrate_images(board_images, synthetic_camera, tmp_path, capsys):
    out = tmp_path / "cal" / "camera.yaml"
    debug = tmp_path / "debug"
    code = main(
        [
            "calibrate",
            "--images",
            str(board_images / "*.png"),
            "--out",
            str(out),
            "--debug-dir",
            str(debug),
            "--note",
            "synthetic",
        ]
    )
    assert code == 0
    cal = load_calibration(out)
    assert cal.image_size == synthetic_camera.size
    assert cal.camera_matrix()[0, 0] == pytest.approx(725.0, rel=5e-3)
    assert cal.meta["target"]["cols"] == 9 and cal.meta["n_detected"] == 14
    assert cal.meta["recording"]["notes"] == "synthetic" and cal.camera == {}
    assert len(list(debug.glob("view_*.png"))) == 14
    captured = capsys.readouterr()
    printed = captured.out
    assert "[fit] RMS" in printed and f"[saved] {out}" in printed
    # Every acceptance criterion of the protocol is reported, passed or not.
    report = printed + captured.err
    for criterion in ("RMS", "coverage", "std of fx"):
        assert re.search(rf"\[(accept\] ok|warn\] not accepted): {criterion}", report)
    assert "[accept] ok: RMS" in printed  # rendered boards: ~0.04 px


def test_acceptance_criteria():
    cal = Calibration.from_opencv(
        np.array([[1000.0, 0, 959.5], [0, 1000.0, 539.5], [0, 0, 1]]),
        np.zeros(4),
        (1920, 1080),
        rms_px=0.6,
        std={"fx": 4.0},
    )
    report = CalibrationReport(cal, (0,), (), (0.6,), 0.7)
    passed = [ok for ok, _ in _acceptance(report)]
    assert passed == [False, True, True]  # RMS 0.6 px; coverage 0.7; std 0.4 %
    report = CalibrationReport(
        replace(cal, rms_px=0.3, std={"fx": 6.0}), (0,), (), (0.3,), 0.5
    )
    assert [ok for ok, _ in _acceptance(report)] == [True, False, False]


def test_calibrate_errors(tmp_path, capsys):
    empty = tmp_path / "empty"
    empty.mkdir()
    for i in range(3):
        cv2.imwrite(str(empty / f"{i}.png"), np.full((60, 80), 255, np.uint8))
    out = str(tmp_path / "c.yaml")
    assert main(["calibrate", "--images", str(empty / "*.png"), "--out", out]) == 2
    assert "0 views with the target" in capsys.readouterr().err
    assert (
        main(["calibrate", "--images", str(tmp_path / "none*.png"), "--out", out]) == 2
    )
    assert "no file matches" in capsys.readouterr().err
    assert main(["calibrate", "--out", out]) == 2
    assert "give a video or --images" in capsys.readouterr().err
    assert (
        main(["calibrate", "--images", "x", "--target", "charuco", "--out", out]) == 2
    )
    assert "marker_mm" in capsys.readouterr().err


def test_board(tmp_path, capsys):
    out = tmp_path / "board.png"
    assert main(["board", "--out", str(out), "--dpi", "254", "--margin-mm", "5"]) == 0
    image = cv2.imread(str(out), cv2.IMREAD_GRAYSCALE)
    # 10 x 7 squares of 25 mm at 10 px/mm plus 5 mm margins.
    assert image.shape == (7 * 250 + 100, 10 * 250 + 100)
    with Image.open(out) as png:  # resolution stored for printing at 100 %
        assert png.info["dpi"] == pytest.approx((254, 254), abs=0.01)
    assert "print at 100 %" in capsys.readouterr().out
    charuco = tmp_path / "charuco.png"
    args = ["board", "--target", "charuco", "--marker-mm", "18", "--out", str(charuco)]
    assert main(args) == 0 and charuco.is_file()


@needs_ffmpeg
def test_inspect(tmp_path, capsys):
    video = tmp_path / "tagged.mov"
    subprocess.run(
        [
            "ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc=size=192x108:rate=10",
            "-t", "0.5", "-pix_fmt", "yuv420p", "-movflags", "use_metadata_tags",
            "-metadata", "com.apple.quicktime.model=iPhone 14 Pro Max",
            "-metadata", f"com.apple.quicktime.camera.lens_model={LENS}",
            str(video),
        ],
        check=True,
    )  # fmt: skip
    assert main(["inspect", str(video)]) == 0
    printed = capsys.readouterr().out
    assert f"[camera] lens_model: {LENS}" in printed and "192x108" in printed
    # A calibration recorded with another lens: reported, exit code 1.
    cal = tmp_path / "cal.yaml"
    fs = cv2.FileStorage(str(cal), cv2.FILE_STORAGE_WRITE)
    fs.write("image_width", 1920)
    fs.write("image_height", 1080)
    fs.write(
        "camera_matrix", np.array([[1400.0, 0, 959.5], [0, 1400.0, 539.5], [0, 0, 1]])
    )
    fs.write("distortion_coefficients", np.zeros((1, 5)))
    fs.startWriteStruct("camera", cv2.FileNode_MAP)
    fs.write("lens_model", "iPhone 14 Pro Max back camera 2.22mm f/2.2")
    fs.endWriteStruct()
    fs.release()
    assert main(["inspect", str(video), "--calibration", str(cal)]) == 1
    out = capsys.readouterr()
    assert "fx 140.00" in out.out and "camera check: mismatch" in out.out
    assert "camera mismatch" in out.err
    assert main(["inspect", str(tmp_path / "missing.mov")]) == 2


@needs_ffmpeg
def test_calibrate_video(synthetic_camera, render_board, tmp_path, capsys):
    video = tmp_path / "board.mp4"
    w, h = synthetic_camera.size
    with VideoWriter(video, w, h, "10/1", crf=12) as writer:
        for rvec, tvec in synthetic_camera.poses(15, seed=6):
            writer.write(np.dstack([render_board(rvec, tvec)] * 3))
    out = tmp_path / "cal.json"
    assert main(["calibrate", str(video), "--every-s", "0.1", "--out", str(out)]) == 0
    cal = load_calibration(out)
    assert cal.meta["n_candidates"] == 15 and cal.meta["recording"]["fps"] == 10.0
    assert cal.camera_matrix()[0, 0] == pytest.approx(725.0, rel=1e-2)
