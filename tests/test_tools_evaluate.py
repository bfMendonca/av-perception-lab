import json

import numpy as np
import pytest
import yaml

from av_perception_lab.camera import Calibration, Undistorter, save_calibration
from av_perception_lab.tools.evaluate import build_parser, main

BOX = [900.0, 500.0, 1020.0, 580.0]


def row(frame, box, distance=8.0):
    x0, y0, x1, y1 = box
    return {"frame": frame, "t": frame / 30, "class_name": "car", "x0": x0, "y0": y0,
            "x1": x1, "y1": y1, "distance_m": distance, "source": "rfdetr-large",
            "raw_label": "car"}  # fmt: skip


def write_refs(path, box=BOX, video="IMG_4100.MOV"):
    reference = {
        "id": "r01",
        "frames": [0, 9],
        "select": {"box": box},
        "distance_m": 8.0,
        "instrument": "laser",
        "quantity": "z",
    }
    data = {
        "video": video,
        "instruments": {"laser": {"uncertainty_m": 0.01}},
        "references": [reference],
    }
    path.write_text(yaml.safe_dump(data))
    return path


@pytest.fixture
def refs(tmp_path):
    return write_refs(tmp_path / "field.yaml")


def test_parser():
    args = build_parser().parse_args(
        ["run", "--refs", "r.yaml", "--bins", "0,10,30", "--estimators", "median,min"]
    )
    assert args.bins == (0.0, 10.0, 30.0) and args.estimators == ("median", "min")
    assert args.level == "reference" and args.iou_min == 0.3 and not args.plot
    with pytest.raises(SystemExit):
        build_parser().parse_args(["run", "--refs", "r.yaml", "--bins", "0,a"])


def test_one_run(fake_run_dir, refs, capsys):
    run = fake_run_dir([row(f, BOX, 8.8) for f in range(10)])
    assert main(["--refs", str(refs), str(run)]) == 0
    out = run / "eval_field"
    data = json.loads((out / "eval.json").read_text())
    (overall,) = data["overall"]
    assert overall["median_abs_rel"] == pytest.approx(0.1)
    assert (out / "eval.md").is_file()
    assert f"[saved] {out / 'eval.md'}" in capsys.readouterr().out


def test_several_runs_and_errors(fake_run_dir, refs, tmp_path, capsys):
    runs = [str(fake_run_dir([row(f, BOX, d) for f in range(10)])) for d in (8.0, 9.0)]
    assert main(["--refs", str(refs), *runs]) == 2
    assert "--out is required with several runs" in capsys.readouterr().err
    out = tmp_path / "compare"
    assert (
        main(["--refs", str(refs), *runs, "--out", str(out), "--level", "frame"]) == 0
    )
    text = (out / "eval.md").read_text()
    assert "| run0 | rfdetr-large | distance_m |" in text
    assert "| run1 | rfdetr-large | distance_m |" in text
    bad = tmp_path / "bad.yaml"
    bad.write_text(yaml.safe_dump({"video": "x.MOV", "references": [{"id": "a"}]}))
    assert main(["--refs", str(bad), runs[0]]) == 2
    assert "references[0].select: missing required option" in capsys.readouterr().err
    other = write_refs(tmp_path / "other.yaml", video="IMG_4200.MOV")
    assert main(["--refs", str(other), runs[0]]) == 2
    assert "use --force" in capsys.readouterr().err
    assert main(["--refs", str(other), runs[0], "--force"]) == 0
    assert "[warn] run run0 processed IMG_4100.MOV" in capsys.readouterr().err
    assert main(["--refs", str(refs), str(tmp_path / "nope")]) == 2
    assert main(["--refs", str(refs), runs[0], "--estimators", "mode"]) == 2


def _camera_details(cal, cal_path, alpha):
    """metrics.json["camera"] as camera.build_camera records it with undistort."""
    return {
        "calibration": {"path": str(cal_path), **cal.describe()},
        "undistort": Undistorter(cal, alpha=alpha).describe(),
    }


def test_undistorted_run_moves_the_reference_box(fake_run_dir, tmp_path):
    k = np.array([[1400.0, 0.0, 959.5], [0.0, 1400.0, 539.5], [0.0, 0.0, 1.0]])
    cal = Calibration.from_opencv(k, np.array([0.25, -0.3, 0.0, 0.0]), (1920, 1080))
    cal_path = tmp_path / "cal.yaml"
    save_calibration(cal, cal_path)
    edge = [100.0, 100.0, 260.0, 200.0]  # near a corner: distortion moves it ~8 px
    edge_refs = write_refs(tmp_path / "edge.yaml", box=edge)
    # The box the run saw: the reference box (original video) after undistortion.
    x0, y0, x1, y1 = edge
    corners = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
    moved = Undistorter(cal, alpha=1.0).undistort_points(corners)
    seen = [*moved.min(0).tolist(), *moved.max(0).tolist()]
    camera = _camera_details(cal, cal_path, 1.0)
    # A later recalibration to the same path must not change the evaluation:
    # the transform comes from the K and distortion recorded in metrics.json.
    recalibrated = Calibration.from_opencv(k * 1.1, np.zeros(4), (1920, 1080))
    save_calibration(recalibrated, cal_path)
    rows = [row(f, seen) for f in range(10)]
    args = ["--refs", str(edge_refs), "--iou-min", "0.9", "--out"]
    for metrics, detected in [({"camera": camera}, 10), ({}, 0)]:
        out = tmp_path / f"eval{detected}"
        assert main([*args, str(out), str(fake_run_dir(rows, metrics))]) == 0
        (cov,) = json.loads((out / "eval.json").read_text())["coverage"]
        assert cov["detected_valid"] == detected  # 0: reference box not undistorted
    cal_path.unlink()  # the file is not needed at all
    assert (
        main(
            [
                *args,
                str(tmp_path / "nofile"),
                str(fake_run_dir(rows, {"camera": camera})),
            ]
        )
        == 0
    )


def test_undistorted_run_with_inconsistent_record_exits_2(
    fake_run_dir, tmp_path, capsys
):
    k = np.array([[1400.0, 0.0, 959.5], [0.0, 1400.0, 539.5], [0.0, 0.0, 1.0]])
    cal = Calibration.from_opencv(k, np.array([0.25, -0.3, 0.0, 0.0]), (1920, 1080))
    camera = _camera_details(cal, tmp_path / "cal.yaml", 0.0)
    camera["undistort"]["intrinsics"]["fx"] += 0.5  # not what this calibration gives
    refs = write_refs(tmp_path / "refs.yaml")
    run = fake_run_dir([row(0, BOX)], {"camera": camera})
    assert main(["--refs", str(refs), "--out", str(tmp_path / "e"), str(run)]) == 2
    assert "differs from the recorded one" in capsys.readouterr().err
    del camera["calibration"]["intrinsics"]
    run = fake_run_dir([row(0, BOX)], {"camera": camera})
    assert main(["--refs", str(refs), "--out", str(tmp_path / "e"), str(run)]) == 2
    assert "camera.calibration/undistort unusable" in capsys.readouterr().err


def test_mcap_run_with_start_is_refused(fake_run_dir, refs, capsys):
    run = fake_run_dir(
        [row(f, BOX, 8.8) for f in range(10)],
        {
            "video": {"path": "/d.mcap", "nominal_fps": 10.0, "time_base": "header"},
            "config": {
                "source": {"start": 2.0, "stride": 1, "max_frames": 0, "duration": 0},
                "distance": {"distance_type": "z"},
            },
        },
    )
    assert main(["--refs", str(refs), str(run)]) == 2
    err = capsys.readouterr().err
    assert "MCAP runs with source.start > 0 is not supported yet" in err
