"""End-to-end CLI runs on a ROS 2 MCAP of the synthetic scene (task_0033).

The scene frames (tests/smoke/smoke_scene.py) are written to a fixture MCAP
(tests/mcap_fixture.py) at irregular message times, with a CameraInfo whose
K is the camera the fake DA3 perceives, so every distance stays analytic
(`expected_box_distance`). The runs go through `cli.main` with the scene
doubles (smoke conftest `scene_runtime`), one output.out_root per run.
Checked: CSV `t` and video.mp4 PTS are the message times, the source
description in metrics.json (metrics["video"], the source's describe()),
intrinsics from the CameraInfo (`camera.calibration.from_source`), the
`rectify` stage on a distorted recording (boxes in rectified pixels, K and
distortion recorded for the evaluation), duplicate and zero header stamps,
a clean error for `rectify` without CameraInfo, and an unchanged video run.
"""

import itertools
from collections.abc import Callable, Sequence
from pathlib import Path

import av
import cv2
import numpy as np
import pytest
import yaml
from mcap_fixture import CameraInfoSpec, ImageSpec, write_mcap
from run_outputs import RunOutputs, check_run_outputs
from smoke_scene import SCENE, expected_box_distance, render_labels, true_box
from test_smoke_pipeline import OBJECT_OF_RAW, REL_TOL

from av_perception_lab import cli
from av_perception_lab.adapters.depth import da3
from av_perception_lab.camera import NO_CAMERA_INFO, Calibration, Undistorter
from av_perception_lab.core.pipeline import Pipeline
from av_perception_lab.core.types import CameraIntrinsics
from av_perception_lab.evaluation.run_data import load_run
from av_perception_lab.io.sinks import CSV_FIELDS

TOPIC = "/cam/image_raw"
INFO_TOPIC = "/cam/camera_info"
T0_NS = 1_762_000_000_123_456_789  # an epoch-like first stamp
# Irregular gaps between the 12 messages (ms): 60 to 140 ms around 10 Hz.
GAPS_MS = (100, 95, 130, 70, 100, 110, 90, 105, 140, 60, 100)
TIMES_NS = tuple(T0_NS + 1_000_000 * t for t in itertools.accumulate((0, *GAPS_MS)))
# A K unlike the base config's focal_px (725/3) and centred off the frame
# centre, OpenCV convention; D empty = a rectified CameraInfo (P is used).
INFO_K = (260.0, 0.0, 158.0, 0.0, 262.0, 88.0, 0.0, 0.0, 1.0)
INFO_INTRINSICS = CameraIntrinsics(260.0, 262.0, 158.5, 88.5)
# Radial barrel distortion of the rectify case (plumb_bob k1, k2, p1, p2, k3).
LENS_D = (-0.3, 0.08, 0.0, 0.0, 0.0)
PTS_TOL_S = 1e-3
# Detector pad (1 px) + one pixel of lens resampling (nearest, then bilinear).
BOX_TOL_PX = 2.0


def _grey(labels: np.ndarray) -> np.ndarray:
    """RGB grey frame of a label image (the scene's grey levels)."""
    greys = np.array([c.grey for c in SCENE.classes], dtype=np.uint8)
    return np.repeat(greys[labels][..., None], 3, axis=-1)


def _scene_mcap(
    path: Path,
    labels: Sequence[np.ndarray] | None = None,
    *,
    times_ns: Sequence[int] = TIMES_NS,
    header_ns: Sequence[int] | None = None,
    compressed: str | None = None,
    info: CameraInfoSpec | None = None,
) -> Path:
    """MCAP of the scene frames on TOPIC (with INFO_TOPIC when `info`)."""
    labels = labels if labels is not None else [
        render_labels(SCENE, i) for i in range(SCENE.n_frames)
    ]  # fmt: skip
    image = ImageSpec(
        topic=TOPIC,
        frames=[_grey(f) for f in labels],
        times_ns=times_ns,
        header_ns=header_ns,
        compressed=compressed,
    )
    return write_mcap(path, images=[image], camera_infos=[info] if info else [])


def _info(k: Sequence[float], d: Sequence[float] = ()) -> CameraInfoSpec:
    w, h = SCENE.size
    return CameraInfoSpec(topic=INFO_TOPIC, width=w, height=h, k=k, d=d)


@pytest.fixture
def run_mcap(run_cli) -> Callable[..., Path]:
    """Factory: run_mcap(mcap, overrides=(), *, perceived) -> run folder
    (the smoke conftest's run_cli with source.mcap/source.topic, no video)."""

    def run(
        mcap: Path, overrides: Sequence[str] = (), *, perceived: CameraIntrinsics
    ) -> Path:
        return run_cli(
            None,
            [f"source.mcap={mcap}", f"source.topic={TOPIC}", *overrides],
            perceived=perceived,
        )

    return run


def _check_distances(out: RunOutputs, n_frames: int = SCENE.n_frames) -> None:
    """Box-only detector, defaults: 3 rows per frame, analytic z distances."""
    frames = [int(r["frame"]) for r in out.rows]
    assert sorted(set(frames)) == list(range(n_frames))
    assert all(frames.count(f) == 3 for f in set(frames))
    assert out.metrics["boxes_without_distance"] == 0
    for row in out.rows:
        z = expected_box_distance(
            SCENE,
            OBJECT_OF_RAW[row["raw_label"]],
            exclude_sky=True,
            excluded=(),
            box_shrink=0.1,
        )
        assert float(row["distance_m"]) == pytest.approx(z, rel=REL_TOL), row


def _frame_times(rows: Sequence[dict[str, str]]) -> dict[int, float]:
    return {int(r["frame"]): float(r["t"]) for r in rows}


def _video_pts(path: Path) -> list[float]:
    """Presentation times (s) of every decoded frame of an mp4, in order."""
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        return sorted(
            float(frame.pts * stream.time_base) for frame in container.decode(stream)
        )


@pytest.mark.parametrize("compressed", [None, "png"], ids=["rgb8", "png"])
def test_mcap_run_produces_outputs_with_message_times(
    run_mcap, scene_videos, tmp_path, compressed
):
    mcap = _scene_mcap(tmp_path / "scene.mcap", compressed=compressed)
    out = check_run_outputs(run_mcap(mcap, perceived=scene_videos.true_k))
    _check_distances(out)

    expected = [(t - TIMES_NS[0]) / 1e9 for t in TIMES_NS]
    times = _frame_times(out.rows)
    assert [times[i] for i in range(SCENE.n_frames)] == pytest.approx(
        expected, abs=0.5e-4
    )  # CSV t is rounded to 4 decimals

    source = out.metrics["video"]  # the source's describe()
    assert source["path"] == str(mcap) and source["topic"] == TOPIC
    assert source["schema"] == (
        "sensor_msgs/msg/CompressedImage" if compressed else "sensor_msgs/msg/Image"
    )
    assert source["time_base"] == "header" and source["duplicates_dropped"] == 0
    assert source["count"] == SCENE.n_frames and source["camera_info_topic"] is None
    assert source["nominal_fps"] == pytest.approx(1000 / np.median(GAPS_MS))
    assert out.metrics["config"]["source"]["mcap"] == str(mcap)
    assert out.metrics["config"]["source"]["topic"] == TOPIC

    pts = _video_pts(out.run_dir / "video.mp4")
    assert pts == pytest.approx(expected, abs=PTS_TOL_S)


def test_mcap_camera_info_intrinsics_used(run_mcap, tmp_path):
    mcap = _scene_mcap(tmp_path / "scene.mcap", info=_info(INFO_K))
    out = check_run_outputs(
        run_mcap(
            mcap,
            ["camera.focal_px=null", "camera.calibration={from_source: true}"],
            perceived=INFO_INTRINSICS,
        )
    )
    # The fake DA3 sees INFO_INTRINSICS: z is analytic only when the pipeline
    # scales with the same focal (725/3 would be 7 % off).
    _check_distances(out)
    m = out.metrics
    assert m["frame_intrinsics"] == pytest.approx(
        {"fx": 260.0, "fy": 262.0, "cx": 158.5, "cy": 88.5}
    )
    assert m["video"]["camera_info_topic"] == INFO_TOPIC
    calibration = Calibration.from_description(m["camera"]["calibration"])
    assert calibration.intrinsics == INFO_INTRINSICS
    assert not np.any(calibration.dist_array())
    assert m["camera"]["undistort"] is None and m["video"]["transform"] is None


def test_mcap_rectify_end_to_end(run_mcap, scene_videos, distort_image, tmp_path):
    # A strong barrel lens on the scene's pinhole K (the smoke conftest's
    # true lens moves the boxes by ~2 px only at 320x180).
    kt = scene_videos.true_k
    k = np.array([[kt.fx, 0, kt.cx - 0.5], [0, kt.fy, kt.cy - 0.5], [0, 0, 1]])
    true = Calibration.from_opencv(k, np.array(LENS_D), SCENE.size)
    kr = Undistorter(true, alpha=0.0).output.intrinsics
    distorted = [
        distort_image(
            render_labels(SCENE, i),
            k,
            np.array(LENS_D),
            interpolation=cv2.INTER_NEAREST,
            border_value=SCENE.index("road"),
        )
        for i in range(SCENE.n_frames)
    ]
    mcap = _scene_mcap(
        tmp_path / "lens.mcap", distorted, info=_info(k.ravel().tolist(), LENS_D)
    )
    run_dir = run_mcap(
        mcap, ["camera.focal_px=null", "rectify={enabled: true}"], perceived=kr
    )
    out = check_run_outputs(run_dir)
    _check_distances(out)

    # Boxes are in rectified pixels: the ideal (pinhole) box of every object
    # mapped from the true K to the rectified K, not the distorted one.
    def rectified(box):
        x0, y0, x1, y1 = box
        sx, sy = kr.fx / kt.fx, kr.fy / kt.fy
        return (
            kr.cx + sx * (x0 - kt.cx),
            kr.cy + sy * (y0 - kt.cy),
            kr.cx + sx * (x1 - kt.cx),
            kr.cy + sy * (y1 - kt.cy),
        )

    moved = 0.0
    for row in out.rows:
        i, cls = int(row["frame"]), OBJECT_OF_RAW[row["raw_label"]]
        got = np.array([float(row[c]) for c in ("x0", "y0", "x1", "y1")])
        want = np.array(rectified(true_box(render_labels(SCENE, i), SCENE, cls)))
        assert np.abs(got - want).max() <= BOX_TOL_PX, (row, want)
        lens_box = np.array(true_box(distorted[i], SCENE, cls))
        moved = max(moved, np.abs(got - lens_box).max())
    assert moved > 2 * BOX_TOL_PX  # rectification visibly moved the boxes

    camera = out.metrics["camera"]
    assert camera["undistort"]["alpha"] == 0.0
    assert out.metrics["video"]["transform"] is not None
    assert out.metrics["frame_intrinsics"] == pytest.approx(
        {"fx": kr.fx, "fy": kr.fy, "cx": kr.cx, "cy": kr.cy}
    )
    # The evaluation rebuilds the lens from metrics.json to transform
    # references annotated on the original (distorted) images.
    run = load_run(run_dir)
    assert run.is_undistorted
    calibration, alpha = run.undistort_calibration
    assert alpha == 0.0
    assert calibration.intrinsics == pytest.approx(true.intrinsics)
    np.testing.assert_allclose(calibration.dist_array().ravel(), LENS_D)
    assert tuple(calibration.image_size) == SCENE.size


def test_mcap_rectify_without_camera_info_fails_cleanly(
    scene_runtime, tmp_path, monkeypatch, capsys
):
    def no_model(*args, **kwargs):
        raise AssertionError("a model was loaded before the configuration error")

    monkeypatch.setattr(da3, "_load_da3", no_model)
    monkeypatch.setattr(Pipeline, "load", no_model)
    monkeypatch.setattr(Pipeline, "resolve_intrinsics", no_model)
    monkeypatch.setattr(cli, "require_ready", no_model)
    mcap = _scene_mcap(tmp_path / "scene.mcap")  # no CameraInfo
    config = tmp_path / "base.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "device": "cpu",
                "detectors": [{"type": "scene_det", "name": "box"}],
                "depth": {"type": "da3", "model": "da3metric-large"},
                "camera": {"estimate": None},
                "rectify": {"enabled": True},
            }
        )
    )
    out_root = tmp_path / "out"
    argv = ["--config", str(config)]
    for override in (
        f"source.mcap={mcap}",
        f"source.topic={TOPIC}",
        f"output.out_root={out_root}",
    ):
        argv += ["--set", override]
    assert cli.main(argv) == 2
    err = capsys.readouterr().err
    assert "rectify.enabled" in err and NO_CAMERA_INFO in err, err
    assert not out_root.exists()  # nothing written


def test_mcap_duplicate_timestamps_and_zero_header_stamps(
    run_mcap, scene_videos, tmp_path
):
    # Message 3 is sent twice at the same log time; every header stamp is 0,
    # so the source falls back to the log time and drops the repeat.
    labels = [render_labels(SCENE, i) for i in range(SCENE.n_frames)]
    labels.insert(4, labels[3])
    times = [*TIMES_NS[:4], TIMES_NS[3], *TIMES_NS[4:]]
    mcap = _scene_mcap(
        tmp_path / "dup.mcap", labels, times_ns=times, header_ns=[0] * len(times)
    )
    out = check_run_outputs(run_mcap(mcap, perceived=scene_videos.true_k))
    _check_distances(out)
    source = out.metrics["video"]
    assert source["time_base"] == "log" and source["duplicates_dropped"] == 1
    assert source["count"] == SCENE.n_frames + 1
    assert out.metrics["frames_processed"] == SCENE.n_frames
    times_s = _frame_times(out.rows)
    t = [times_s[i] for i in range(SCENE.n_frames)]
    assert all(b > a for a, b in itertools.pairwise(t))
    assert t == pytest.approx([(x - TIMES_NS[0]) / 1e9 for x in TIMES_NS], abs=0.5e-4)


def test_video_run_unchanged(run_cli, scene_videos):
    out = check_run_outputs(
        run_cli(scene_videos.pinhole, perceived=scene_videos.true_k)
    )
    assert len(CSV_FIELDS) == 61 and len(out.rows[0]) == 61
    _check_distances(out)
    times = _frame_times(out.rows)
    assert [times[i] for i in range(SCENE.n_frames)] == pytest.approx(
        [i / 10 for i in range(SCENE.n_frames)]
    )
    source = out.metrics["video"]
    assert source["fps"] == 10.0 and "topic" not in source
    assert out.metrics["config"]["source"]["mcap"] is None
    assert out.video.n_frames == SCENE.n_frames
