import csv
import json
import math
import shutil
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from det_depth_distance.config import RenderConfig, RunConfig, SourceConfig
from det_depth_distance.core.pipeline import SetupInfo
from det_depth_distance.core.types import (
    CameraIntrinsics,
    DepthResult,
    Frame,
    FrameResult,
    MapGeometry,
)
from det_depth_distance.distance import DistanceConfig, box_depth_stats
from det_depth_distance.io.sinks import (
    CSV_FIELDS,
    CsvSink,
    MetricsSink,
    NpzSink,
    RunContext,
    VideoSink,
    latency_stats,
    make_run_dir,
)
from det_depth_distance.io.source import VideoSource
from det_depth_distance.io.video import VideoWriter, probe

GEOMETRY = MapGeometry(100, 100, 50, 50, 0.5, 0.5)
K = CameraIntrinsics(25.0, 25.0, 25.0, 25.0)


def test_csv_fields_order():
    # Columns of task_0001 first, in order; each later task appends its columns here.
    assert CSV_FIELDS == (
        "frame",
        "t",
        "class_name",
        "class_id",
        "confidence",
        "x0",
        "y0",
        "x1",
        "y1",
        "map_c0",
        "map_r0",
        "map_c1",
        "map_r1",
        "n_box_px",
        "n_valid",
        "n_inliers",
        "inlier_frac",
        "mean",
        "median",
        "std",
        "var",
        "p10",
        "p25",
        "p75",
        "p90",
        "min",
        "max",
        "distance_m",
        "source",
        "raw_label",
    )


def test_latency_stats():
    assert latency_stats([]) == {}
    s = latency_stats([1.0, 2.0, 3.0])
    assert s["mean"] == 2.0 and s["min"] == 1.0 and s["max"] == 3.0


def test_make_run_dir(tmp_path):
    run_dir = make_run_dir(
        tmp_path / "out", ["IMG_4021", "rfdetr-large", "", "da3metric-large"]
    )
    assert run_dir.is_dir()
    assert run_dir.name.endswith("_IMG_4021_rfdetr-large_da3metric-large")


@pytest.fixture
def result(make_instances):
    instances = make_instances(
        [[0, 0, 40, 40], [50, 50, 60, 60]],
        labels=["car", "suv"],
        score=[0.91234, 0.5],
        source="yolo",
    )
    depth = np.full((50, 50), 12.0, dtype=np.float32)
    stats = tuple(
        box_depth_stats(depth, box, GEOMETRY, DistanceConfig(min_valid_px=30))
        for box in instances.xyxy
    )
    frame = Frame(7, 0.23456, np.zeros((100, 100, 3), np.uint8))
    timings = {"det": 2.0, "det.yolo": 2.0, "aggregate": 0.1, "depth": 5.0, "dist": 0.2}
    return FrameResult(
        frame, instances, stats, DepthResult(depth, K, None, GEOMETRY), timings
    )


@pytest.fixture
def ctx(tmp_path):
    cfg = RunConfig(detectors=({"type": "yolo"},), depth={"type": "da3"})
    pipeline = {
        "components": {
            "detectors": [
                {"type": "yolo", "name": "yolo", "options": {"threshold": 0.4}}
            ],
            "depth": {
                "type": "da3",
                "name": "da3metric-large",
                "kind": "metric_focal",
                "fixed_intrinsics": {"fx": 1.0, "fy": 1.0, "cx": 0.0, "cy": 0.0},
            },
            "aggregator": {"type": "concat", "options": {}},
        },
        "pipeline": {"exclude_sky": True},
    }
    source = SimpleNamespace(describe=lambda: {"path": "clip.mov", "width": 100})
    setup = SetupInfo(CameraIntrinsics.centered(90.0, 100, 100), 90.0, "spy")
    return RunContext(tmp_path, source, setup, pipeline, cfg)


def test_csv_sink(ctx, result):
    sink = CsvSink()
    sink.open(ctx)
    sink.write(result)
    sink.close()
    with open(ctx.run_dir / "detections.csv", newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert tuple(rows[0]) == CSV_FIELDS
    assert [r["class_name"] for r in rows] == ["car", "suv"]
    assert rows[0]["t"] == "0.2346" and rows[0]["confidence"] == "0.9123"
    assert rows[0]["source"] == "yolo" and rows[1]["raw_label"] == "suv"
    assert float(rows[0]["distance_m"]) == pytest.approx(12.0)
    assert rows[1]["distance_m"] == ""  # 25 px < min_valid_px


def test_npz_sink(ctx, result):
    sink = NpzSink()
    sink.open(ctx)
    sink.write(result)
    sink.write(result)
    sink.close()
    data = np.load(ctx.run_dir / "depth.npz")
    assert set(data) == {
        "depth",
        "frame_index",
        "intrinsics",
        "geometry",
        "geometry_fields",
    }
    assert data["depth"].shape == (2, 50, 50) and data["depth"].dtype == np.float16
    assert data["intrinsics"][0].tolist() == [25.0, 25.0, 25.0, 25.0]
    assert list(data["geometry_fields"])[:2] == ["frame_w", "frame_h"]


def test_metrics_sink(ctx, result):
    monitor = SimpleNamespace(
        summary=lambda: {
            "peak_vram_torch_allocated_gb": 1.0,
            "device_mem_used_delta_gb_approx": 2.0,
            "torch": "2.8.0",
            "cuda": "12.8",
            "gpu": "fake",
        }
    )
    sink = MetricsSink(monitor, lambda: {"render": [3.0]})
    sink.open(ctx)
    sink.write(result)
    sink.close()
    metrics = json.loads((ctx.run_dir / "metrics.json").read_text())
    assert not {"detector", "args", "depth"} & set(metrics)
    assert metrics["components"]["depth"]["kind"] == "metric_focal"
    assert metrics["pipeline"] == {"exclude_sky": True}
    assert metrics["config"]["distance"]["exclude_sky"] is True
    assert set(metrics["timings_ms"]) == {
        "det",
        "det.yolo",
        "aggregate",
        "depth",
        "dist",
        "render",
    }
    assert metrics["focal_frame_px"] == 90.0 and metrics["focal_source"] == "spy"
    assert metrics["focal_map_intrinsics"]["fx"] == 1.0
    assert metrics["detections_by_source"] == {"yolo": 2}
    assert metrics["detections_total"] == {"car": 1, "suv": 1}
    assert (metrics["boxes_total"], metrics["boxes_without_distance"]) == (2, 1)
    assert metrics["distance"]["k"] == 1.5
    assert "ultralytics" in metrics["versions"] and metrics["versions"]["gpu"] == "fake"
    assert math.isfinite(metrics["wall_time_s"])


def test_metrics_sink_writes_nothing_without_frames(ctx):
    sink = MetricsSink(SimpleNamespace(), dict)
    sink.open(ctx)
    sink.close()
    assert not (ctx.run_dir / "metrics.json").exists()


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg/ffprobe")
def test_video_sink_colours_by_source_with_several_detectors(ctx, result, tmp_path):
    clip = tmp_path / "clip.mp4"
    with VideoWriter(clip, 100, 100, "10/1") as writer:
        writer.write(np.zeros((100, 100, 3), np.uint8))
    detectors = [
        {"name": "rfdetr-large", "options": {"threshold": 0.5}},
        {"name": "yolo", "options": {"threshold": 0.4}},
    ]
    components = {**ctx.pipeline["components"], "detectors": detectors}
    ctx = replace(
        ctx,
        source=VideoSource(SourceConfig(video=clip)),
        pipeline={**ctx.pipeline, "components": components},
    )
    sink = VideoSink(RenderConfig(layout="side-by-side"), DistanceConfig())
    sink.open(ctx)
    assert sink._conf == "conf rfdetr-large ≥ 0.50, yolo ≥ 0.40"
    sink.write(result)
    sink.close()
    assert probe(ctx.run_dir / "video.mp4").n_frames == 1
    assert len(sink.render_ms) == 1
