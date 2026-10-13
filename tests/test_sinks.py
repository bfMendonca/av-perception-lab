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
    FrameAgreement,
    FrameResult,
    MapGeometry,
    PairAgreement,
    Provenance,
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
        "n_sources",
        "sources",
        "source_scores",
        "raw_labels",
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
    # Unfused (concat) rows: one member, the instance itself.
    assert [rows[0][k] for k in ("n_sources", "sources", "source_scores")] == [
        "1",
        "yolo",
        "0.9123",
    ]
    assert rows[1]["raw_labels"] == "suv"
    # The task_0003 columns of unfused rows only restate the task_0002 ones.
    for row in rows:
        assert (row["sources"], row["raw_labels"]) == (row["source"], row["raw_label"])
        assert row["source_scores"] == row["confidence"]
    assert float(rows[0]["distance_m"]) == pytest.approx(12.0)
    assert rows[1]["distance_m"] == ""  # 25 px < min_valid_px


@pytest.fixture
def fused_result(result):
    """`result` with the first instance fused from two detectors (three members)."""
    members = (
        Provenance("yolo26l-objv1-150", "suv", 0.71),
        Provenance("rfdetr-large", "truck", 0.9031),
        Provenance("yolo26l-objv1-150", "pickup truck", 0.65),
    )
    sources = (members, (Provenance("yolo", "suv", 0.5),))
    instances = replace(
        result.instances.with_labels(["vehicle", "vehicle"]), sources=sources
    )
    pair = PairAgreement("rfdetr-large", "yolo26l-objv1-150", 3, 1, 2, 2.7)
    agreement = FrameAgreement(
        {"rfdetr-large": 4, "yolo26l-objv1-150": 5},
        {"rfdetr-large": 0, "yolo26l-objv1-150": 1},
        (pair,),
    )
    return replace(result, instances=instances, agreement=agreement)


def test_csv_sink_fused_row(ctx, fused_result):
    sink = CsvSink()
    sink.open(ctx)
    sink.write(fused_result)
    sink.close()
    with open(ctx.run_dir / "detections.csv", newline="") as fh:
        rows = list(csv.DictReader(fh))
    first = rows[0]
    assert first["class_name"] == "vehicle" and first["n_sources"] == "2"
    assert first["sources"] == "rfdetr-large|yolo26l-objv1-150"
    assert first["source_scores"] == "0.9031|0.71"
    assert first["raw_labels"] == "truck|suv"  # best member of each detector
    assert rows[1]["n_sources"] == "1"


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
    # One detector, concat: no agreement summary, every instance has one source.
    assert "agreement" not in metrics
    assert metrics["detections_by_n_sources"] == {"1": 2}
    assert metrics["detections_total"] == {"car": 1, "suv": 1}
    assert (metrics["boxes_total"], metrics["boxes_without_distance"]) == (2, 1)
    assert metrics["distance"]["k"] == 1.5
    assert "ultralytics" in metrics["versions"] and metrics["versions"]["gpu"] == "fake"
    assert math.isfinite(metrics["wall_time_s"])


def test_metrics_sink_agreement(ctx, result, fused_result):
    monitor = SimpleNamespace(
        summary=lambda: dict.fromkeys(
            [
                "peak_vram_torch_allocated_gb",
                "device_mem_used_delta_gb_approx",
                "torch",
                "cuda",
                "gpu",
            ]
        )
    )
    sink = MetricsSink(monitor, dict)
    sink.open(ctx)
    sink.write(result)  # no agreement in this frame (e.g. one detector)
    sink.write(fused_result)
    sink.close()
    metrics = json.loads((ctx.run_dir / "metrics.json").read_text())
    assert metrics["detections_by_n_sources"] == {"1": 3, "2": 1}
    agreement = metrics["agreement"]
    assert agreement["per_frame"]["frame"] == [7]
    assert agreement["pairs"][0]["mean_iou"] == pytest.approx(0.9)
    assert agreement["detectors"]["yolo26l-objv1-150"] == {
        "instances": 5,
        "duplicates": 1,
    }
    assert (agreement["iou_thr"], agreement["class_aware"]) == (0.5, True)


def test_metrics_sink_writes_nothing_without_frames(ctx):
    sink = MetricsSink(SimpleNamespace(), dict)
    sink.open(ctx)
    sink.close()
    assert not (ctx.run_dir / "metrics.json").exists()


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg/ffprobe")
def test_video_sink_single_detector_concat_is_unchanged(ctx, result, tmp_path):
    # Regression invariant: one detector + concat draws what task_0002 drew.
    clip = tmp_path / "clip.mp4"
    with VideoWriter(clip, 100, 100, "10/1") as writer:
        writer.write(np.zeros((100, 100, 3), np.uint8))
    ctx = replace(ctx, source=VideoSource(SourceConfig(video=clip)))
    sink = VideoSink(RenderConfig(), DistanceConfig())
    sink.open(ctx)
    drawn = []
    sink._renderer.render = lambda rgb, boxes, hud, depth: (
        drawn.append((boxes, hud)) or rgb
    )
    sink.write(result)
    sink.close()
    (boxes, hud) = drawn[0]
    assert [b.color_key for b in boxes] == [None, None]  # colour by label
    assert [b.label for b in boxes] == ["car 91% · 12.0 m", "suv 50% · n/a"]
    assert len(hud) == 4  # no fusion or agreement line
    assert hud[1].endswith("conf ≥ 0.40")


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


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg/ffprobe")
def test_video_sink_fused_labels_colours_and_hud(ctx, fused_result, tmp_path):
    clip = tmp_path / "clip.mp4"
    with VideoWriter(clip, 100, 100, "10/1") as writer:
        writer.write(np.zeros((100, 100, 3), np.uint8))
    detectors = [
        {"name": "rfdetr-large", "options": {"threshold": 0.5}},
        {"name": "yolo26l-objv1-150", "options": {"threshold": 0.4}},
    ]
    aggregator = {"type": "wbf", "options": {"iou_thr": 0.55}, "weights": {}}
    components = {
        **ctx.pipeline["components"],
        "detectors": detectors,
        "aggregator": aggregator,
    }
    ctx = replace(
        ctx,
        source=VideoSource(SourceConfig(video=clip)),
        pipeline={**ctx.pipeline, "components": components},
    )
    sink = VideoSink(RenderConfig(), DistanceConfig())
    sink.open(ctx)
    drawn = []
    sink._renderer.render = lambda rgb, boxes, hud, depth: (
        drawn.append((boxes, hud)) or rgb
    )
    sink.write(fused_result)
    sink.close()
    (boxes, hud) = drawn[0]
    assert [b.key for b in boxes] == ["vehicle", "vehicle"]  # colour by label
    assert boxes[0].label.startswith("vehicle 91% ×2 · ")
    assert boxes[1].label.startswith("vehicle 50% · ")
    assert hud[-2:] == [
        "wbf · IoU ≥ 0.55",
        "agree 3 · only rfdetr-large 1 · only yolo26l-objv1-150 2",
    ]
