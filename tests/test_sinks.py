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
    SemanticMap,
    TrackEstimate,
)
from det_depth_distance.distance import (
    DistanceConfig,
    box_depth_stats,
    instance_depth_stats,
    resample_mask,
)
from det_depth_distance.io.sinks import (
    CSV_FIELDS,
    CsvSink,
    MetricsSink,
    NpzSink,
    RunContext,
    VideoSink,
    _region_comparison,
    latency_stats,
    make_run_dir,
)
from det_depth_distance.io.source import VideoSource
from det_depth_distance.io.video import VideoWriter, probe

GEOMETRY = MapGeometry(100, 100, 50, 50, 0.5, 0.5)
MASK_COLUMNS = ("used_mask", "mask_px", "mask_erode", "box_distance_m")  # task_0004
K = CameraIntrinsics(25.0, 25.0, 25.0, 25.0)
TRACK_COLUMNS = (  # task_0006
    "track_id",
    "distance_smooth_m",
    "distance_smooth_std_m",
    "rel_speed_mps",
    "rel_speed_std_mps",
)


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
        "used_mask",
        "mask_px",
        "mask_erode",
        "box_distance_m",
        "mask_source",
        "mask_score",
        "track_id",
        "distance_smooth_m",
        "distance_smooth_std_m",
        "rel_speed_mps",
        "rel_speed_std_mps",
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
            "refiners": [],
            "semantic": [],
            "tracker": None,
        },
        "pipeline": {
            "exclude_sky": True,
            "agreement": None,
            "region": "mask",
            "mask_erode": 1,
            "exclude_semantic": {},
            "smoothing": None,
        },
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
    # Box rows: the task_0004 columns are empty, used_mask is 0; no mask, no score.
    for row in rows:
        assert [row[k] for k in MASK_COLUMNS] == ["0", "", "", ""]
        assert (row["mask_source"], row["mask_score"]) == ("", "")
        # No tracker, no smoothing: the task_0006 columns are empty.
        assert [row[k] for k in TRACK_COLUMNS] == [""] * 5


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
    assert metrics["pipeline"] == ctx.pipeline["pipeline"]
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
    assert metrics["distance_regions"] == {"mask": 0, "box": 2}
    assert "region_comparison" not in metrics  # region "mask"
    assert metrics["distance"]["k"] == 1.5
    assert "ultralytics" in metrics["versions"] and metrics["versions"]["gpu"] == "fake"
    assert {"transformers", "timm"} <= set(metrics["versions"])  # None without infer
    # No refiner and no semantic segmenter: the box rows have no mask producer.
    assert (metrics["mask_sources"], metrics["refined"]) == ({}, {})
    assert "semantic" not in metrics
    assert metrics["distance"]["exclude_semantic"] == []
    assert math.isfinite(metrics["wall_time_s"])


def _monitor():
    keys = (
        "peak_vram_torch_allocated_gb",
        "device_mem_used_delta_gb_approx",
        "torch",
        "cuda",
        "gpu",
    )
    return SimpleNamespace(summary=lambda: dict.fromkeys(keys))


def _metrics(ctx, result):
    sink = MetricsSink(_monitor(), dict)
    sink.open(ctx)
    sink.write(result)
    sink.close()
    return json.loads((ctx.run_dir / "metrics.json").read_text())


def test_metrics_sink_camera_without_calibration(ctx, result):
    # task_0007 keys; without camera.calibration the camera section is empty.
    metrics = _metrics(ctx, result)
    assert metrics["frame_intrinsics"] == {
        "fx": 90.0,
        "fy": 90.0,
        "cx": 50.0,
        "cy": 50.0,
    }
    assert metrics["camera"] == {}


def test_metrics_sink_camera_with_calibration(ctx, result):
    k = CameraIntrinsics(91.0, 92.0, 49.5, 51.0)
    setup = SetupInfo(k, 91.0, "calibration c.yaml", {"estimated_focal_frame_px": 95.0})
    camera = {"calibration": {"path": "c.yaml"}, "undistort": None}
    metrics = _metrics(replace(ctx, setup=setup, camera=camera), result)
    assert metrics["frame_intrinsics"] == {
        "fx": 91.0,
        "fy": 92.0,
        "cx": 49.5,
        "cy": 51.0,
    }
    assert metrics["camera"] == {**camera, "estimated_focal_frame_px": 95.0}
    assert metrics["focal_source"] == "calibration c.yaml"
    no_k = SetupInfo(None, None, "predicted per frame by the depth model")
    assert _metrics(replace(ctx, setup=no_k), result)["frame_intrinsics"] is None


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
    sink._renderer.render = lambda rgb, boxes, hud, depth, **kw: (
        drawn.append((boxes, hud, kw)) or rgb
    )
    sink.write(result)
    sink.close()
    (boxes, hud, kw) = drawn[0]
    assert kw == {"semantic": []}  # no semantic segmenter: no layer
    assert [b.color_key for b in boxes] == [None, None]  # colour by label
    assert [b.label for b in boxes] == ["car 91% · 12.0 m", "suv 50% · n/a"]
    assert len(hud) == 4  # no fusion or agreement line
    assert hud[1].endswith("conf ≥ 0.40")
    # No instance used a mask: the distance line is task_0002's.
    assert hud[2] == "median · iqr · shrink 0.1 · z"
    assert [b.mask for b in boxes] == [None, None]


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
    sink._renderer.render = lambda rgb, boxes, hud, depth, **kw: (
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


@pytest.fixture
def both_result(result):
    """`result` measured with region "both": the first instance has a mask over
    a thin 4 m strip of the 12 m map (mask 4 m; in the box the strip is an
    outlier, 12 m), the second has none."""
    depth = result.depth.depth_m.copy()
    depth[:, 5:8] = 4.0
    masks = np.zeros((2, 100, 100), dtype=bool)
    masks[0, :80, 10:16] = True  # frame columns 10..16 -> map columns 5..8
    instances = replace(result.instances, masks=masks)
    cfg = DistanceConfig(min_valid_px=30)
    pairs = [
        instance_depth_stats(
            depth, box, GEOMETRY, cfg, region="both", instance_mask=masks[i]
        )
        for i, box in enumerate(instances.xyxy)
    ]
    return replace(
        result,
        instances=instances,
        stats=tuple(p for p, _ in pairs),
        depth=replace(result.depth, depth_m=depth),
        box_stats=tuple(b for _, b in pairs),
    )


def test_csv_sink_both_row(ctx, both_result):
    sink = CsvSink()
    sink.open(ctx)
    sink.write(both_result)
    sink.close()
    with open(ctx.run_dir / "detections.csv", newline="") as fh:
        rows = list(csv.DictReader(fh))
    first, second = rows
    stats = both_result.stats[0]
    assert first["used_mask"] == "1" and float(first["distance_m"]) == pytest.approx(
        4.0
    )
    # 3 map columns: one erosion would leave 18 px < min_valid_px, so none is applied.
    assert first["mask_px"] == str(stats.mask_px) == "60" and first["mask_erode"] == "0"
    assert float(first["box_distance_m"]) == pytest.approx(12.0)
    assert [second[k] for k in MASK_COLUMNS] == ["0", "", "", ""]
    # Detector mask (no refiner): its producer is the instance's source, no score.
    assert (first["mask_source"], first["mask_score"]) == ("yolo", "")
    assert (second["mask_source"], second["mask_score"]) == ("", "")


def test_region_comparison_by_hand():
    assert _region_comparison([]) == {"n": 0}
    out = _region_comparison(
        [("person", 8.0, 10.0), ("person", 9.0, 9.5), ("car", 12.0, 12.0)]
    )
    # differences -2, -0.5, 0; relative -0.2, -0.0526, 0
    assert out["n"] == 3
    assert out["median_signed_m"] == pytest.approx(-0.5)
    assert out["median_abs_m"] == pytest.approx(0.5)
    assert out["p90_abs_m"] == pytest.approx(1.7)
    assert out["median_rel"] == pytest.approx(-0.5 / 9.5)
    person = out["by_label"]["person"]
    assert person["n"] == 2 and person["median_signed_m"] == pytest.approx(-1.25)
    assert person["p90_abs_m"] == pytest.approx(1.85)
    assert person["median_rel"] == pytest.approx((-0.2 - 0.5 / 9.5) / 2)
    assert out["by_label"]["car"]["median_abs_m"] == 0.0
    json.dumps(out)


def test_metrics_sink_region_comparison(ctx, both_result):
    cfg = replace(ctx.config, region="both")
    ctx = replace(ctx, config=cfg)
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
    sink.write(both_result)
    sink.close()
    metrics = json.loads((ctx.run_dir / "metrics.json").read_text())
    assert metrics["distance_regions"] == {"mask": 1, "box": 1}
    comparison = metrics["region_comparison"]
    assert comparison["n"] == 1  # the second instance has no box/mask pair
    assert comparison["median_signed_m"] == pytest.approx(4.0 - 12.0)
    assert set(comparison["by_label"]) == {"car"}
    assert metrics["distance"]["region"] == "both"


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg/ffprobe")
@pytest.mark.parametrize("mask_alpha", [0.35, 0.0])
def test_video_sink_masks_and_region_hud(ctx, both_result, tmp_path, mask_alpha):
    clip = tmp_path / "clip.mp4"
    with VideoWriter(clip, 100, 100, "10/1") as writer:
        writer.write(np.zeros((100, 100, 3), np.uint8))
    ctx = replace(ctx, source=VideoSource(SourceConfig(video=clip)))
    sink = VideoSink(RenderConfig(mask_alpha=mask_alpha), DistanceConfig())
    sink.open(ctx)
    drawn = []
    sink._renderer.render = lambda rgb, boxes, hud, depth, **kw: (
        drawn.append((boxes, hud)) or rgb
    )
    sink.write(both_result)
    sink.close()
    (boxes, hud) = drawn[0]
    if mask_alpha:
        np.testing.assert_array_equal(boxes[0].mask, both_result.instances.masks[0])
        assert boxes[1].mask is not None and not boxes[1].mask.any()
    else:
        assert [b.mask for b in boxes] == [None, None]
    # One instance used its mask: the distance line names the policy.
    assert hud[2] == "median · iqr · shrink 0.1 · z · region mask · erode 1"
    assert len(hud) == 4


# task_0005: refiner and semantic outputs. The refiner name contains a dot, as
# default names derived from checkpoints do.
SAM = "sam2.1-hiera-large"
SCENE = ("road", "sky")
LANES = ("background", "lane")


def _maps() -> dict[str, SemanticMap]:
    scene = np.ones((100, 100), np.int32)
    scene[:, :50] = 0  # road: left half
    lanes = np.zeros((100, 100), np.int32)
    lanes[:30] = 1  # lane: top 30 rows (overlaps the road on the top-left)
    return {"scene": SemanticMap(scene, SCENE), "lanes": SemanticMap(lanes, LANES)}


@pytest.fixture
def refined_result(result):
    """`result` after a SAM refiner: a mask on the first instance (score 0.87654),
    an empty refined mask on the second; with two semantic maps."""
    masks = np.zeros((2, 100, 100), dtype=bool)
    masks[0, 5:30, 5:30] = True
    instances = replace(
        result.instances,
        masks=masks,
        mask_source=(SAM, ""),
        mask_score=np.array([0.87654, np.nan], np.float32),
    )
    return replace(result, instances=instances, semantic=_maps())


@pytest.fixture
def semantic_ctx(ctx):
    components = {
        **ctx.pipeline["components"],
        "refiners": [{"type": "sam2", "name": SAM, "options": {"targets": "missing"}}],
        "semantic": [
            {"type": "segformer", "name": "scene", "options": {}},
            {"type": "twinlitenet", "name": "lanes", "options": {}},
        ],
    }
    policy = {
        **ctx.pipeline["pipeline"],
        "exclude_semantic": {"scene": ["road"], "lanes": ["lane"]},
    }
    return replace(ctx, pipeline={"components": components, "pipeline": policy})


def test_csv_sink_refined_rows(semantic_ctx, refined_result):
    sink = CsvSink()
    sink.open(semantic_ctx)
    sink.write(refined_result)
    sink.close()
    with open(semantic_ctx.run_dir / "detections.csv", newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert tuple(rows[0]) == CSV_FIELDS  # no semantic column, even with segmenters
    assert (rows[0]["mask_source"], rows[0]["mask_score"]) == (SAM, "0.8765")
    assert (rows[1]["mask_source"], rows[1]["mask_score"]) == ("", "")


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg/ffprobe")
@pytest.mark.parametrize(
    ("overlay", "drawn"), [(None, ["scene", "lanes"]), (("lanes",), ["lanes"])]
)
def test_video_sink_semantic_layers_and_hud(
    semantic_ctx, refined_result, tmp_path, overlay, drawn
):
    clip = tmp_path / "clip.mp4"
    with VideoWriter(clip, 100, 100, "10/1") as writer:
        writer.write(np.zeros((100, 100, 3), np.uint8))
    ctx = replace(semantic_ctx, source=VideoSource(SourceConfig(video=clip)))
    sink = VideoSink(RenderConfig(semantic_overlay=overlay), DistanceConfig())
    sink.open(ctx)
    calls = []
    sink._renderer.render = lambda rgb, boxes, hud, depth, **kw: (
        calls.append((hud, kw)) or rgb
    )
    sink.write(refined_result)
    sink.close()
    hud, kw = calls[0]
    maps = refined_result.semantic
    assert [names for _, names in kw["semantic"]] == [
        maps[n].class_names for n in drawn
    ]
    assert hud[-3:] == [
        f"refine {SAM} · 1/2",
        "semantic scene · excl road",
        "semantic lanes · excl lane",
    ]
    off = VideoSink(RenderConfig(semantic_alpha=0.0), DistanceConfig())
    off.open(ctx)
    off._renderer.render = lambda rgb, boxes, hud, depth, **kw: (
        calls.append((hud, kw)) or rgb
    )
    off.write(refined_result)
    off.close()
    assert calls[-1][1] == {"semantic": []}


def test_npz_sink_semantic_labels(ctx, result, refined_result):
    sink = NpzSink()
    sink.open(ctx)
    sink.write(refined_result)
    sink.write(refined_result)
    sink.close()
    data = np.load(ctx.run_dir / "depth.npz")
    assert set(data) == {
        "depth",
        "frame_index",
        "intrinsics",
        "geometry",
        "geometry_fields",
        "semantic_labels_scene",
        "semantic_classes_scene",
        "semantic_labels_lanes",
        "semantic_classes_lanes",
    }
    labels = data["semantic_labels_scene"]
    assert labels.shape == (2, 50, 50) and labels.dtype == np.uint8
    expected = resample_mask(
        refined_result.semantic["scene"].labels, GEOMETRY, (0, 0, 50, 50)
    )
    np.testing.assert_array_equal(labels[1], expected)
    assert data["semantic_classes_lanes"].tolist() == list(LANES)


def _metrics(ctx, *results):
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
    for r in results:
        sink.write(r)
    sink.close()
    return json.loads((ctx.run_dir / "metrics.json").read_text())


def test_metrics_sink_refined_and_semantic(semantic_ctx, refined_result):
    cfg = replace(
        semantic_ctx.config,
        semantic=(
            {"name": "scene", "type": "segformer"},
            {"name": "lanes", "type": "twinlitenet"},
        ),
        exclude_semantic=("scene.road", "lanes.lane"),
    )
    metrics = _metrics(
        replace(semantic_ctx, config=cfg), refined_result, refined_result
    )
    assert metrics["mask_sources"] == {SAM: 2}
    assert metrics["refined"] == {SAM: {"targets": 4, "filled": 2}}
    assert metrics["pipeline"]["exclude_semantic"] == {
        "scene": ["road"],
        "lanes": ["lane"],
    }
    assert metrics["distance"]["exclude_semantic"] == ["scene.road", "lanes.lane"]
    semantic = metrics["semantic"]
    scene, lanes = semantic["maps"]["scene"], semantic["maps"]["lanes"]
    assert scene["exclude"] == ["road"] and scene["class_frac_mean"] == {
        "road": 0.5,
        "sky": 0.5,
    }
    assert scene["excluded_frac_mean"] == pytest.approx(0.5)
    assert lanes["excluded_frac_mean"] == pytest.approx(0.3)
    # Union of the exclusions: the overlap (top-left) is counted once.
    assert semantic["excluded_frac_mean"] == pytest.approx(0.65)
    assert (
        semantic["excluded_frac_mean"]
        < scene["excluded_frac_mean"] + lanes["excluded_frac_mean"]
    )
    assert {"transformers", "timm"} <= set(metrics["versions"])


def test_refined_targets_follow_the_policy(semantic_ctx, refined_result):
    components = semantic_ctx.pipeline["components"]
    refiners = [{**components["refiners"][0], "options": {"targets": "all"}}]
    ctx = replace(
        semantic_ctx,
        pipeline={
            **semantic_ctx.pipeline,
            "components": {**components, "refiners": refiners},
        },
    )
    assert _metrics(ctx, refined_result)["refined"] == {
        SAM: {"targets": 2, "filled": 1}
    }


# task_0006: tracking and smoothing outputs.
TRACKER = {
    "type": "iou",
    "name": "iou",
    "options": {"iou_thr": 0.3, "max_age_s": 0.5},
}
SMOOTHING = {"filter": "kalman", "accel_std": 2.0}


def tracked(result, *estimates):
    """`result` with track ids 4 and 9 and the given estimates (None = no smoothing)."""
    instances = replace(result.instances, track_id=np.array([4, 9], np.int64))
    return replace(
        result, instances=instances, temporal=estimates if estimates else None
    )


@pytest.fixture
def track_ctx(ctx):
    components = {**ctx.pipeline["components"], "tracker": TRACKER}
    policy = {**ctx.pipeline["pipeline"], "smoothing": SMOOTHING}
    return replace(ctx, pipeline={"components": components, "pipeline": policy})


def _csv_rows(ctx, *results):
    sink = CsvSink()
    sink.open(ctx)
    for r in results:
        sink.write(r)
    sink.close()
    with open(ctx.run_dir / "detections.csv", newline="") as fh:
        return list(csv.DictReader(fh))


def test_csv_sink_track_columns(track_ctx, result):
    kalman = tracked(result, TrackEstimate(11.98765, 0.12345, -1.23456, 0.3, 5), None)
    ema = tracked(result, TrackEstimate(12.5, math.nan, math.nan, math.nan, 2), None)
    only_ids = tracked(result)
    rows = _csv_rows(track_ctx, kalman, ema, only_ids)
    assert [rows[0][k] for k in TRACK_COLUMNS] == [
        "4",
        "11.9877",
        "0.1235",
        "-1.2346",
        "0.3",
    ]
    assert float(rows[0]["distance_m"]) == pytest.approx(12.0)  # raw stays
    # The second instance has no distance yet: id only.
    assert [rows[1][k] for k in TRACK_COLUMNS] == ["9", "", "", "", ""]
    # ema / median: no std, no speed.
    assert [rows[2][k] for k in TRACK_COLUMNS] == ["4", "12.5", "", "", ""]
    # Tracking without smoothing.
    assert [rows[4][k] for k in TRACK_COLUMNS] == ["4", "", "", "", ""]


def _metrics(ctx, *results):
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
    for r in results:
        sink.write(r)
    sink.close()
    return json.loads((ctx.run_dir / "metrics.json").read_text())


def test_metrics_sink_tracks(track_ctx, result):
    frames = []
    for k in range(12):  # 1.1 s at 10 fps: track 4 is long, track 9 too
        est = TrackEstimate(12.0 + 0.1 * k, 0.1, 1.0, 0.1, k + 1)
        r = tracked(result, est, None)
        frames.append(replace(r, frame=replace(r.frame, index=k, t=k / 10)))
    metrics = _metrics(track_ctx, *frames)
    tracks = metrics["tracks"]
    assert (tracks["n_tracks"], tracks["short_tracks"]) == (2, 0)
    first = tracks["per_track"][0]
    assert (first["id"], first["label"], first["n_frames"]) == (4, "car", 12)
    assert first["rel_speed_mps"] == {"p5": 1.0, "median": 1.0, "p95": 1.0}
    assert tracks["per_track"][1]["distance_m"] is None  # 25 px < min_valid_px
    assert metrics["pipeline"]["smoothing"] == SMOOTHING
    assert metrics["components"]["tracker"] == TRACKER


def test_metrics_sink_without_tracker_has_no_tracks(ctx, result):
    assert "tracks" not in _metrics(ctx, result)


def _video_labels_and_hud(ctx, render, results, tmp_path):
    clip = tmp_path / "clip.mp4"
    with VideoWriter(clip, 100, 100, "10/1") as writer:
        writer.write(np.zeros((100, 100, 3), np.uint8))
    ctx = replace(ctx, source=VideoSource(SourceConfig(video=clip)))
    sink = VideoSink(render, DistanceConfig())
    sink.open(ctx)
    drawn = []
    sink._renderer.render = lambda rgb, boxes, hud, depth, **kw: (
        drawn.append(([b.label for b in boxes], hud)) or rgb
    )
    for r in results:
        sink.write(r)
    sink.close()
    return drawn


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg/ffprobe")
@pytest.mark.parametrize(
    ("render", "first"),
    [
        (RenderConfig(), "#4 car 91% · 11.5 m · -1.2 m/s"),
        (RenderConfig(show_smoothed=False), "#4 car 91% · 12.0 m · -1.2 m/s"),
        (RenderConfig(show_speed=False), "#4 car 91% · 11.5 m"),
    ],
)
def test_video_sink_track_labels_and_hud(track_ctx, result, tmp_path, render, first):
    est = TrackEstimate(11.5, 0.2, -1.25, 0.3, 5)
    later = replace(
        tracked(result, est, None),
        instances=replace(result.instances, track_id=np.array([4, 17], np.int64)),
    )
    drawn = _video_labels_and_hud(
        track_ctx, render, [tracked(result, est, None), later], tmp_path
    )
    (labels, hud), (_, later_hud) = drawn
    assert labels == [first, "#9 suv 50% · n/a"]
    assert hud[-1] == "track iou · IoU ≥ 0.3 · age 0.5 s · kalman q 2 · ids 9"
    assert later_hud[-1].endswith("· ids 17")  # tracks opened so far
