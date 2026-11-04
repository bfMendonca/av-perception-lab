"""Structural contract of a finished run folder, shared by the local smoke
tests (tests/smoke/) and the GPU smoke (tests/gpu/test_smoke_gpu.py).

`check_run_outputs` derives every expectation from the run's own resolved
configuration (metrics.json["config"]): the files (depth.npz only with
output.save_depth, trajectories.jsonl only with motion), the 61 CSV columns
in CSV_FIELDS order, the base
metrics keys plus the conditional ones (agreement with >= 2 detectors,
region_comparison with region "both", semantic with segmenters, tracks with
a tracker), and the ffprobe of video.mp4 (one frame per processed frame,
frame size or side-by-side width). Distances are left to the caller.
"""

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from av_perception_lab.io.sinks import CSV_FIELDS
from av_perception_lab.io.video import VideoInfo, probe

# Keys every metrics.json has (io/sinks.py MetricsSink.close).
BASE_METRICS_KEYS: frozenset[str] = frozenset(
    {
        "video",
        "run_dir",
        "frames_processed",
        "components",
        "pipeline",
        "focal_frame_px",
        "focal_map_intrinsics",
        "focal_source",
        "frame_intrinsics",
        "camera",
        "distance",
        "timings_ms",
        "wall_time_s",
        "end_to_end_fps",
        "peak_vram_torch_allocated_gb",
        "device_mem_used_delta_gb_approx",
        "detections_total",
        "detections_by_source",
        "detections_by_n_sources",
        "boxes_total",
        "boxes_without_distance",
        "distance_regions",
        "mask_sources",
        "refined",
        "versions",
        "repo_commit",
        "host",
        "config",
    }
)
NPZ_KEYS: frozenset[str] = frozenset(
    {"depth", "frame_index", "intrinsics", "geometry", "geometry_fields"}
)


@dataclass(frozen=True, slots=True)
class RunOutputs:
    """Parsed outputs of a run folder that passed `check_run_outputs`."""

    run_dir: Path
    rows: list[dict[str, str]]  # detections.csv, cells as written
    metrics: dict[str, Any]
    video: VideoInfo  # ffprobe of video.mp4


def read_detections(run_dir: Path) -> tuple[list[str], list[dict[str, str]]]:
    """(header, rows) of `run_dir`/detections.csv."""
    with open(run_dir / "detections.csv", newline="") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)
        return list(reader.fieldnames or []), rows


def single_run_dir(out_root: Path) -> Path:
    """The one run folder a single CLI run created under `out_root`."""
    (run_dir,) = out_root.iterdir()
    return run_dir


def _conditional_keys(config: dict[str, Any]) -> set[str]:
    """metrics.json keys that the resolved configuration turns on."""
    keys = set()
    if len(config["detectors"]) >= 2 and config["agreement"]["enabled"]:
        keys.add("agreement")
    if config["distance"]["region"] == "both":
        keys.add("region_comparison")
    if config["semantic"]:
        keys.add("semantic")
    if config["tracking"] is not None:
        keys.add("tracks")
    return keys


def check_run_outputs(run_dir: Path) -> RunOutputs:
    """Assert the structural contract of `run_dir` and return its parsed outputs.

    Raises:
        AssertionError: a missing file or key, a wrong CSV header, or a video
            whose frame count or size disagrees with metrics.json.
    """
    metrics = json.loads((run_dir / "metrics.json").read_text())
    config = metrics["config"]
    save_depth = config["output"]["save_depth"]
    files = {p.name for p in run_dir.iterdir()}
    expected_files = {"video.mp4", "detections.csv", "metrics.json"}
    if save_depth:
        expected_files.add("depth.npz")
    if config["motion"] is not None:
        expected_files.add("trajectories.jsonl")
    assert files == expected_files, files

    header, rows = read_detections(run_dir)
    assert header == list(CSV_FIELDS) and len(header) == 61, header
    assert len(rows) == metrics["boxes_total"]

    missing = BASE_METRICS_KEYS - set(metrics)
    assert not missing, f"metrics.json lacks {sorted(missing)}"
    extra = set(metrics) - BASE_METRICS_KEYS
    assert extra == _conditional_keys(config), (extra, _conditional_keys(config))

    n_frames = metrics["frames_processed"]
    assert n_frames > 0
    video = probe(run_dir / "video.mp4")
    width, height = metrics["video"]["width"], metrics["video"]["height"]
    assert video.n_frames == n_frames and video.height == height
    if config["render"]["layout"] == "side-by-side":
        assert video.width > 2 * width  # frame, depth panel and its legend
    else:
        assert video.width == width

    if save_depth:
        geometry = metrics["components"]["depth"]["geometry"]
        names = [s["name"] for s in config["semantic"]]
        with np.load(run_dir / "depth.npz") as npz:
            assert set(npz.files) == NPZ_KEYS | {
                f"semantic_{part}_{name}"
                for name in names
                for part in ("labels", "classes")
            }
            assert npz["depth"].shape == (
                n_frames,
                geometry["map_h"],
                geometry["map_w"],
            )
            assert npz["frame_index"].shape == (n_frames,)
    return RunOutputs(run_dir, rows, metrics, video)
