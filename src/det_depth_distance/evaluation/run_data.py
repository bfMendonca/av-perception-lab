"""A finished run loaded for evaluation: detections.csv as numpy columns and
metrics.json as a mapping.

Numeric columns are float64 with empty cells as NaN ("n/a"); text columns
are object arrays. Only the task_0001/0002 columns are required (up to
raw_label); later columns are optional, so older runs still load and their
missing estimators are left out of the report.
"""

import csv
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from det_depth_distance.camera import Calibration
from det_depth_distance.core.options import ConfigError
from det_depth_distance.core.types import CameraIntrinsics
from det_depth_distance.io.sinks import CSV_FIELDS

# Distance estimators read from the CSV, in report order: the configured
# statistic, the inlier statistics, the box distance of region "both" (task
# 0004) and the smoothed distance (task 0006).
ESTIMATORS: tuple[str, ...] = (
    "distance_m",
    "median",
    "mean",
    "p10",
    "p25",
    "p75",
    "p90",
    "min",
    "box_distance_m",
    "distance_smooth_m",
)
# Inlier statistics exist whenever some depth pixel is valid; like distance_m
# they only count as an estimate when the region had min_valid_px pixels.
INLIER_ESTIMATORS: tuple[str, ...] = (
    "median",
    "mean",
    "p10",
    "p25",
    "p75",
    "p90",
    "min",
)
REQUIRED_COLUMNS: tuple[str, ...] = CSV_FIELDS[: CSV_FIELDS.index("raw_label") + 1]
TEXT_COLUMNS: frozenset[str] = frozenset(
    {
        "class_name",
        "source",
        "raw_label",
        "sources",
        "source_scores",
        "raw_labels",
        "mask_source",
    }
)


@dataclass(frozen=True, slots=True)
class RunData:
    """detections.csv columns and metrics.json of one run folder."""

    run_dir: Path
    columns: Mapping[str, np.ndarray]  # float64 (empty -> NaN) or object (text)
    metrics: Mapping[str, Any]

    @property
    def name(self) -> str:
        return self.run_dir.name

    @property
    def video_name(self) -> str:
        """File name of the processed video."""
        return Path(self.metrics["video"]["path"]).name

    @property
    def distance_type(self) -> str:
        """ "z" or "euclidean" (config.distance.distance_type)."""
        return self.metrics["config"]["distance"]["distance_type"]

    @property
    def focal_frame_px(self) -> float | None:
        return self.metrics.get("focal_frame_px")

    @property
    def depth_kind(self) -> str | None:
        """components.depth.kind: "metric_focal", "metric" or "relative"."""
        return self.metrics["components"]["depth"].get("kind")

    @property
    def frame_intrinsics(self) -> CameraIntrinsics | None:
        """Frame-resolution K given to the depth model (None: K predicted per
        frame, or a run older than task_0007)."""
        k = self.metrics.get("frame_intrinsics")
        return CameraIntrinsics(**k) if k else None

    @property
    def is_undistorted(self) -> bool:
        """Whether the run undistorted its frames (camera.calibration.undistort)."""
        return bool((self.metrics.get("camera") or {}).get("undistort"))

    @property
    def undistort_calibration(self) -> tuple[Calibration, float] | None:
        """(calibration, alpha) the run undistorted its frames with, or None.

        Rebuilt from the K and distortion recorded in metrics.json["camera"],
        never re-read from the calibration file: a later recalibration written
        to the same path must not silently change the evaluation of old runs.

        Raises:
            ConfigError: metrics.json["camera"] lacks the recorded calibration.
        """
        if not self.is_undistorted:
            return None
        camera = self.metrics["camera"]
        try:
            calibration = Calibration.from_description(camera["calibration"])
            return calibration, float(camera["undistort"]["alpha"])
        except (ConfigError, KeyError, TypeError, ValueError) as exc:
            raise ConfigError(
                f"run {self.name}: metrics.json camera.calibration/undistort "
                f"unusable ({exc})"
            ) from None

    @property
    def sources(self) -> tuple[str, ...]:
        """Groups of instances matched separately: every detector with the concat
        aggregator; with a fusing aggregator its output is one set, named after
        the aggregator type (the CSV `source` of a fused row is only its
        representative's detector)."""
        components = self.metrics["components"]
        aggregator = components["aggregator"]["type"]
        if aggregator != "concat":
            return (aggregator,)
        return tuple(d["name"] for d in components["detectors"])

    def row_sources(self) -> np.ndarray:
        """Source group (see `sources`) of every CSV row."""
        groups = self.sources
        if len(groups) == 1:
            return np.full(len(self), groups[0], dtype=object)
        return self.columns["source"]

    @property
    def has_tracks(self) -> bool:
        ids = self.columns.get("track_id")
        return ids is not None and bool(np.isfinite(ids).any())

    def available_estimators(self) -> tuple[str, ...]:
        """ESTIMATORS present in the CSV with at least one value."""
        return tuple(
            e
            for e in ESTIMATORS
            if e in self.columns and np.isfinite(self.columns[e]).any()
        )

    def processed_frames(self) -> np.ndarray:
        """Source frame indices of every processed frame:
        round(start * fps) + stride * k for k < frames_processed (as io.video.read_frames)."""
        src = self.metrics["config"]["source"]
        first = round(src["start"] * self.metrics["video"]["fps"])
        return first + src["stride"] * np.arange(self.metrics["frames_processed"])

    def frame_times(self, frames: np.ndarray) -> np.ndarray:
        """Frame.t (s) of source frame indices."""
        src = self.metrics["config"]["source"]
        fps = self.metrics["video"]["fps"]
        return src["start"] + (frames - round(src["start"] * fps)) / fps

    def __len__(self) -> int:
        return len(self.columns["frame"])


# metrics.json keys read by RunData (written since task_0002/0006).
_METRICS_PATHS: tuple[tuple[str, ...], ...] = (
    ("video", "path"),
    ("video", "fps"),
    ("frames_processed",),
    ("components", "detectors"),
    ("components", "aggregator", "type"),
    ("components", "depth"),
    ("config", "source", "start"),
    ("config", "source", "stride"),
    ("config", "distance", "distance_type"),
)


def load_run(run_dir: Path) -> RunData:
    """Load a run folder.

    Raises:
        ConfigError: detections.csv or metrics.json missing, required columns
            missing, or metrics.json lacking the keys RunData reads.
    """
    run_dir = Path(run_dir)
    csv_path, metrics_path = run_dir / "detections.csv", run_dir / "metrics.json"
    for path in (csv_path, metrics_path):
        if not path.is_file():
            raise ConfigError(f"{run_dir}: {path.name} not found (not a run folder?)")
    metrics = json.loads(metrics_path.read_text())
    for keys in _METRICS_PATHS:
        node: Any = metrics
        for key in keys:
            if not isinstance(node, Mapping) or key not in node:
                raise ConfigError(
                    f"{metrics_path}: missing {'.'.join(keys)} (run older than task_0002?)"
                )
            node = node[key]
    with open(csv_path, newline="") as fh:
        reader = csv.DictReader(fh)
        header = reader.fieldnames or []
        missing = [c for c in REQUIRED_COLUMNS if c not in header]
        if missing:
            raise ConfigError(f"{csv_path}: missing columns {', '.join(missing)}")
        known = [c for c in CSV_FIELDS if c in header]
        cells: dict[str, list[str]] = {c: [] for c in known}
        for row in reader:
            for c in known:
                cells[c].append(row[c])
    columns = {
        c: (
            np.array(values, dtype=object)
            if c in TEXT_COLUMNS
            else np.array(
                [float(v) if v != "" else np.nan for v in values], dtype=np.float64
            )
        )
        for c, values in cells.items()
    }
    return RunData(run_dir, columns, metrics)
