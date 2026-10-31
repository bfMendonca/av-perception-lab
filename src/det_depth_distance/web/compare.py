"""Results of runs for the web: summary, distance series, config diff,
comparison and the artifacts of a run folder (read-only, no torch).

Run ids are only ever resolved through `history.run_folder` (no `..`,
absolute paths or symlinks out of `out_root`); a `job-<id>` history entry
(a failed job that left no folder) raises `NoRunFiles`. Every input is
optional: folders written before the web interface may lack
`metrics.json`, `config.yaml`, `track_id` or the smoothed columns, and the
values that need them come back as None (with a note).

detections.csv is read with `evaluation.run_data.read_columns`, only the
columns below (`CSV_COLUMNS`):

- `frame`: source frame index (consecutive pairs, video time);
- `t`: Frame.t in seconds (x of the series);
- `track_id`: tracker id (jitter is per track; absent before task_0006);
- `distance_m`: the configured distance statistic (raw);
- `distance_smooth_m`: the smoothed distance (task_0006; optional);
- `class_name`: canonical label of the track (shown in the track list).

`jitter_m` is the median over tracks of each track's median absolute change
of `distance_m` between consecutive processed frames (`frame` step equal to
`source.stride`, or the smallest step in the CSV when no config is recorded):
gaps where the track was lost are not frame-to-frame changes and are
skipped. `jitter_smooth_m` is the same on `distance_smooth_m`.
"""

import contextlib
import functools
import math
import mimetypes
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import quote

import numpy as np
import yaml

from det_depth_distance.core.options import ConfigError
from det_depth_distance.evaluation.run_data import read_columns
from det_depth_distance.web import history
from det_depth_distance.web.paths import safe_join

CSV_COLUMNS = (
    "frame",
    "t",
    "track_id",
    "distance_m",
    "distance_smooth_m",
    "class_name",
)
_REQUIRED = ("frame", "t", "distance_m")
MAX_POINTS = 1500  # per series: enough for a chart a few thousand px wide

# The summary values, how to show them and which direction is an improvement
# (`better`: "lower" | "higher" | None). The front-end renders this list as is.
SUMMARY_METRICS: tuple[dict[str, Any], ...] = (
    {
        "key": "median_distance_m",
        "label": "Median distance",
        "unit": "m",
        "better": None,
        "help": "Median of distance_m over every box with a distance.",
    },
    {
        "key": "jitter_m",
        "label": "Jitter",
        "unit": "m",
        "better": "lower",
        "help": "Median over tracks of the frame-to-frame |Δ distance_m|.",
    },
    {
        "key": "jitter_smooth_m",
        "label": "Smoothed jitter",
        "unit": "m",
        "better": "lower",
        "help": "The same on distance_smooth_m.",
    },
    {
        "key": "fps",
        "label": "Throughput",
        "unit": "frames/s",
        "better": "higher",
        "help": "End-to-end processed frames per second (metrics.json).",
    },
    {"key": "frames", "label": "Frames", "unit": None, "better": None},
    {"key": "boxes", "label": "Boxes", "unit": None, "better": None},
    {"key": "tracks", "label": "Tracks", "unit": None, "better": None},
)

# Media types mimetypes does not know (or knows differently per platform).
_MEDIA_TYPES = {
    ".jsonl": "application/x-ndjson",
    ".yaml": "application/yaml",
    ".yml": "application/yaml",
    ".csv": "text/csv",
    ".mkv": "video/x-matroska",
    ".mp4": "video/mp4",
    ".json": "application/json",
}
_ROLES = {name: role for role, name in history.RUN_FILES.items()}
# Config paths every web job gets from the server (JobQueue.submit: the run
# folder and its unique tag): they always differ, so a comparison flags them.
SERVER_SET_PATHS = frozenset({"output.out_root", "output.tag"})


class NoRunFiles(FileNotFoundError):
    """The history entry exists but has no run folder (a failed job)."""


# -- run folder, config ---------------------------------------------------------


def run_dir(out_root: Path, run_id: str) -> Path:
    """The folder of `run_id`.

    Raises:
        ValueError: not a plain folder name (see `history.run_folder`).
        NoRunFiles: a `job-<id>` entry (failed before its folder existed).
        FileNotFoundError: no such run.
    """
    try:
        return history.run_folder(out_root, run_id)
    except FileNotFoundError:
        if history.job_data(out_root, run_id) is not None:
            raise NoRunFiles(
                f"run {run_id} has no files: the job ended before its run "
                "folder was created"
            ) from None
        raise


def _read_json(path: Path) -> dict[str, Any]:
    """The same tolerant reader as the history ({} when unusable)."""
    return history._read_json(path)


def effective_config(out_root: Path, run_id: str) -> tuple[dict[str, Any], str] | None:
    """(configuration the run used, where it came from) or None.

    `config.yaml` (written by the web worker), else the `config` the CLI
    records in metrics.json, else the config a failed job was submitted with.

    Raises:
        ValueError, FileNotFoundError: as `history.load_run`.
    """
    job = history.job_data(out_root, run_id)
    if job is not None:
        config = job.get("config")
        return (config, "job") if isinstance(config, dict) and config else None
    folder = history.run_folder(out_root, run_id)
    with contextlib.suppress(OSError, yaml.YAMLError):
        path = folder / "config.yaml"
        if path.is_file():
            data = yaml.safe_load(path.read_text())
            if isinstance(data, dict):
                return data, "config.yaml"
    config = _read_json(folder / "metrics.json").get("config")
    return (config, "metrics.json") if isinstance(config, dict) else None


def _path_key(prefix: str, key: Any) -> str:
    return f"{prefix}.{key}" if prefix else str(key)


def config_diff(a: Any, b: Any) -> list[dict[str, Any]]:
    """Leaf differences between two configurations, in `a`'s key order then
    `b`'s new keys: `{"path", "a", "b", "change"}` with dotted paths and lists
    by index (`detectors[0].threshold`, the validation error format).

    `change` is "changed", "added" (only in `b`; `a` is None) or "removed"
    (only in `a`). An added or removed subtree is one entry holding it whole;
    a mapping replaced by a scalar (or a list) is one "changed" leaf.
    """
    out: list[dict[str, Any]] = []

    def walk(x: Any, y: Any, path: str) -> None:
        if isinstance(x, Mapping) and isinstance(y, Mapping):
            for key in [*x, *(k for k in y if k not in x)]:
                sub = _path_key(path, key)
                if key not in y:
                    out.append(
                        {"path": sub, "a": x[key], "b": None, "change": "removed"}
                    )
                elif key not in x:
                    out.append({"path": sub, "a": None, "b": y[key], "change": "added"})
                else:
                    walk(x[key], y[key], sub)
        elif isinstance(x, list) and isinstance(y, list):
            for i in range(max(len(x), len(y))):
                sub = f"{path}[{i}]"
                if i >= len(y):
                    out.append({"path": sub, "a": x[i], "b": None, "change": "removed"})
                elif i >= len(x):
                    out.append({"path": sub, "a": None, "b": y[i], "change": "added"})
                else:
                    walk(x[i], y[i], sub)
        elif x != y or type(x) is not type(y) and not _same_number(x, y):
            out.append({"path": path, "a": x, "b": y, "change": "changed"})

    walk(a, b, "")
    return out


def _same_number(x: Any, y: Any) -> bool:
    """2 and 2.0 are the same YAML value; True and 1 are not."""
    numbers = (int, float)
    return (
        isinstance(x, numbers)
        and isinstance(y, numbers)
        and not isinstance(x, bool)
        and not isinstance(y, bool)
        and x == y
    )


# -- CSV ----------------------------------------------------------------------


@functools.lru_cache(maxsize=16)
def _columns(path: str, _mtime_ns: int, _size: int) -> dict[str, np.ndarray]:
    """The CSV_COLUMNS of a detections.csv; cached per file version (a CSV
    still being written by a running job changes its mtime/size)."""
    return read_columns(Path(path), required=_REQUIRED, only=CSV_COLUMNS)


def _load_columns(folder: Path) -> tuple[dict[str, np.ndarray] | None, str | None]:
    """(columns, None) or (None, why they are unavailable)."""
    path = folder / "detections.csv"
    try:
        stat = path.stat()
    except OSError:
        return None, "no detections.csv"
    try:
        return _columns(str(path), stat.st_mtime_ns, stat.st_size), None
    except (ConfigError, OSError, ValueError, UnicodeDecodeError) as exc:
        return None, f"detections.csv unreadable ({exc})"


def _num(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value) if math.isfinite(value) else None


def _get(data: Mapping[str, Any], *keys: str) -> Any:
    node: Any = data
    for key in keys:
        if not isinstance(node, Mapping):
            return None
        node = node.get(key)
    return node


def _source_cfg(folder: Path, metrics: Mapping[str, Any]) -> Mapping[str, Any]:
    """`source` section of the run's configuration ({} when not recorded)."""
    source = _get(metrics, "config", "source")
    if isinstance(source, Mapping):
        return source
    with contextlib.suppress(OSError, yaml.YAMLError):
        path = folder / "config.yaml"
        if path.is_file():
            data = yaml.safe_load(path.read_text())
            if isinstance(data, Mapping) and isinstance(data.get("source"), Mapping):
                return data["source"]
    return {}


def _frame_step(frames: np.ndarray, source: Mapping[str, Any]) -> float | None:
    stride = _num(source.get("stride"))
    if stride and stride >= 1:
        return stride
    steps = np.diff(np.unique(frames[np.isfinite(frames)]))
    return float(steps.min()) if steps.size else None


def _track_ids(cols: Mapping[str, np.ndarray]) -> np.ndarray | None:
    ids = cols.get("track_id")
    if ids is None or not np.isfinite(ids).any():
        return None
    return ids


def _jitter(
    frames: np.ndarray, ids: np.ndarray, values: np.ndarray, step: float | None
) -> float | None:
    if step is None:
        return None
    per_track = []
    for tid in np.unique(ids[np.isfinite(ids)]):
        sel = np.flatnonzero(ids == tid)
        sel = sel[np.argsort(frames[sel], kind="stable")]
        f, v = frames[sel], values[sel]
        consecutive = (np.diff(f) == step) & np.isfinite(v[1:]) & np.isfinite(v[:-1])
        if consecutive.any():
            per_track.append(float(np.median(np.abs(np.diff(v))[consecutive])))
    return float(np.median(per_track)) if per_track else None


# -- summary ------------------------------------------------------------------


def run_summary(out_root: Path, run_id: str) -> dict[str, Any]:
    """`{"median_distance_m", "jitter_m", "jitter_smooth_m", "fps", "frames",
    "boxes", "tracks", "notes"}` of a run (None where unknown; `notes` says
    why). See the module docstring for the columns and the jitter.

    Raises:
        ValueError, FileNotFoundError, NoRunFiles: as `run_dir`.
    """
    folder = run_dir(out_root, run_id)
    metrics = _read_json(folder / "metrics.json")
    cols, problem = _load_columns(folder)
    notes = [problem] if problem else []
    if not metrics:
        notes.append("no metrics.json: throughput unknown")
    summary: dict[str, Any] = {
        "median_distance_m": None,
        "jitter_m": None,
        "jitter_smooth_m": None,
        "fps": _num(metrics.get("end_to_end_fps")),
        "frames": None,
        "boxes": None,
        "tracks": None,
    }
    frames_processed = metrics.get("frames_processed")
    if isinstance(frames_processed, int) and not isinstance(frames_processed, bool):
        summary["frames"] = frames_processed
    wall = _num(metrics.get("wall_time_s"))
    if summary["fps"] is None and summary["frames"] and wall:
        summary["fps"] = summary["frames"] / wall
    if cols is not None:
        frames, dist = cols["frame"], cols["distance_m"]
        summary["boxes"] = len(frames)
        if summary["frames"] is None:
            summary["frames"] = int(np.unique(frames[np.isfinite(frames)]).size)
        finite = dist[np.isfinite(dist)]
        if finite.size:
            summary["median_distance_m"] = float(np.median(finite))
        ids = _track_ids(cols)
        if ids is None:
            notes.append("no track_id column: jitter needs tracks")
        else:
            summary["tracks"] = int(np.unique(ids[np.isfinite(ids)]).size)
            step = _frame_step(frames, _source_cfg(folder, metrics))
            summary["jitter_m"] = _jitter(frames, ids, dist, step)
            smooth = cols.get("distance_smooth_m")
            if smooth is not None and np.isfinite(smooth).any():
                summary["jitter_smooth_m"] = _jitter(frames, ids, smooth, step)
            else:
                notes.append("no smoothed distances (distance_smooth_m)")
    summary["notes"] = notes
    return summary


def compare_runs(out_root: Path, a: str, b: str) -> dict[str, Any]:
    """Summaries of `a` and `b`, deltas `b - a` of the metrics both have and
    the configuration diff (None when a run recorded no configuration; each
    entry flags `server_set` for SERVER_SET_PATHS).

    Raises:
        ValueError, FileNotFoundError, NoRunFiles: as `run_dir`, for either.
    """
    summary_a, summary_b = run_summary(out_root, a), run_summary(out_root, b)
    deltas = {}
    for metric in SUMMARY_METRICS:
        key = metric["key"]
        va, vb = _num(summary_a.get(key)), _num(summary_b.get(key))
        if va is None or vb is None:
            continue
        deltas[key] = {"abs": vb - va, "rel": (vb - va) / abs(va) if va else None}
    config_a, config_b = effective_config(out_root, a), effective_config(out_root, b)
    return {
        "a": a,
        "b": b,
        "metrics": list(SUMMARY_METRICS),
        "summary_a": summary_a,
        "summary_b": summary_b,
        "deltas": deltas,
        "config_diff": [
            {**d, "server_set": d["path"] in SERVER_SET_PATHS}
            for d in config_diff(config_a[0], config_b[0])
        ]
        if config_a and config_b
        else None,
        "config_origin": {
            "a": config_a[1] if config_a else None,
            "b": config_b[1] if config_b else None,
        },
    }


# -- distance series --------------------------------------------------------


def _decimate(values: np.ndarray, max_points: int) -> np.ndarray:
    """Indices keeping, per bucket, the first point and the min/max of
    `values`: the shape (peaks included) survives at ≤ max_points."""
    n = len(values)
    if n <= max_points:
        return np.arange(n)
    buckets = max(1, max_points // 3)
    keep: list[int] = []
    for chunk in np.array_split(np.arange(n), buckets):
        keep.append(int(chunk[0]))
        v = values[chunk]
        if np.isfinite(v).any():
            keep += [int(chunk[np.nanargmin(v)]), int(chunk[np.nanargmax(v)])]
    return np.unique(np.array(keep))


def _floats(values: np.ndarray) -> list[float | None]:
    return [float(v) if math.isfinite(v) else None for v in values.tolist()]


def _video_times(
    frames: np.ndarray, t: np.ndarray, source: Mapping[str, Any], fps: float | None
) -> np.ndarray:
    """Time of each row in the annotated video.mp4, which holds the processed
    frames back to back at fps/stride: source frame f is at
    (f - round(start·fps)) / fps. Without the video fps, t - start (or t from
    the first row) is the best guess."""
    start = _num(source.get("start"))
    if fps:
        return (frames - round((start or 0.0) * fps)) / fps
    origin = start if start is not None else (float(np.nanmin(t)) if t.size else 0.0)
    return t - origin


def distance_series(
    out_root: Path, run_id: str, track_id: int | None, max_points: int = MAX_POINTS
) -> dict[str, Any]:
    """Distance over time of one track (None: the longest track; every row
    when the run has no tracks).

    `{"track", "found", "tracks": [{"id", "rows", "t0", "t1", "class_name"}],
    "t", "raw", "smoothed" (None without distance_smooth_m), "video_t",
    "points" (rows before decimation), "downsampled"}`; NaN cells are null.
    A track the run does not have answers `found: False` and empty lists.

    Raises:
        ValueError, FileNotFoundError, NoRunFiles: as `run_dir`.
    """
    folder = run_dir(out_root, run_id)
    metrics = _read_json(folder / "metrics.json")
    cols, problem = _load_columns(folder)
    empty: dict[str, Any] = {
        "track": track_id,
        "found": False,
        "tracks": [],
        "t": [],
        "raw": [],
        "smoothed": None,
        "video_t": [],
        "points": 0,
        "downsampled": False,
        "note": problem,
    }
    if cols is None:
        return empty
    frames, t = cols["frame"], cols["t"]
    ids = _track_ids(cols)
    labels = cols.get("class_name")
    tracks = []
    if ids is not None:
        uniq, counts = np.unique(ids[np.isfinite(ids)], return_counts=True)
        for tid, count in sorted(zip(uniq, counts, strict=True), key=lambda p: -p[1]):
            sel = ids == tid
            tracks.append(
                {
                    "id": int(tid),
                    "rows": int(count),
                    "t0": float(np.nanmin(t[sel])),
                    "t1": float(np.nanmax(t[sel])),
                    "class_name": str(labels[sel][0]) if labels is not None else None,
                }
            )
        if track_id is None:
            track_id = tracks[0]["id"]
        sel = np.flatnonzero(ids == track_id)
    else:
        sel = np.arange(len(frames)) if track_id is None else np.array([], dtype=int)
    result = {**empty, "track": track_id, "tracks": tracks}
    if not sel.size:
        return result
    sel = sel[np.argsort(t[sel], kind="stable")]
    raw = cols["distance_m"][sel]
    keep = _decimate(raw, max_points)
    smooth = cols.get("distance_smooth_m")
    has_smooth = smooth is not None and np.isfinite(smooth).any()
    fps = _num(_get(metrics, "video", "fps"))
    video_t = _video_times(frames[sel], t[sel], _source_cfg(folder, metrics), fps)
    return {
        **result,
        "found": True,
        "t": _floats(t[sel][keep]),
        "raw": _floats(raw[keep]),
        "smoothed": _floats(smooth[sel][keep]) if has_smooth else None,
        "video_t": _floats(video_t[keep]),
        "points": int(sel.size),
        "downsampled": bool(keep.size < sel.size),
    }


# -- artifacts ------------------------------------------------------------------


def media_type(name: str) -> str:
    """Media type of a run file (application/octet-stream when unknown)."""
    suffix = Path(name).suffix.lower()
    if suffix in _MEDIA_TYPES:
        return _MEDIA_TYPES[suffix]
    return mimetypes.guess_type(name)[0] or "application/octet-stream"


def _visible(rel: Path) -> bool:
    return (
        not any(part.startswith(".") for part in rel.parts) and ".tmp." not in rel.name
    )


def list_artifacts(out_root: Path, run_id: str) -> list[dict[str, Any]]:
    """Every file of the run folder (sub-folders included, hidden and
    temporary files left out), sorted by path: `{"name", "size",
    "media_type", "modified", "role", "url"}`. `role` names the files the
    UI knows (`history.RUN_FILES`), None for anything else.

    Raises:
        ValueError, FileNotFoundError, NoRunFiles: as `run_dir`.
    """
    folder = run_dir(out_root, run_id)
    items = []
    for path in sorted(folder.rglob("*")):
        rel = path.relative_to(folder)
        if path.is_symlink() or not path.is_file() or not _visible(rel):
            continue
        stat = path.stat()
        name = rel.as_posix()
        items.append(
            {
                "name": name,
                "size": stat.st_size,
                "media_type": media_type(name),
                "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(
                    timespec="seconds"
                ),
                "role": _ROLES.get(name),
                "url": f"/api/runs/{quote(run_id, safe='')}/files/{quote(name)}",
            }
        )
    return items


def artifact_path(out_root: Path, run_id: str, rel: str) -> Path:
    """The file `rel` of the run folder, as listed by `list_artifacts`.

    Raises:
        ValueError: `rel` resolves outside the folder or is hidden.
        FileNotFoundError, NoRunFiles: no such run or file.
    """
    folder = run_dir(out_root, run_id)
    path = safe_join(folder, rel)
    if (
        path == folder
        or not _visible(path.relative_to(folder))
        or not _visible(Path(rel))
    ):
        raise ValueError(f"{rel!r} is not a run artifact")
    if not path.is_file():
        raise FileNotFoundError(rel)
    return path
