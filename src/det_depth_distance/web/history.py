"""History of runs: a read-only scan of the run folders under `out_root`.

Folders written by the CLI before the web interface (only `detections.csv`,
maybe `metrics.json`) are listed too; every file is optional. A run's
`status` comes from `web_run.json` when present, else from `metrics.json`
(written last by the CLI, so its absence marks an incomplete run).
Run ids are folder names and only ever resolved through `safe_join`.
"""

import csv
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from det_depth_distance.web.paths import safe_join

RUN_FILES = {
    "video": "video.mp4",
    "preview": "preview.jpg",
    "config": "config.yaml",
    "detections": "detections.csv",
    "metrics": "metrics.json",
}
_MARKERS = ("detections.csv", "metrics.json", "web_run.json", "video.mp4")
JOBS_DIR = (
    ".web_jobs"  # same folder as web.jobs.JOBS_DIR (history must not import jobs)
)
_STAMP = "%Y%m%d_%H%M%S"


@dataclass(frozen=True, slots=True)
class RunRecord:
    """One run folder, as listed in the history."""

    id: str
    created: str  # ISO 8601, local time
    video: str | None
    detectors: list[str] = field(default_factory=list)
    depth: str | None = None
    frames: int | None = None
    wall_s: float | None = None
    status: str = "incomplete"  # queued|running|done|failed|cancelled|incomplete
    complete: bool = False  # metrics.json written and the run finished
    has_config: bool = False
    error: str | None = None  # why a job failed (job records only)
    video_id: str | None = None  # library video (web_run.json or job record)
    log_tail: list[str] = field(default_factory=list)  # job records only

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _read_json(path: Path) -> dict[str, Any]:
    """Parsed JSON object, or {} when missing, unreadable or not an object."""
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _str_or_none(value: Any) -> str | None:
    return value if isinstance(value, str) else None


def _csv_frames(path: Path) -> int | None:
    """Distinct `frame` values of a detections.csv (None when unreadable)."""
    try:
        with open(path, newline="") as fh:
            return len({row.get("frame") for row in csv.DictReader(fh)}) or None
    except (OSError, csv.Error, UnicodeDecodeError):
        return None


def _created(folder: Path) -> str:
    try:
        return datetime.strptime(folder.name[:15], _STAMP).isoformat()
    except ValueError:
        return datetime.fromtimestamp(folder.stat().st_mtime).isoformat()


def _record(folder: Path) -> RunRecord:
    metrics = _read_json(folder / "metrics.json")
    web = _read_json(folder / "web_run.json")
    components = metrics.get("components")
    components = components if isinstance(components, dict) else {}
    detectors = [
        str(d["name"])
        for d in components.get("detectors") or []
        if isinstance(d, dict) and "name" in d
    ]
    depth = components.get("depth")
    video = (metrics.get("video") or {}).get("path") if metrics else None
    has_metrics = bool(metrics)
    status = web.get("status")
    if not isinstance(status, str):
        status = "done" if has_metrics else "incomplete"
    frames = metrics.get("frames_processed")
    if not isinstance(frames, int):
        frames = _csv_frames(folder / "detections.csv")
    wall = metrics.get("wall_time_s")
    return RunRecord(
        id=folder.name,
        created=_created(folder),
        video=Path(video).name if isinstance(video, str) else None,
        detectors=detectors,
        depth=str(depth["name"])
        if isinstance(depth, dict) and "name" in depth
        else None,
        frames=frames,
        wall_s=float(wall) if isinstance(wall, int | float) else None,
        status=status,
        complete=has_metrics and status == "done",
        # config.yaml (web runs) or the config recorded by the CLI in metrics
        has_config=(folder / "config.yaml").is_file()
        or isinstance(metrics.get("config"), dict),
        video_id=_str_or_none(web.get("video_id")),
    )


def _job_record(path: Path, out_root: Path) -> RunRecord | None:
    """Entry for a failed/cancelled job whose run folder does not exist."""
    data = _read_json(path)
    status, run_dir = data.get("status"), data.get("run_dir")
    if status not in ("failed", "cancelled") or not isinstance(data.get("id"), str):
        return None
    if isinstance(run_dir, str) and (out_root / Path(run_dir).name).is_dir():
        return None  # the folder's own record covers it
    config = data.get("config") if isinstance(data.get("config"), dict) else {}
    source = config.get("source") if isinstance(config.get("source"), dict) else {}
    video = source.get("video")
    tail = data.get("log_tail")
    return RunRecord(
        id=f"job-{data['id']}",
        created=str(data.get("created") or ""),
        video=Path(video).name if isinstance(video, str) else None,
        status=status,
        error=_str_or_none(data.get("error")),
        video_id=_str_or_none(data.get("video_id")),
        log_tail=[str(x) for x in tail] if isinstance(tail, list) else [],
        has_config=bool(config),
    )


def scan_runs(out_root: Path) -> list[RunRecord]:
    """Every run folder under `out_root` plus failed jobs without one, newest first."""
    if not out_root.is_dir():
        return []
    folders = [
        p
        for p in out_root.iterdir()
        if p.is_dir()
        and not p.name.startswith(".")
        and any((p / m).exists() for m in _MARKERS)
    ]
    records = [_record(p) for p in folders]
    jobs = out_root / JOBS_DIR
    if jobs.is_dir():
        records += [r for p in jobs.glob("*.json") if (r := _job_record(p, out_root))]
    return sorted(records, key=lambda r: r.created, reverse=True)


def run_folder(out_root: Path, run_id: str) -> Path:
    """Folder of `run_id` inside `out_root`.

    Raises:
        ValueError: `run_id` is not a plain folder name (separators, `..`,
            absolute path, or a symlink leading outside `out_root`).
        FileNotFoundError: no such run.
    """
    if not run_id or run_id in (".", "..") or "/" in run_id or "\\" in run_id:
        raise ValueError(f"invalid run id {run_id!r}")
    folder = safe_join(out_root, run_id)
    if not folder.is_dir():
        raise FileNotFoundError(run_id)
    return folder


def job_data(out_root: Path, run_id: str) -> dict[str, Any] | None:
    """The job record behind a `job-<id>` history entry (None for any other id,
    or when it lists no such entry); never resolves outside `out_root`."""
    if not run_id.startswith("job-") or not run_id[4:].isalnum():
        return None
    path = out_root / JOBS_DIR / f"{run_id[4:]}.json"
    return _read_json(path) if _job_record(path, out_root) is not None else None


def load_run(out_root: Path, run_id: str) -> RunRecord:
    """The record of one run (errors as in `run_folder`); `job-<id>` is a failed
    job that left no folder."""
    if run_id.startswith("job-"):
        if not run_id[4:].isalnum():
            raise ValueError(f"invalid run id {run_id!r}")
        record = _job_record(out_root / JOBS_DIR / f"{run_id[4:]}.json", out_root)
        if record is None:
            raise FileNotFoundError(run_id)
        return record
    return _record(run_folder(out_root, run_id))
