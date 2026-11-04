"""History of runs: a read-only scan of the run folders under `out_root`.

Folders written by the CLI before the web interface (only `detections.csv`,
maybe `metrics.json`) are listed too; every file is optional. A run's
`status` comes from `web_run.json` when present, else from `metrics.json`
(written last by the CLI, so its absence marks an incomplete run).
Run ids are folder names and only ever resolved through `safe_join`.
"""

import csv
import json
import os
import shutil
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from av_perception_lab.web.paths import JOBS_DIR, safe_join

RUN_FILES = {
    "video": "video.mp4",
    "preview": "preview.jpg",
    "config": "config.yaml",
    "detections": "detections.csv",
    "metrics": "metrics.json",
}
_MARKERS = ("detections.csv", "metrics.json", "web_run.json", "video.mp4")
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


def _is_run(folder: Path) -> bool:
    """Whether `folder` holds a run (any marker file): what `scan_runs` lists."""
    return any((folder / m).exists() for m in _MARKERS)


def _listed_folder(out_root: Path, run_id: str) -> Path:
    """`run_folder`, but only for folders `scan_runs` lists."""
    folder = run_folder(out_root, run_id)
    if not _is_run(folder):
        raise FileNotFoundError(run_id)
    return folder


def scan_runs(out_root: Path) -> list[RunRecord]:
    """Every run folder under `out_root` plus failed jobs without one, newest first."""
    if not out_root.is_dir():
        return []
    folders = [
        p
        for p in out_root.iterdir()
        if p.is_dir() and not p.name.startswith(".") and _is_run(p)
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
    # A leading dot: hidden folders (the job records, staging) are no runs.
    if not run_id or run_id.startswith(".") or "/" in run_id or "\\" in run_id:
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


def _job_path(out_root: Path, run_id: str) -> Path:
    """Record file of a listed `job-<id>` entry (errors as in `load_run`)."""
    if not run_id[4:].isalnum():
        raise ValueError(f"invalid run id {run_id!r}")
    path = out_root / JOBS_DIR / f"{run_id[4:]}.json"
    if _job_record(path, out_root) is None:
        raise FileNotFoundError(run_id)
    return path


def run_size(out_root: Path, run_id: str) -> int:
    """Bytes a run uses: regular files only, symlinks neither followed nor
    counted; for `job-<id>` the size of its record file (errors as in
    `load_run`)."""
    if run_id.startswith("job-"):
        return _job_path(out_root, run_id).stat().st_size
    total = 0
    for root, _dirs, files in os.walk(_listed_folder(out_root, run_id)):
        for name in files:
            path = Path(root) / name
            if not path.is_symlink():
                try:
                    total += path.stat().st_size
                except OSError:
                    pass  # vanished while walking
    return total


def delete_run(out_root: Path, run_id: str) -> None:
    """Permanently remove a run folder (symlinks inside are unlinked, never
    followed) and the job records pointing at it or, for `job-<id>`, its record
    (errors as in `load_run`; a folder `scan_runs` does not list is not found)."""
    if run_id.startswith("job-"):
        _job_path(out_root, run_id).unlink()
        return
    folder = _listed_folder(out_root, run_id)
    shutil.rmtree(folder)
    # A failed/cancelled job record would resurface as a `job-<id>` entry.
    for record in (out_root / JOBS_DIR).glob("*.json"):
        run_dir = _read_json(record).get("run_dir")
        if isinstance(run_dir, str) and Path(run_dir).name == folder.name:
            record.unlink(missing_ok=True)
