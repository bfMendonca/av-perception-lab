"""Job queue: one runner, one worker subprocess at a time (one GPU).

`submit` enqueues a job; the single runner task starts the worker
(`worker_argv(config_path)`), parses its JSON lines into the job's
append-only event list and waits for it to exit. Subscribers replay the list
from the start (`events`), woken by an `asyncio.Condition`, so a late SSE
client sees the whole history. Cancel kills the worker's process group with
SIGKILL (the same lesson as the ffmpeg decoder: SIGTERM can hang on a full
pipe). A crashed worker ends the job as `failed` with the log tail. The run
folder of a job that did not finish is flagged in its `web_run.json`.
On shutdown (`aclose`) every unfinished job ends `cancelled`; a record left
`running`/`queued` by a server that died anyway is settled as interrupted when
the next queue starts. The server process never imports torch; only the
worker does.
"""

import asyncio
import contextlib
import json
import os
import re
import shutil
import signal
import tempfile
import uuid
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from det_depth_distance.web.paths import JOBS_DIR, VIDEO_ID_ENV, WebPaths

LOG_TAIL_LINES = 30
RECENT_FINISHED = 20  # finished jobs `recent()` returns besides the unfinished
SHUTDOWN_ERROR = "the server shut down"
INTERRUPTED_ERROR = "interrupted: the server stopped before the job ended"
_VIDEO_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
TERMINAL = ("done", "failed", "cancelled")


def valid_video_id(video_id: str) -> bool:
    """A library id: one plain name, never a path."""
    return bool(_VIDEO_ID.fullmatch(video_id)) and ".." not in video_id


@dataclass
class Job:
    """State of one submitted run."""

    id: str
    config: dict[str, Any]
    video_id: str | None
    status: str = "queued"  # queued|running|done|failed|cancelled
    error: str | None = None
    run_dir: str | None = None
    frames: int = 0
    log_tail: list[str] = field(default_factory=list)
    events: list[dict[str, Any]] = field(default_factory=list)
    created: str = ""
    closed: bool = False  # the final `end` event is in `events`
    cancel_requested: bool = False
    proc: asyncio.subprocess.Process | None = field(default=None, repr=False)
    cond: asyncio.Condition = field(default_factory=asyncio.Condition, repr=False)

    def to_dict(self, position: int | None = None) -> dict[str, Any]:
        """JSON view; `position` is the place in the queue (0 = next)."""
        return {
            "id": self.id,
            "status": self.status,
            "position": position,
            "error": self.error,
            "run_id": Path(self.run_dir).name if self.run_dir else None,
            "frames": self.frames,
            "log_tail": list(self.log_tail),
            "created": self.created,
            "video_id": self.video_id,
            "video": self.video_name(),
        }

    def video_name(self) -> str | None:
        """Base name of the run's source.video (None when it has none)."""
        source = self.config.get("source")
        video = source.get("video") if isinstance(source, dict) else None
        return Path(video).name if isinstance(video, str) and video else None


class JobQueue:
    """FIFO queue with a single runner."""

    def __init__(
        self, paths: WebPaths, worker_argv: Callable[[Path], list[str]]
    ) -> None:
        self.paths = paths
        self.worker_argv = worker_argv
        self._jobs: dict[str, Job] = {}
        self._waiting: list[Job] = []
        self._wake = asyncio.Event()
        self._runner: asyncio.Task[None] | None = None
        self._settle_leftovers()

    # -- public API ---------------------------------------------------------

    def submit(self, config: dict[str, Any], video_id: str | None = None) -> Job:
        """Enqueue a run (call from the running event loop).

        The run folder is forced under `paths.out_root`, where history looks.

        Raises:
            ValueError: `video_id` is not a plain library id.
        """
        if video_id is not None and not valid_video_id(video_id):
            raise ValueError(f"invalid video id {video_id!r}")
        config = json.loads(json.dumps(config))  # private deep copy
        output = config.get("output")
        job_id = uuid.uuid4().hex[:12]
        # A unique tag makes the run folder (`..._<tag>`) recognisable as this
        # job's among the folders other processes write into the same out_root.
        config["output"] = {
            **(output if isinstance(output, dict) else {}),
            "out_root": str(self.paths.out_root),
            "tag": f"web-{job_id}",
        }
        job = Job(job_id, config, video_id)
        job.created = datetime.now().astimezone().isoformat(timespec="seconds")
        job.events.append({"event": "queued"})
        self._jobs[job.id] = job
        self._persist(job)
        self._waiting.append(job)
        self._ensure_runner()
        self._wake.set()
        return job

    def get(self, job_id: str) -> Job:
        """The job, or KeyError."""
        return self._jobs[job_id]

    def recent(self) -> list[Job]:
        """Unfinished jobs plus the `RECENT_FINISHED` latest finished ones,
        newest first."""
        jobs = list(self._jobs.values())[::-1]
        finished = [j for j in jobs if j.status in TERMINAL][:RECENT_FINISHED]
        keep = {id(j) for j in finished}
        return [j for j in jobs if j.status not in TERMINAL or id(j) in keep]

    def active(self) -> bool:
        """A run is queued or running (Install packages waits for it)."""
        return any(j.status not in TERMINAL for j in self._jobs.values())

    def position(self, job: Job) -> int | None:
        """Place in the waiting line (0 = starts next); None once started."""
        return self._waiting.index(job) if job in self._waiting else None

    def cancel(self, job_id: str) -> None:
        """Cancel a queued or running job (no-op when it already ended)."""
        job = self._jobs[job_id]
        if job.status in TERMINAL:
            return
        job.cancel_requested = True
        job.status = "cancelled"
        if job in self._waiting:
            self._waiting.remove(job)
            self._persist(job)
            asyncio.get_running_loop().create_task(self._close(job))
        else:
            self._persist(job)
            if job.proc is not None:
                _kill(job.proc)

    async def events(
        self, job_id: str, start: int = 0
    ) -> AsyncIterator[tuple[int, dict[str, Any]]]:
        """`(index, event)` of every event of the job from `start` (0 = the
        first); ends after its `end` event."""
        job = self._jobs[job_id]
        seen = max(0, min(start, len(job.events)))
        while True:
            async with job.cond:
                await job.cond.wait_for(
                    lambda n=seen: len(job.events) > n or job.closed
                )
                batch = job.events[seen:]
                done = job.closed
            for i, event in enumerate(batch, seen):
                yield i, event
            seen += len(batch)
            if done and seen >= len(job.events):
                return

    async def aclose(self) -> None:
        """Server shutdown: stop the runner (which kills the running worker's
        process group) and end every unfinished job as `cancelled`, so no
        record or run folder stays `running`/`queued`."""
        if self._runner is not None:
            self._runner.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._runner
            self._runner = None
        self._waiting.clear()
        for job in list(self._jobs.values()):
            if job.proc is not None and job.proc.returncode is None:
                _kill(job.proc)
            if job.closed:
                continue
            if job.status not in TERMINAL:
                job.cancel_requested = True
                job.status, job.error = "cancelled", job.error or SHUTDOWN_ERROR
            self._flag_run(job)
            await self._close(job)

    # -- internals ----------------------------------------------------------

    def _settle_leftovers(self) -> None:
        """Records a previous server left `running`/`queued` (killed before
        `aclose` ran): a running job is `failed`, a queued one `cancelled`,
        both with `INTERRUPTED_ERROR`, and so is the run folder's status."""
        folder = self.paths.out_root / JOBS_DIR
        if not folder.is_dir():
            return
        for path in folder.glob("*.json"):
            try:
                record = json.loads(path.read_text())
            except (OSError, ValueError):
                continue
            if not isinstance(record, dict):
                continue
            status = record.get("status")
            if status not in ("running", "queued"):
                continue
            record["status"] = "failed" if status == "running" else "cancelled"
            record["error"] = record.get("error") or INTERRUPTED_ERROR
            with contextlib.suppress(OSError):
                tmp = path.with_suffix(".tmp")
                tmp.write_text(json.dumps(record))
                os.replace(tmp, path)
            run_dir = record.get("run_dir")
            if isinstance(run_dir, str) and run_dir:
                _set_run_status(Path(run_dir), record["status"], only_unfinished=True)

    def _ensure_runner(self) -> None:
        if self._runner is None or self._runner.done():
            self._runner = asyncio.get_running_loop().create_task(self._run_forever())

    async def _emit(self, job: Job, event: dict[str, Any]) -> None:
        async with job.cond:
            job.events.append(event)
            job.cond.notify_all()

    async def _close(self, job: Job) -> None:
        """Append the final event and wake every subscriber."""
        async with job.cond:
            job.events.append(
                {"event": "end", "status": job.status, "error": job.error}
            )
            job.closed = True
            self._persist(job)
            job.cond.notify_all()

    async def _run_forever(self) -> None:
        while True:
            while not self._waiting:
                self._wake.clear()
                await self._wake.wait()
            job = self._waiting.pop(0)
            try:
                await self._execute(job)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - a bug here must not stall the queue
                if job.status not in TERMINAL:
                    job.status, job.error = "failed", f"{type(exc).__name__}: {exc}"
                self._flag_run(job)
                await self._close(job)

    async def _execute(self, job: Job) -> None:
        tmp = Path(tempfile.mkdtemp(prefix="ddd-job-"))
        proc = None
        try:
            path = tmp / "config.yaml"
            path.write_text(yaml.safe_dump(job.config, sort_keys=False))
            env = {**os.environ, VIDEO_ID_ENV: job.video_id or ""}
            proc = await asyncio.create_subprocess_exec(
                *self.worker_argv(path),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,  # log only: never mixed with events
                env=env,
                limit=1 << 20,
                start_new_session=True,  # own process group: kill takes the ffmpeg too
            )
            job.proc = proc
            if job.cancel_requested:  # cancelled while the worker was starting
                _kill(proc)
            else:
                job.status = "running"
                self._persist(job)
                await self._emit(job, {"event": "running"})
            done_event, _ = await asyncio.gather(
                self._read(job, proc), self._drain_stderr(job, proc)
            )
            code = await proc.wait()
            if job.cancel_requested:
                job.status = "cancelled"
            elif code == 0 and done_event:
                job.status = "done"
            else:
                job.status = "failed"
                job.error = job.error or (
                    f"worker exited with code {code}"
                    + (f": {job.log_tail[-1]}" if job.log_tail else "")
                )
            self._flag_run(job)
            await self._close(job)
        finally:
            if proc is not None and proc.returncode is None:
                _kill(proc)
                with contextlib.suppress(Exception):
                    await asyncio.shield(proc.wait())
            job.proc = None
            shutil.rmtree(tmp, ignore_errors=True)

    async def _drain_stderr(self, job: Job, proc: asyncio.subprocess.Process) -> None:
        """Feed the log tail from stderr; `\\r` (progress bars) also ends a line."""
        assert proc.stderr is not None
        pending = ""
        while chunk := await proc.stderr.read(65536):
            pending += chunk.decode(errors="replace")
            *lines, pending = re.split(r"[\r\n]", pending)
            pending = pending[-4096:]
            self._log(job, lines)
        self._log(job, [pending])

    @staticmethod
    def _log(job: Job, lines: list[str]) -> None:
        job.log_tail.extend(line.rstrip() for line in lines if line.strip())
        del job.log_tail[:-LOG_TAIL_LINES]

    async def _read(self, job: Job, proc: asyncio.subprocess.Process) -> bool:
        """Consume the worker output; True when a `done` event was seen."""
        assert proc.stdout is not None
        saw_done = False
        async for raw in proc.stdout:
            line = raw.decode(errors="replace").rstrip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except ValueError:
                event = None
            if not isinstance(event, dict) or "event" not in event:
                self._log(job, [line])
                continue
            kind = event["event"]
            if kind == "started":
                job.run_dir = event.get("run_dir")
                self._persist(job)
            elif kind == "frame":
                job.frames = int(event.get("n", job.frames))
            elif kind == "error":
                job.error = str(event.get("message", ""))
            elif kind == "done":
                saw_done = True
                job.run_dir = event.get("run_dir") or job.run_dir
            await self._emit(job, event)
        return saw_done

    def _persist(self, job: Job) -> None:
        """Rewrite the job's record (history lists failed jobs without a folder)."""
        record = {
            "id": job.id,
            "status": job.status,
            "created": job.created,
            "video_id": job.video_id,
            "error": job.error,
            "log_tail": job.log_tail,
            "run_dir": job.run_dir,
            "config": job.config,
        }
        folder = self.paths.out_root / JOBS_DIR
        with contextlib.suppress(OSError):
            folder.mkdir(parents=True, exist_ok=True)
            tmp = folder / f"{job.id}.tmp"
            tmp.write_text(json.dumps(record))
            os.replace(tmp, folder / f"{job.id}.json")

    def _flag_run(self, job: Job) -> None:
        """Mark the job's run folder `failed`/`cancelled` in its web_run.json."""
        if job.status not in ("failed", "cancelled") or not job.run_dir:
            return
        _set_run_status(Path(job.run_dir), job.status)


def _set_run_status(run_dir: Path, status: str, only_unfinished: bool = False) -> None:
    """Write `status` into the run folder's web_run.json (with
    `only_unfinished`, only over a `running`/`queued` one)."""
    meta = run_dir / "web_run.json"
    with contextlib.suppress(OSError, ValueError):
        data = json.loads(meta.read_text()) if meta.is_file() else {}
        if not isinstance(data, dict):
            return
        if only_unfinished and data.get("status") not in ("running", "queued"):
            return
        data["status"] = status
        meta.write_text(json.dumps(data))


def _kill(proc: asyncio.subprocess.Process) -> None:
    """SIGKILL the worker and its children (process group)."""
    with contextlib.suppress(ProcessLookupError, PermissionError):
        os.killpg(proc.pid, signal.SIGKILL)
    with contextlib.suppress(ProcessLookupError):
        proc.kill()
