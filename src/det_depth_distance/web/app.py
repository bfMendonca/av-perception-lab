"""FastAPI application: thin routes over `schema` and `paths` (no torch)."""

import dataclasses
import json
import shutil
import sys
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from pathlib import Path, PurePosixPath
from typing import Any

import yaml
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool
from starlette.requests import ClientDisconnect

from det_depth_distance import config
from det_depth_distance.web import compare, gpu, history, schema, uploads
from det_depth_distance.web.jobs import Job, JobQueue, valid_video_id
from det_depth_distance.web.library import (
    InvalidVideo,
    Library,
    OutsideRoots,
    VideoRecord,
)
from det_depth_distance.web.paths import WebPaths, safe_join

# Built front-end (`cd web && npm run build`); not committed, served when present.
DEFAULT_STATIC_DIR = Path(__file__).resolve().parents[3] / "web" / "dist"


class ValidateBody(BaseModel):
    config: dict[str, Any]


class ParseBody(BaseModel):
    yaml: str


class DumpBody(BaseModel):
    config: dict[str, Any]
    extras: dict[str, Any] = {}


class RunBody(BaseModel):
    config: dict[str, Any]
    video_id: str | None = None


class ImportBody(BaseModel):
    path: str


VIDEO_FILES = ("file", "preview", "thumb")


def default_worker_argv(config_path: Path) -> list[str]:
    """Command line of the worker subprocess for a config file."""
    return [sys.executable, "-m", "det_depth_distance.web.worker", str(config_path)]


def _known_keys() -> set[str]:
    """Top-level YAML keys a configuration may hold."""
    fields = {f.name for f in dataclasses.fields(config.RunConfig)}
    return fields - set(config.PIPELINE_DISTANCE_KEYS)


def create_app(
    paths: WebPaths,
    worker_argv: Callable[[Path], list[str]] = default_worker_argv,
    static_dir: Path = DEFAULT_STATIC_DIR,
) -> FastAPI:
    """Build the application: catalog, config, videos, runs, jobs and history.

    `worker_argv` builds the worker command line (tests inject a fake).
    `static_dir` is the built front-end, served at `/` (any path outside
    `/api` that is not a file falls back to its `index.html`); it is looked
    up per request, so a build made after start-up is picked up.
    """
    jobs = JobQueue(paths, worker_argv)
    library = Library(paths.library_root, paths.data_roots)

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        yield
        await jobs.aclose()  # kills a running worker
        library.close()  # and a running preview encode

    app = FastAPI(title="det-depth-distance", lifespan=lifespan)
    app.state.paths = paths
    app.state.jobs = jobs
    app.state.library = library

    def job_or_404(job_id: str) -> Job:
        try:
            return jobs.get(job_id)
        except KeyError:
            raise HTTPException(404, f"unknown job {job_id!r}") from None

    def video_or_404(video_id: str) -> VideoRecord:
        try:
            return library.get(video_id)
        except KeyError:
            raise HTTPException(404, f"unknown video {video_id!r}") from None

    def results(compute: Callable[[], Any], *run_ids: str) -> Any:
        """Run a web.compare call; unknown, invalid (traversal) and file-less
        (`job-<id>`) runs are a readable 404, never a 500."""
        try:
            return compute()
        except compare.NoRunFiles as exc:
            raise HTTPException(404, str(exc)) from None
        except (ValueError, FileNotFoundError):
            raise HTTPException(
                404, f"unknown run {' or '.join(repr(r) for r in run_ids)}"
            ) from None

    def run_folder_or_404(run_id: str) -> Path:
        try:
            return history.run_folder(paths.out_root, run_id)
        except (ValueError, FileNotFoundError):
            raise HTTPException(404, f"unknown run {run_id!r}") from None

    @app.get("/api/catalog")
    def catalog() -> dict[str, Any]:
        return schema.module_catalog()

    @app.post("/api/config/validate")
    def validate(body: ValidateBody) -> dict[str, Any]:
        return {"errors": schema.validate_config(body.config)}

    @app.post("/api/config/parse")
    def parse(body: ParseBody) -> dict[str, Any]:
        try:
            data = yaml.safe_load(body.yaml) or {}
        except yaml.YAMLError as exc:
            raise HTTPException(422, f"invalid YAML: {exc}") from None
        if not isinstance(data, dict):
            raise HTTPException(422, "the top level must be a mapping")
        known = _known_keys()
        return {
            "config": {k: v for k, v in data.items() if k in known},
            "extras": {k: v for k, v in data.items() if k not in known},
        }

    @app.post("/api/config/dump")
    def dump(body: DumpBody) -> dict[str, str]:
        merged = {**body.config, **body.extras}
        return {"yaml": yaml.safe_dump(merged, sort_keys=False, allow_unicode=True)}

    @app.get("/api/videos")
    def list_videos() -> list[dict[str, Any]]:
        return [r.to_dict() for r in library.list()]

    @app.get("/api/videos/roots")
    def video_roots() -> dict[str, Any]:
        # `cwd`: where a relative import path starts (the UI checks a path
        # against the roots before offering it; the server checks again).
        return {
            "roots": [str(r) for r in library.allowed_roots()],
            "cwd": str(Path.cwd().resolve()),
        }

    @app.post("/api/videos")
    async def upload_video(request: Request) -> dict[str, Any]:
        # Streamed to the library's disk part by part (uploads.receive_files);
        # whatever is left in the staging folder is removed in any case.
        stage = await run_in_threadpool(library.staging_dir)
        try:
            try:
                parts = await uploads.receive_files(
                    request.headers.get("content-type", ""),
                    request.stream(),
                    stage,
                    ("file", "meta"),
                )
            except uploads.UploadError as exc:
                raise HTTPException(400, str(exc)) from None
            except ClientDisconnect:
                raise HTTPException(400, "the client disconnected") from None
            if "file" not in parts:
                raise HTTPException(400, "the upload has no `file` part")
            video, meta = parts["file"], parts.get("meta")
            try:
                record = await run_in_threadpool(
                    library.add_staged,
                    stage,
                    video.path,
                    video.filename,
                    video.sha256,
                    meta.path if meta else None,
                )
            except InvalidVideo as exc:
                raise HTTPException(422, str(exc)) from None
            return record.to_dict()
        finally:
            await run_in_threadpool(shutil.rmtree, stage, True)

    @app.post("/api/videos/import")
    def import_video(body: ImportBody) -> dict[str, Any]:
        try:
            return library.add_from_path(Path(body.path)).to_dict()
        except OutsideRoots as exc:
            raise HTTPException(403, str(exc)) from None
        except FileNotFoundError as exc:
            raise HTTPException(404, str(exc)) from None
        except InvalidVideo as exc:
            raise HTTPException(422, str(exc)) from None

    @app.get("/api/videos/{video_id}")
    def get_video(video_id: str) -> dict[str, Any]:
        return video_or_404(video_id).to_dict()

    @app.get("/api/videos/{video_id}/{name}")
    def video_file(video_id: str, name: str) -> FileResponse:
        record = video_or_404(video_id)
        if name not in VIDEO_FILES:
            raise HTTPException(404, f"unknown video file {name!r}")
        files = {"file": record.original, "preview": record.preview}
        path = files.get(name, record.thumb)
        if path is None:
            raise HTTPException(404, "the preview is not ready")
        if not path.is_file():
            raise HTTPException(404, f"{name} not available")
        return FileResponse(path)

    def submit_run(config: dict[str, Any], video_id: str | None) -> Any:
        """Validate and queue a run (`POST /api/runs` and rerun): 422 with
        `errors` when the config or the video id is invalid."""
        errors: list[dict[str, str]] = []
        if video_id is not None:
            try:
                if not valid_video_id(video_id):
                    raise KeyError(video_id)
                record = library.get(video_id)
            except KeyError:
                errors.append({"path": "video_id", "message": "unknown video id"})
            else:
                if record.missing:
                    errors.append(
                        {
                            "path": "video_id",
                            "message": "the video file is missing or changed "
                            "on the server: import it again",
                        }
                    )
                # The library video is what runs (and what the history
                # records): source.video is bound to video_id.
                source = config.get("source")
                source = source if isinstance(source, dict) else {}
                config = {**config, "source": {**source, "video": str(record.original)}}
        errors += schema.validate_config(config)
        if errors:
            return JSONResponse({"errors": errors}, status_code=422)
        job = jobs.submit(config, video_id)
        return {"job_id": job.id}

    @app.post("/api/runs", response_model=None)
    async def start_run(body: RunBody) -> Any:
        return submit_run(body.config, body.video_id)

    @app.get("/api/jobs/{job_id}")
    async def get_job(job_id: str) -> dict[str, Any]:
        job = job_or_404(job_id)
        return job.to_dict(jobs.position(job))

    @app.get("/api/jobs/{job_id}/events")
    async def job_events(job_id: str) -> StreamingResponse:
        job_or_404(job_id)

        async def stream() -> AsyncIterator[str]:
            async for event in jobs.events(job_id):
                yield f"data: {json.dumps(event)}\n\n"

        return StreamingResponse(
            stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.delete("/api/jobs/{job_id}")
    async def cancel_job(job_id: str) -> dict[str, Any]:
        job = job_or_404(job_id)
        jobs.cancel(job_id)
        return job.to_dict(jobs.position(job))

    @app.get("/api/runs")
    def list_runs() -> list[dict[str, Any]]:
        return [r.to_dict() for r in history.scan_runs(paths.out_root)]

    @app.get("/api/runs/{run_id}")
    def get_run(run_id: str) -> dict[str, Any]:
        try:
            return history.load_run(paths.out_root, run_id).to_dict()
        except (ValueError, FileNotFoundError):
            raise HTTPException(404, f"unknown run {run_id!r}") from None

    # Results and comparison (web.compare). Declared before the
    # /api/runs/{run_id}/{name} file route, which would match their names.
    @app.get("/api/compare")
    def compare_runs(a: str, b: str) -> dict[str, Any]:
        return results(lambda: compare.compare_runs(paths.out_root, a, b), a, b)

    @app.get("/api/runs/{run_id}/summary")
    def run_summary(run_id: str) -> dict[str, Any]:
        summary = results(lambda: compare.run_summary(paths.out_root, run_id), run_id)
        return {"summary": summary, "metrics": list(compare.SUMMARY_METRICS)}

    @app.get("/api/runs/{run_id}/series")
    def run_series(run_id: str, track: int | None = None) -> dict[str, Any]:
        return results(
            lambda: compare.distance_series(paths.out_root, run_id, track), run_id
        )

    @app.get("/api/runs/{run_id}/artifacts")
    def run_artifacts(run_id: str) -> list[dict[str, Any]]:
        return results(lambda: compare.list_artifacts(paths.out_root, run_id), run_id)

    @app.get("/api/runs/{run_id}/files/{rel:path}")
    def run_artifact(run_id: str, rel: str) -> FileResponse:
        path = results(
            lambda: compare.artifact_path(paths.out_root, run_id, rel), run_id
        )
        return FileResponse(path, media_type=compare.media_type(path.name))

    def recorded_config(run_id: str) -> tuple[dict[str, Any], str]:
        found = results(
            lambda: compare.effective_config(paths.out_root, run_id), run_id
        )
        if found is None:
            raise HTTPException(404, f"run {run_id} recorded no configuration")
        return found

    @app.post("/api/runs/{run_id}/reopen")
    def reopen_run(run_id: str) -> dict[str, Any]:
        config, origin = recorded_config(run_id)
        known = _known_keys()
        stem = run_id if not run_id.startswith("job-") else f"{run_id}_config"
        return {
            "config": {k: v for k, v in config.items() if k in known},
            "extras": {k: v for k, v in config.items() if k not in known},
            "origin": origin,
            "file_name": f"{stem}.yaml",
        }

    @app.post("/api/runs/{run_id}/rerun", response_model=None)
    async def rerun(run_id: str) -> Any:
        config, _origin = recorded_config(run_id)
        record = results(lambda: history.load_run(paths.out_root, run_id), run_id)
        config = {k: v for k, v in config.items() if k in _known_keys()}
        if record.video_id is not None:
            return submit_run(config, record.video_id)
        # Not a library video: rerun its source.video only from an import
        # folder (the same rule as importing a server path).
        source = config.get("source")
        video = source.get("video") if isinstance(source, dict) else None
        if not isinstance(video, str) or not video:
            raise HTTPException(409, f"run {run_id} recorded no source video")
        real = (Path.cwd() / video).resolve()
        roots = library.allowed_roots()
        if not any(real == r or real.is_relative_to(r) for r in roots):
            raise HTTPException(
                409,
                f"{Path(video).name} is not in the video library nor in an import "
                "folder: import it (or start the server with --data-root on its "
                "folder) and process it from the editor",
            )
        if not real.is_file():
            raise HTTPException(
                409, f"{Path(video).name} no longer exists on the server"
            )
        return submit_run(config, None)

    @app.get("/api/runs/{run_id}/{name}")
    def run_file(run_id: str, name: str) -> FileResponse:
        folder = run_folder_or_404(run_id)
        if name not in history.RUN_FILES:
            raise HTTPException(404, f"unknown run file {name!r}")
        path = folder / history.RUN_FILES[name]
        if not path.is_file():
            raise HTTPException(404, f"{history.RUN_FILES[name]} not available")
        return FileResponse(path)

    @app.get("/api/gpu")
    def gpu_state() -> dict[str, int | None]:
        return {"free_mib": gpu.free_memory_mib()}

    @app.get("/{rel:path}", include_in_schema=False)
    def front_end(rel: str) -> FileResponse:
        if rel == "api" or rel.startswith("api/"):
            raise HTTPException(404, "not found")
        index = static_dir / "index.html"
        if not index.is_file():
            raise HTTPException(404, "front-end not built (cd web && npm run build)")
        try:
            target = safe_join(static_dir, rel) if rel else index
        except ValueError:
            raise HTTPException(404, "not found") from None
        if target.is_file():
            return FileResponse(target)
        # Client-side routes have no extension; a missing file (e.g. a stale
        # hashed asset) is a 404, never HTML served as JS/CSS.
        if rel.startswith("assets/") or PurePosixPath(rel).suffix:
            raise HTTPException(404, f"{rel} not found")
        return FileResponse(index)

    return app
