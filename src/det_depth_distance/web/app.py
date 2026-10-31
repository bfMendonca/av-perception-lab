"""FastAPI application: thin routes over `schema` and `paths` (no torch)."""

import dataclasses
import json
import sys
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import yaml
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel

from det_depth_distance import config
from det_depth_distance.web import gpu, history, schema
from det_depth_distance.web.jobs import Job, JobQueue, valid_video_id
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
    """Build the application: catalog, config, runs, jobs and history.

    `worker_argv` builds the worker command line (tests inject a fake).
    `static_dir` is the built front-end, served at `/` (any path outside
    `/api` that is not a file falls back to its `index.html`); it is looked
    up per request, so a build made after start-up is picked up.
    """
    jobs = JobQueue(paths, worker_argv)

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        yield
        await jobs.aclose()  # kills a running worker

    app = FastAPI(title="det-depth-distance", lifespan=lifespan)
    app.state.paths = paths
    app.state.jobs = jobs

    def job_or_404(job_id: str) -> Job:
        try:
            return jobs.get(job_id)
        except KeyError:
            raise HTTPException(404, f"unknown job {job_id!r}") from None

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

    @app.post("/api/runs", response_model=None)
    async def start_run(body: RunBody) -> Any:
        errors = schema.validate_config(body.config)
        if body.video_id is not None and not valid_video_id(body.video_id):
            errors.append({"path": "video_id", "message": "invalid video id"})
        if errors:
            return JSONResponse({"errors": errors}, status_code=422)
        job = jobs.submit(body.config, body.video_id)
        return {"job_id": job.id}

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
        return FileResponse(target if target.is_file() else index)

    return app
