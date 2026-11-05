"""FastAPI application: thin routes over `schema` and `paths` (no torch).

Request guards (`web.security`): only the allowed `Host` names are answered,
state-changing requests from another origin are refused, JSON bodies are
capped, and served files are never sniffed nor run as pages of the app.

Model readiness (`web.models`): `/api/models/status` checks the models of a
config on disk, `/api/models/prepare` queues their download (one at a time,
independent from the GPU job queue; with `packages: true` it installs a
model's per-model extra after a clean dry-run, never the base stack), and a
run whose models are not all ready is refused with 409 before it is queued.

Config presets (`web.presets`): `/api/presets` lists the built-in setups of
`configs/` and the user's own (`--presets-root`), reads one as `{config,
extras}`, saves (PUT) and deletes user presets; `source` and `output` are
never stored.
"""

import asyncio
import dataclasses
import json
import shutil
import sys
from collections.abc import AsyncIterator, Callable, Collection, Sequence
from contextlib import asynccontextmanager
from pathlib import Path, PurePosixPath
from typing import Any, Literal

import yaml
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, ConfigDict
from starlette.concurrency import run_in_threadpool
from starlette.middleware.trustedhost import TrustedHostMiddleware
from starlette.requests import ClientDisconnect

from av_perception_lab import config
from av_perception_lab.core import requirements
from av_perception_lab.core.options import ConfigError
from av_perception_lab.web import (
    compare,
    gpu,
    history,
    models,
    presets,
    schema,
    security,
    uploads,
)
from av_perception_lab.web.jobs import Job, JobQueue, valid_video_id
from av_perception_lab.web.library import (
    InvalidVideo,
    Library,
    OutsideRoots,
    VideoRecord,
    video_media_type,
)
from av_perception_lab.web.paths import WebPaths, safe_join

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


# Model bodies refuse unknown fields: a client cannot add a URL or a path to
# a download (sources and targets come from the adapters' code).
class SaveBody(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    config: dict[str, Any]
    extras: dict[str, Any] = {}
    overwrite: bool = False


class StatusBody(BaseModel):
    model_config = ConfigDict(extra="forbid")
    config: dict[str, Any]


class ModelKey(BaseModel):
    model_config = ConfigDict(extra="forbid")
    stage: str
    index: int | None = None


class PrepareBody(BaseModel):
    model_config = ConfigDict(extra="forbid")
    config: dict[str, Any]
    models: list[ModelKey] | Literal["all"]
    # Install the named models' per-model extras instead of their weights
    # (after a fresh, clean dry-run; never the base stack).
    packages: bool = False


VIDEO_FILES = ("file", "preview", "thumb")
UPLOAD_PATH = "/api/videos"
MAX_META_BYTES = 64 << 20  # the `.meta.json` part of an upload (read whole)
MAX_CONCURRENT_UPLOADS = 2
DISK_MARGIN_BYTES = 256 << 20  # free space kept on the library disk
MAX_YAML_CHARS = 1 << 20
MAX_YAML_NODES = 100_000  # after alias expansion (a config has a few hundred)
VIDEO_OUTSIDE_ROOTS = (
    "only videos of the library (send video_id) or of an import folder "
    "(--data-root) can be processed"
)


def mcap_topic_problems(
    config: dict[str, Any], mcap: dict[str, Any]
) -> list[dict[str, str]]:
    """Early, UI-friendly errors of a run on an MCAP library entry (`mcap` is
    its summary): `source.topic` must be one of its image topics, and
    `rectify.enabled` / `camera.calibration.from_source` need that topic's
    CameraInfo (rectify also needs its distortion). `load_config` and
    `build_camera` stay the authority; these name the topic's state."""
    images = {t["name"]: t for t in mcap.get("images") or []}
    choices = ", ".join(images) or "none"
    source = config.get("source")
    topic = source.get("topic") if isinstance(source, dict) else None
    if not isinstance(topic, str) or not topic:
        return [
            {
                "path": "source.topic",
                "message": f"choose an image topic of the MCAP ({choices})",
            }
        ]
    entry = images.get(topic)
    if entry is None:
        return [
            {
                "path": "source.topic",
                "message": f"{topic} is not an image topic of the MCAP "
                f"(image topics: {choices})",
            }
        ]
    if not entry.get("supported", True):
        return [
            {
                "path": "source.topic",
                "message": f"{topic} cannot be read: {entry.get('unsupported_reason')}",
            }
        ]
    section = config.get("rectify")
    rectify = isinstance(section, dict) and section.get("enabled") is True
    camera = config.get("camera")
    calibration = camera.get("calibration") if isinstance(camera, dict) else None
    from_source = (
        isinstance(calibration, dict) and calibration.get("from_source") is True
    )
    errors = []
    for wanted, path, use in (
        (rectify, "rectify.enabled", "rectification"),
        (from_source, "camera.calibration.from_source", "the intrinsics"),
    ):
        if not wanted:
            continue
        if not entry.get("has_camera_info"):
            message = f"{topic} has no CameraInfo: {use} needs one"
        elif entry.get("camera_info_error"):
            message = (
                f"the CameraInfo of {topic} ({entry['camera_info_topic']}) cannot "
                f"be used: {entry['camera_info_error']}"
            )
        elif path == "rectify.enabled" and not entry.get("has_distortion"):
            message = (
                f"the CameraInfo of {topic} has no distortion (already "
                "rectified): nothing to rectify"
            )
        else:
            continue
        errors.append({"path": path, "message": message})
    return errors


def yaml_expanded_nodes(text: str) -> int:
    """Nodes of a YAML document once its aliases are expanded (each alias
    counts its whole target, as the JSON reply repeats it); anchors and
    aliases stay usable (a hand-written config may share a value with one).

    Raises:
        yaml.YAMLError: not YAML.
        ValueError: a recursive alias (cannot be sent as JSON).
    """
    memo: dict[int, int] = {}
    active: set[int] = set()

    def size(node: yaml.Node) -> int:
        key = id(node)
        if key in memo:
            return memo[key]
        if key in active:
            raise ValueError("a YAML alias refers to itself")
        active.add(key)
        total = 1
        if isinstance(node, yaml.SequenceNode):
            total += sum(size(child) for child in node.value)
        elif isinstance(node, yaml.MappingNode):
            total += sum(size(k) + size(v) for k, v in node.value)
        active.discard(key)
        memo[key] = total
        return total

    root = yaml.compose(text, Loader=yaml.SafeLoader)
    return 0 if root is None else size(root)


def _split_config(data: dict[str, Any]) -> dict[str, dict]:
    """`{config, extras}`: the known top-level keys and the rest."""
    known = _known_keys()
    return {
        "config": {k: v for k, v in data.items() if k in known},
        "extras": {k: v for k, v in data.items() if k not in known},
    }


def _free_bytes(path: Path) -> int:
    """Free space of the disk `path` (or its nearest existing parent) is on."""
    for folder in (path, *path.resolve().parents):
        if folder.exists():
            return shutil.disk_usage(folder).free
    return 0


def _file(path: Path, media_type: str) -> FileResponse:
    """A served file with `security.file_headers`."""
    return FileResponse(
        path,
        media_type=media_type,
        headers=security.file_headers(media_type, path.name),
    )


def default_worker_argv(config_path: Path) -> list[str]:
    """Command line of the worker subprocess for a config file."""
    return [sys.executable, "-m", "av_perception_lab.web.worker", str(config_path)]


def _known_keys() -> set[str]:
    """Top-level YAML keys a configuration may hold."""
    fields = {f.name for f in dataclasses.fields(config.RunConfig)}
    return fields - set(config.PIPELINE_DISTANCE_KEYS)


def create_app(
    paths: WebPaths,
    worker_argv: Callable[[Path], list[str]] = default_worker_argv,
    static_dir: Path = DEFAULT_STATIC_DIR,
    allowed_hosts: Sequence[str] = security.LOOPBACK_HOSTS,
    dev_origins: Collection[str] = (),
    builtin_presets_dir: Path = presets.BUILTIN_DIR,
) -> FastAPI:
    """Build the application: catalog, config, videos, runs, jobs and history.

    `worker_argv` builds the worker command line (tests inject a fake).
    `static_dir` is the built front-end, served at `/` (any path outside
    `/api` that is not a file falls back to its `index.html`); it is looked
    up per request, so a build made after start-up is picked up.
    `allowed_hosts` are the `Host` names answered (`security.allowed_hosts`);
    `dev_origins` are other origins whose state-changing requests pass (the
    Vite dev server).
    `builtin_presets_dir` holds the read-only built-in presets (`configs/`).
    The weights root is `core.requirements.weights_root()` at creation (the
    server sets it from `--weights-root`); it may not overlap the library or
    out_root (ValueError).
    """
    root = requirements.weights_root()
    problem = security.weights_root_problem(root, paths)
    if problem is not None:
        raise ValueError(problem)
    jobs = JobQueue(paths, worker_argv, weights_root=root)
    prepares = models.PrepareQueue(root)
    library = Library(paths.library_root, paths.data_roots)
    upload_slots = asyncio.Semaphore(MAX_CONCURRENT_UPLOADS)

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        yield
        await prepares.aclose()  # stops a running download between chunks
        await jobs.aclose()  # kills a running worker
        library.close()  # and a running preview encode

    app = FastAPI(title="av-perception-lab", lifespan=lifespan)
    # The last added runs first: Host, then Origin, then the body cap.
    app.add_middleware(security.BodyLimit, exempt=(UPLOAD_PATH,))
    app.add_middleware(security.OriginGuard, dev_origins=dev_origins)
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=list(allowed_hosts))
    app.state.paths = paths
    app.state.jobs = jobs
    app.state.library = library
    app.state.prepares = prepares

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

    def titled(record: history.RunRecord) -> dict[str, Any]:
        """A history entry, titled by its library video's name when it has
        one (uploads are stored as `original.<ext>`)."""
        data = record.to_dict()
        if record.video_id is not None:
            try:
                data["video"] = library.get(record.video_id).name
            except KeyError:
                pass
        return data

    def job_view(job: Job) -> dict[str, Any]:
        data = job.to_dict(jobs.position(job))
        if job.video_id is not None:
            try:
                data["video"] = library.get(job.video_id).name
            except KeyError:
                pass
        return data

    def video_problem(video: str) -> tuple[Path | None, str | None]:
        """(resolved path, None) when a bare `source.video` may be processed:
        it resolves inside an import folder or is a library original; else
        (None, why). The reason never tells whether a path outside exists."""
        real = security.resolve(video)
        originals = {r.original.resolve() for r in library.list()}
        if not (security.within(real, library.allowed_roots()) or real in originals):
            return None, VIDEO_OUTSIDE_ROOTS
        if not real.is_file():
            return None, f"{Path(video).name} does not exist on the server"
        return real, None

    def prepare_or_404(job_id: str) -> models.PrepareJob:
        try:
            return prepares.get(job_id)
        except KeyError:
            raise HTTPException(404, f"unknown prepare job {job_id!r}") from None

    async def model_statuses(cfg: dict[str, Any]) -> list[dict[str, Any]]:
        """Status rows of a valid config (disk checks in the thread pool); the
        model a Prepare is working on reads `preparing`."""
        rows = models.config_rows(cfg)
        imports = models.config_imports(rows)
        statuses = await run_in_threadpool(
            lambda: [models.status_row(row, root, imports) for row in rows]
        )
        busy = prepares.preparing()
        for row, status in zip(rows, statuses, strict=True):
            if row.req is not None and row.req in busy:
                status["state"] = "preparing"
                status["preparable"] = False
        return statuses

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
        if len(body.yaml) > MAX_YAML_CHARS:
            raise HTTPException(413, f"YAML longer than {MAX_YAML_CHARS >> 10} KiB")
        try:
            nodes = yaml_expanded_nodes(body.yaml)
            if nodes > MAX_YAML_NODES:
                raise HTTPException(
                    422,
                    f"the YAML expands to more than {MAX_YAML_NODES} values "
                    "(aliases repeating large anchors?)",
                )
            data = yaml.safe_load(body.yaml) or {}
        except yaml.YAMLError as exc:
            raise HTTPException(422, f"invalid YAML: {exc}") from None
        except (ValueError, RecursionError) as exc:
            raise HTTPException(422, f"unsupported YAML: {exc}") from None
        if not isinstance(data, dict):
            raise HTTPException(422, "the top level must be a mapping")
        return _split_config(data)

    @app.post("/api/config/dump")
    def dump(body: DumpBody) -> dict[str, str]:
        merged = {**body.config, **body.extras}
        return {"yaml": yaml.safe_dump(merged, sort_keys=False, allow_unicode=True)}

    def _preset_dict(info: presets.PresetInfo) -> dict[str, str]:
        return dataclasses.asdict(info)

    @app.get("/api/presets")
    def list_presets() -> list[dict[str, str]]:
        return [
            _preset_dict(i)
            for i in presets.list_presets(builtin_presets_dir, paths.presets_root)
        ]

    @app.get("/api/presets/{preset_id}")
    def get_preset(preset_id: str) -> dict[str, dict]:
        try:
            data = presets.read_preset(
                preset_id, builtin_presets_dir, paths.presets_root
            )
        except presets.PresetNotFound as exc:
            raise HTTPException(404, str(exc)) from None
        except presets.PresetInvalid as exc:
            raise HTTPException(422, str(exc)) from None
        return _split_config(data)

    @app.put("/api/presets")
    def save_preset(body: SaveBody) -> dict[str, str]:
        if paths.presets_root is None:
            raise HTTPException(404, "saving presets is disabled (no --presets-root)")
        data = {
            k: v
            for k, v in {**body.config, **body.extras}.items()
            if k not in presets.EXCLUDED_KEYS
        }
        try:
            text = yaml.safe_dump(data, sort_keys=False, allow_unicode=True)
        except yaml.YAMLError as exc:  # RepresenterError: not plain data
            raise HTTPException(
                422, f"the preset cannot be written as YAML: {exc}"
            ) from None
        if len(text) > MAX_YAML_CHARS:
            raise HTTPException(413, f"YAML longer than {MAX_YAML_CHARS >> 10} KiB")
        known = _known_keys()
        errors = schema.validate_config({k: v for k, v in data.items() if k in known})
        if errors:
            raise HTTPException(422, errors)
        try:
            info = presets.save_preset(
                paths.presets_root,
                builtin_presets_dir,
                body.name,
                data,
                body.overwrite,
            )
        except (
            presets.PresetExists
        ) as exc:  # 409 = overwritable; a built-in name is a 422
            raise HTTPException(409, str(exc)) from None
        except presets.PresetInvalid as exc:
            raise HTTPException(422, str(exc)) from None
        except yaml.YAMLError as exc:
            raise HTTPException(
                422, f"the preset cannot be written as YAML: {exc}"
            ) from None
        return _preset_dict(info)

    @app.delete("/api/presets/{preset_id}", status_code=204)
    def delete_preset(preset_id: str) -> Response:
        try:
            presets.delete_preset(preset_id, paths.presets_root)
        except presets.PresetNotFound as exc:
            raise HTTPException(404, str(exc)) from None
        except presets.PresetReadOnly as exc:
            raise HTTPException(403, str(exc)) from None
        return Response(status_code=204)

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

    @app.post(UPLOAD_PATH)
    async def upload_video(request: Request) -> dict[str, Any]:
        # Too large (413), no room on the library disk (507), too many at once
        # (429): refused before anything is read.
        try:
            declared = int(request.headers.get("content-length", "0"))
        except ValueError:
            declared = 0
        limit = uploads.MAX_UPLOAD_BYTES
        if declared > limit + (1 << 20):  # + the multipart framing
            raise HTTPException(413, f"upload larger than {limit >> 20} MiB")
        free = await run_in_threadpool(_free_bytes, paths.library_root)
        if free < declared + DISK_MARGIN_BYTES:
            raise HTTPException(507, "not enough free space on the library disk")
        if upload_slots.locked():
            raise HTTPException(
                429, "too many uploads at once: wait for the others to finish"
            )
        async with upload_slots:
            return await receive_upload(request, limit)

    async def receive_upload(request: Request, limit: int) -> dict[str, Any]:
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
                    max_bytes=limit,
                    field_limits={"meta": MAX_META_BYTES},
                )
            except uploads.UploadError as exc:
                raise HTTPException(exc.status, str(exc)) from None
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
                raise HTTPException(422, models.client_text(str(exc), root)) from None
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
            raise HTTPException(422, models.client_text(str(exc), root)) from None

    @app.get("/api/videos/{video_id}")
    def get_video(video_id: str) -> dict[str, Any]:
        return video_or_404(video_id).to_dict()

    # Declared before the /api/videos/{video_id}/{name} file route, which
    # would match it.
    @app.get("/api/videos/{video_id}/topics")
    def video_topics(video_id: str) -> dict[str, Any]:
        record = video_or_404(video_id)
        if record.mcap is None:
            raise HTTPException(404, f"{record.name} is not an MCAP (no topics)")
        return record.mcap

    @app.get("/api/videos/{video_id}/{name}")
    def video_file(video_id: str, name: str) -> FileResponse:
        record = video_or_404(video_id)
        if name not in VIDEO_FILES:
            raise HTTPException(404, f"unknown video file {name!r}")
        if name == "file":
            # A missing/changed original, or an import whose folder is no
            # longer an import root (or now resolves elsewhere), is not served.
            if record.missing:
                raise HTTPException(404, "the original is missing or changed")
            if record.origin == "import" and not security.within(
                record.original.resolve(), library.allowed_roots()
            ):
                raise HTTPException(404, "the original is outside the import folders")
            return _file(record.original, video_media_type(record.original))
        if name == "preview":
            if record.preview is None:
                raise HTTPException(404, "the preview is not ready")
            if not record.preview.is_file():
                raise HTTPException(404, "preview not available")
            return _file(record.preview, "video/mp4")
        if not record.thumb.is_file():
            raise HTTPException(404, "thumb not available")
        return _file(record.thumb, "image/jpeg")

    async def submit_run(config: dict[str, Any], video_id: str | None) -> Any:
        """Validate and queue a run (`POST /api/runs` and rerun): 422 with
        `errors` when the config or the video id is invalid, 409 with
        `models_not_ready` when a model of the config is not ready."""
        errors: list[dict[str, str]] = []
        source = config.get("source")
        for key in ("video", "mcap"):
            value = source.get(key) if isinstance(source, dict) else None
            if video_id is None and isinstance(value, str) and value:
                # A client path reaches ffmpeg or the MCAP reader: only the
                # library and import folders.
                _real, problem = video_problem(value)
                if problem is not None:
                    errors.append({"path": f"source.{key}", "message": problem})
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
                # The library entry is what runs (and what the history
                # records): source.video, or source.mcap for an MCAP, is
                # bound to video_id.
                source = config.get("source")
                source = dict(source) if isinstance(source, dict) else {}
                if record.kind == "mcap":
                    source.pop("video", None)
                    source["mcap"] = str(record.original)
                    errors += mcap_topic_problems(config, record.mcap or {})
                else:
                    source.pop("mcap", None)
                    source["video"] = str(record.original)
                config = {**config, "source": source}
        problems = schema.validate_config(config)
        if not problems:  # paths are only discoverable in a valid config
            problems = security.path_problems(config, paths)
        errors += problems
        if errors:
            return JSONResponse({"errors": errors}, status_code=422)
        try:
            pending = models.not_ready(await model_statuses(config))
        except ConfigError as exc:
            return JSONResponse({"errors": [schema.split_error(exc)]}, 422)
        if pending:
            return JSONResponse({"models_not_ready": pending}, status_code=409)
        if prepares.installing():  # after the awaits, right before queueing
            return JSONResponse(
                {
                    "detail": "Python packages are being installed on the server: "
                    "submit the run when the install ends (and the server was "
                    "restarted)",
                    "packages_installing": True,
                },
                status_code=409,
            )
        job = jobs.submit(config, video_id)
        return {"job_id": job.id}

    @app.post("/api/runs", response_model=None)
    async def start_run(body: RunBody) -> Any:
        return await submit_run(body.config, body.video_id)

    @app.get("/api/jobs")
    def list_jobs() -> list[dict[str, Any]]:
        # This server's unfinished jobs and its latest finished ones (a
        # reloaded page re-subscribes to them).
        return [job_view(job) for job in jobs.recent()]

    @app.get("/api/jobs/{job_id}")
    async def get_job(job_id: str) -> dict[str, Any]:
        return job_view(job_or_404(job_id))

    @app.get("/api/jobs/{job_id}/events")
    async def job_events(job_id: str, request: Request) -> StreamingResponse:
        job_or_404(job_id)
        # An EventSource reconnect sends the id of the last event it got:
        # resume after it instead of replaying (and rewinding) everything.
        try:
            start = int(request.headers.get("last-event-id", "")) + 1
        except ValueError:
            start = 0

        async def stream() -> AsyncIterator[str]:
            async for index, event in jobs.events(job_id, start):
                yield f"id: {index}\ndata: {json.dumps(event)}\n\n"

        return StreamingResponse(
            stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.delete("/api/jobs/{job_id}")
    async def cancel_job(job_id: str) -> dict[str, Any]:
        job = job_or_404(job_id)
        jobs.cancel(job_id)
        return job_view(job)

    @app.get("/api/runs")
    def list_runs() -> list[dict[str, Any]]:
        return [titled(r) for r in history.scan_runs(paths.out_root)]

    @app.get("/api/runs/{run_id}")
    def get_run(run_id: str) -> dict[str, Any]:
        try:
            return titled(history.load_run(paths.out_root, run_id))
        except (ValueError, FileNotFoundError):
            raise HTTPException(404, f"unknown run {run_id!r}") from None

    @app.get("/api/runs/{run_id}/size")
    def run_size(run_id: str) -> dict[str, int]:
        try:
            return {"bytes": history.run_size(paths.out_root, run_id)}
        except (ValueError, FileNotFoundError):
            raise HTTPException(404, f"unknown run {run_id!r}") from None

    @app.delete("/api/runs/{run_id}", status_code=204)
    def delete_run(run_id: str) -> Response:
        try:
            folder = history.run_folder(paths.out_root, run_id)
        except FileNotFoundError:
            folder = None  # a `job-<id>` record has no folder
        except ValueError:
            raise HTTPException(404, f"unknown run {run_id!r}") from None
        if folder is not None and jobs.holds_run(folder):
            raise HTTPException(409, "run is still queued or running")
        try:
            history.delete_run(paths.out_root, run_id)
        except (ValueError, FileNotFoundError):
            raise HTTPException(404, f"unknown run {run_id!r}") from None
        except OSError as exc:
            raise HTTPException(500, f"could not delete: {exc}") from None
        return Response(status_code=204)

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
        return _file(path, compare.media_type(path.name))

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
            return await submit_run(config, record.video_id)
        # Not a library entry: rerun its source.video (or source.mcap) only
        # from an import folder (the same rule as importing a server path).
        source = config.get("source")
        source = source if isinstance(source, dict) else {}
        video = source.get("video") or source.get("mcap")
        if not isinstance(video, str) or not video:
            raise HTTPException(409, f"run {run_id} recorded no source video")
        _real, problem = video_problem(video)
        if problem == VIDEO_OUTSIDE_ROOTS:
            raise HTTPException(
                409,
                f"{Path(video).name} is not in the video library nor in an import "
                "folder: import it (or start the server with --data-root on its "
                "folder) and process it from the editor",
            )
        if problem is not None:
            raise HTTPException(409, problem)
        return await submit_run(config, None)

    @app.post("/api/models/status", response_model=None)
    async def models_status(body: StatusBody) -> Any:
        errors = schema.validate_config(body.config)
        if errors:
            return JSONResponse({"errors": errors}, status_code=422)
        try:
            return models.report(
                await model_statuses(body.config), prepares.restart_required
            )
        except ConfigError as exc:
            return JSONResponse({"errors": [schema.split_error(exc)]}, 422)

    @app.post("/api/models/prepare", response_model=None)
    async def start_prepare(body: PrepareBody) -> Any:
        # The same checks as a run: a valid config whose paths pass.
        errors = schema.validate_config(body.config)
        if not errors:
            errors = security.path_problems(body.config, paths)
        if errors:
            return JSONResponse({"errors": errors}, status_code=422)
        keys: list[models.Key] | Literal["all"] = (
            "all" if body.models == "all" else [(k.stage, k.index) for k in body.models]
        )
        plans = None
        if body.packages:
            if keys == "all":
                raise HTTPException(
                    422, "Install packages names its models (not 'all')"
                )
            try:
                plans = await run_in_threadpool(
                    prepares.install_plans, keys, body.config
                )
            except models.UnknownModel as exc:
                raise HTTPException(422, exc.args[0]) from None
            except ConfigError as exc:
                raise HTTPException(422, models.client_text(str(exc), root)) from None
            # Checked after the await, right before queueing (one loop thread).
            if jobs.active():
                raise HTTPException(
                    409,
                    "a run is queued or running: install packages when it ends "
                    "(a running worker must not see a half-installed package)",
                )
        try:
            job = prepares.submit(
                keys, body.config, packages=body.packages, plans=plans
            )
        except models.UnknownModel as exc:
            raise HTTPException(422, exc.args[0]) from None
        # ConfigError, UnknownSource, UserProvidedOnly, NotInstallable,
        # nothing selected
        except ValueError as exc:
            raise HTTPException(422, models.client_text(str(exc), root)) from None
        return job.to_dict(prepares.position(job))

    @app.get("/api/models/prepare")
    async def list_prepares() -> list[dict[str, Any]]:
        return [job.to_dict(prepares.position(job)) for job in prepares.recent()]

    @app.get("/api/models/prepare/{job_id}")
    async def get_prepare(job_id: str) -> dict[str, Any]:
        job = prepare_or_404(job_id)
        return job.to_dict(prepares.position(job))

    @app.get("/api/models/prepare/{job_id}/events")
    async def prepare_events(job_id: str, request: Request) -> StreamingResponse:
        prepare_or_404(job_id)
        try:
            start = int(request.headers.get("last-event-id", "")) + 1
        except ValueError:
            start = 0

        async def stream() -> AsyncIterator[str]:
            async for index, event in prepares.events(job_id, start):
                yield f"id: {index}\ndata: {json.dumps(event)}\n\n"

        return StreamingResponse(
            stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.delete("/api/models/prepare/{job_id}")
    async def cancel_prepare(job_id: str) -> dict[str, Any]:
        job = prepare_or_404(job_id)
        prepares.cancel(job_id)
        return job.to_dict(prepares.position(job))

    @app.get("/api/runs/{run_id}/{name}")
    def run_file(run_id: str, name: str) -> FileResponse:
        folder = run_folder_or_404(run_id)
        if name not in history.RUN_FILES:
            raise HTTPException(404, f"unknown run file {name!r}")
        path = folder / history.RUN_FILES[name]
        if not path.is_file():
            raise HTTPException(404, f"{history.RUN_FILES[name]} not available")
        return _file(path, compare.media_type(path.name))

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
