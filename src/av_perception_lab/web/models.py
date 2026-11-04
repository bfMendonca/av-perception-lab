"""Model readiness for the web server: status of a config's models and the
prepare queue (no torch, no model loaded).

`config_rows` lists the models a configuration would load exactly like
`--prepare` does (`cli.config_requirements`: every component, plus the focal
model of `camera.estimate`), keyed by `(stage, index)` — `detectors[1]` is
`("detectors", 1)`, `depth` is `("depth", None)`. `status_row` checks one of
them on disk (`core.requirements.check`). Clients name models only by these
keys: what is downloaded, from where and to where comes from the adapters'
code, never from the request.

`PrepareQueue` runs one Prepare at a time, independent from the GPU
`JobQueue`. Its runner awaits `core.provision.prepare` in a worker thread
(downloads are I/O; the event loop stays free); progress callbacks arrive on
that thread (or on Hugging Face hub threads) and are marshalled to the loop
with `call_soon_threadsafe`. Events are an append-only list per job, replayed
to SSE subscribers like `web.jobs`. Cancel sets the job's own flag and the
engine's `threading.Event`; the next job starts only once the worker thread
has returned (sources check the event between chunks), so two Prepares never
write the same files. Nothing is persisted: finished jobs are kept in memory
(the latest `RECENT_FINISHED`) and results are logged.

A row whose packages are missing carries the packages plan
(`core.provision.packages.plan_packages`): the base stack's one-time command
(never run by the server), a per-model extra installable after a clean
dry-run, or the state `conflict` with the resolver's reason. A job submitted
with `packages=True` installs per-model extras (`install_packages`: fresh
dry-run, then `uv sync` naming every installed extra) and streams the `uv`
output as `output` events; afterwards `restart_required` is set. Runs use the
new packages at once (each run starts a fresh worker process, and the status
checks re-scan the environment); a server restart is only needed when a
package the server process itself imported was upgraded.

What the web may prepare: a row is `preparable` when it is missing and a
Prepare can fix it. It is not when its missing files come from a source the
adapter does not vouch for (`reason="unknown_source"`: `Asset.known` /
`CodeRepo.known` False, fail closed), when every missing file can only be
provided by the user (`reason="user_provided"`: `Asset.downloadable` False,
the code layer ready; the message carries the manual-download hint) or when
only packages are missing (`reason="packages_only"`). "Prepare all" leaves
the first two out and lists them in the job's `skipped`. The queue runs the
engine with `allow_unknown=False`, so an unknown source is never fetched even
when its file disappears after the job was queued.

Text sent to clients names the weights root `<weights root>`, never its
absolute path, and is scrubbed of credentials.
"""

import asyncio
import json
import logging
import re
import threading
import time
import uuid
from collections.abc import AsyncIterator, Iterable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from av_perception_lab import cli, config
from av_perception_lab.core.provision import (
    install_packages,
    prepare,
    with_package_plan,
)
from av_perception_lab.core.provision import packages as pkgs
from av_perception_lab.core.provision.engine import UNKNOWN_SOURCE
from av_perception_lab.core.requirements import (
    CODE_DIR,
    Asset,
    CodeRepo,
    ModelStatus,
    Requirements,
    check,
    redact_root,
)

log = logging.getLogger(__name__)

ROOT_LABEL = "<weights root>"
RECENT_FINISHED = 20
PROGRESS_INTERVAL_S = 0.2  # at most one progress event per item and interval
TERMINAL = ("done", "failed", "cancelled")
SHUTDOWN_ERROR = "the server shut down"
INSTALLED_NOTE = (
    "packages installed: new runs use them (each run starts a fresh worker "
    "process); restart the server only if it should use an upgraded package "
    "it had already imported"
)
OUTPUT_LIMIT = 2000  # `output` events kept per job
_INDEXED = re.compile(r"^(?P<stage>.+)\[(?P<index>\d+)\]$")

Key = tuple[str, int | None]


class UnknownModel(KeyError):
    """A model key that is not one of the config's models."""


class NotInstallable(ValueError):
    """A packages job for a model the server must not install packages for
    (the base stack, no per-model extra)."""


class UnknownSource(ValueError):
    """A model whose missing files come from a source its adapter does not
    vouch for (`Asset.known` False, e.g. a Hugging Face repo typed in the
    config): the web server never downloads it."""


class UserProvidedOnly(ValueError):
    """A model whose only missing files are ones the user must provide
    (`Asset.downloadable` False): a Prepare cannot fetch them."""


@dataclass(frozen=True)
class ModelRow:
    """One model of a config: where it is (`stage`, `index`), its adapter
    `type` and `name`, and its requirements (`error` instead when the
    declaration is unsafe)."""

    stage: str
    index: int | None
    type: str
    name: str
    req: Requirements | None
    error: str | None = None

    @property
    def key(self) -> Key:
        return (self.stage, self.index)

    def head(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "index": self.index,
            "type": self.type,
            "name": self.name,
        }


def split_path(path: str) -> Key:
    """`detectors[1]` -> ("detectors", 1); `depth` -> ("depth", None)."""
    match = _INDEXED.match(path)
    if match is None:
        return path, None
    return match["stage"], int(match["index"])


def config_rows(data: dict[str, Any]) -> list[ModelRow]:
    """Models a valid configuration would load (nothing is loaded).

    Raises:
        ConfigError: the configuration is invalid.
    """
    cfg = config.config_from_mapping(json.loads(json.dumps(data)))
    rows = []
    for path, name, type_name, req in cli.config_requirements(cfg):
        stage, index = split_path(path)
        if isinstance(req, str):
            rows.append(ModelRow(stage, index, type_name, name, None, req))
        else:
            rows.append(ModelRow(stage, index, type_name, name, req))
    return rows


def redact(text: str | None, root: Path) -> str | None:
    """`core.requirements.redact_root` with this module's label."""
    return redact_root(text, root, ROOT_LABEL)


def client_text(text: str, root: Path) -> str:
    """Text for a client: credentials scrubbed, the weights root as
    `<weights root>`, the project folder as `.` and the home as `~`."""
    project = pkgs.default_pyproject().parent
    return pkgs.relative(redact(text, root) or "", project)


def _ready(item: Asset | CodeRepo, root: Path) -> bool:
    try:
        return item.present(root).state == "ready"
    except (OSError, RuntimeError, ValueError):
        return False


def unknown_sources(req: Requirements, root: Path) -> list[Asset | CodeRepo]:
    """Downloadable assets and code repos of `req` that are not ready and
    whose source the adapter does not vouch for (`known` False). A kind that
    is never downloaded (`UserProvided`) is not a source: see
    `user_provided_only`."""
    items: list[Asset | CodeRepo] = [
        a for a in req.assets if type(a).downloadable and not a.known
    ]
    items += [c for c in req.code if not c.known]
    return [item for item in items if not _ready(item, root)]


def user_provided_only(req: Requirements, status: ModelStatus, root: Path) -> bool:
    """True when what a Prepare could fetch is all there (code layer ready,
    every asset that is not ready is non-downloadable) and at least one
    user-provided file is missing: only the user can make the model ready."""
    if status.layers.get("code", "ready") != "ready":
        return False
    pending = [a for a in req.assets if not _ready(a, root)]
    return bool(pending) and not any(type(a).downloadable for a in pending)


def status_view(status: ModelStatus, root: Path) -> dict[str, Any]:
    """JSON fields of a `ModelStatus` (message redacted)."""
    return {
        "state": status.state,
        "layers": dict(status.layers),
        "missing": list(status.missing),
        "size_bytes": status.size_bytes,
        "message": redact(status.message, root),
    }


PACKAGE_PREFIX = "package:"


def _joined(values: Iterable[str | None]) -> str | None:
    unique = list(dict.fromkeys(v for v in values if v))
    return "; ".join(unique) or None


def declaration_view(req: Requirements | None, root: Path) -> dict[str, Any]:
    """`{source, license, target}` of a declaration, for display: where it
    comes from (each kind's `describe_source`), the declared licences and the
    targets relative to the weights root ("; "-joined, None when empty;
    redacted, never an absolute path)."""
    if req is None:
        return {"source": None, "license": None, "target": None}
    sources = [a.describe_source() for a in req.assets]
    sources += [c.describe_source() for c in req.code]
    targets = [a.target for a in req.assets] + [
        f"{CODE_DIR}/{c.name}" for c in req.code
    ]
    return {
        "source": redact(_joined(sources), root),
        "license": redact(_joined(a.license for a in req.assets), root),
        "target": redact(_joined(targets), root),
    }


def with_action(
    row: dict[str, Any], unknown: bool, manual: bool = False
) -> dict[str, Any]:
    """Add `reason` (why the web cannot prepare it: "unknown_source",
    "user_provided", "packages_only", or None) and `preparable` (a web
    Prepare can fix it)."""
    missing = row.get("missing") or []
    if unknown:
        reason = "unknown_source"
    elif manual:
        reason = "user_provided"
    elif missing and all(m.startswith(PACKAGE_PREFIX) for m in missing):
        reason = "packages_only"
    else:
        reason = None
    row["reason"] = reason
    row["preparable"] = row["state"] == "missing" and reason is None
    return row


def plan_view(plan: pkgs.PackagePlan | None, root: Path) -> dict[str, Any] | None:
    """`{action, extra, command, message}` of a packages plan (None when the
    packages are present); redacted."""
    if plan is None or plan.action == "ready":
        return None
    return {
        "action": plan.action,
        "extra": plan.extra,
        "command": plan.command,
        "message": redact(plan.message, root),
    }


def status_row(
    row: ModelRow,
    root: Path,
    imports: dict[str, tuple[str, ...]] | None = None,
) -> dict[str, Any]:
    """`{stage, index, type, name, state, layers, missing, size_bytes,
    message, source, license, target, reason, preparable, packages}` of one
    model, checked on disk (call from a thread: it walks files and may run
    `git rev-parse` and a cached `uv sync --dry-run`). `imports` are the
    import names per extra of the whole config (extra detection)."""
    if row.req is None:
        return with_action(
            {
                **row.head(),
                "state": "failed",
                "layers": {},
                "missing": [],
                "size_bytes": None,
                "message": redact(row.error, root),
                **declaration_view(None, root),
                "packages": None,
            },
            False,
        )
    status = check(row.req, root)
    plan = None
    if status.layers.get("packages", "ready") != "ready":
        plan = pkgs.plan_packages(row.req, imports=imports)
        status = with_package_plan(status, plan)
    view = status_view(status, root)
    unknown = view["state"] != "failed" and bool(unknown_sources(row.req, root))
    if unknown:
        view["state"], view["message"] = "missing", UNKNOWN_SOURCE
    manual = (
        not unknown
        and view["state"] == "missing"
        and user_provided_only(row.req, status, root)
    )
    return with_action(
        {
            **row.head(),
            **view,
            **declaration_view(row.req, root),
            "packages": plan_view(plan, root),
        },
        unknown,
        manual,
    )


def config_imports(rows: Iterable[ModelRow]) -> dict[str, tuple[str, ...]]:
    """Import names per extra declared by the models of a config."""
    return pkgs.imports_of(r.req for r in rows if r.req is not None)


def report(
    statuses: list[dict[str, Any]], restart_required: bool = False
) -> dict[str, Any]:
    """The `POST /api/models/status` body: the rows, the bytes still to
    download (models that are not ready and declare a size) and whether
    packages were installed since the server started."""
    total = sum(s["size_bytes"] or 0 for s in statuses if s["state"] != "ready")
    return {
        "models": statuses,
        "total_size_bytes": total,
        "restart_required": restart_required,
    }


def not_ready(statuses: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """The rows whose state is not `ready` (a run of them is refused)."""
    return [s for s in statuses if s["state"] != "ready"]


@dataclass
class ModelProgress:
    """State of one model inside a prepare job."""

    row: ModelRow
    state: str = "queued"  # queued|preparing|ready|missing|failed|...|cancelled
    item: str | None = None  # asset key / code:<name> being fetched
    done: int = 0
    total: int | None = None
    layers: dict[str, str] = field(default_factory=dict)
    missing: list[str] = field(default_factory=list)
    size_bytes: int | None = None
    message: str | None = None
    installed: bool = False  # a packages job ran `uv sync` to completion
    # a weights job fetched everything it could; only user files are missing
    manual: bool = False
    # packages job: the plan shown to the user (computed by the server) and
    # the config-wide import names it was computed with
    expected: pkgs.PackagePlan | None = field(default=None, repr=False)
    imports: dict[str, tuple[str, ...]] | None = field(default=None, repr=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.row.head(),
            "installed": self.installed,
            "state": self.state,
            "item": self.item,
            "done": self.done,
            "total": self.total,
            "layers": dict(self.layers),
            "missing": list(self.missing),
            "size_bytes": self.size_bytes,
            "message": self.message,
        }


@dataclass
class PrepareJob:
    """One submitted Prepare (one or more models of a config)."""

    id: str
    models: list[ModelProgress]
    kind: str = "weights"  # weights (prepare) | packages (install packages)
    status: str = "queued"  # queued|running|done|failed|cancelled
    error: str | None = None
    created: str = ""
    events: list[dict[str, Any]] = field(default_factory=list)
    closed: bool = False
    cancel_requested: bool = False
    cancel_event: threading.Event = field(default_factory=threading.Event, repr=False)
    changed: asyncio.Event = field(default_factory=asyncio.Event, repr=False)
    current: ModelProgress | None = field(default=None, repr=False)
    # models of a `models: "all"` request left out: {stage, index, type,
    # name, message} (an unknown source, or only user-provided files missing)
    skipped: list[dict[str, Any]] = field(default_factory=list)
    output_lines: int = 0

    def to_dict(self, position: int | None = None) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "status": self.status,
            "position": position,
            "error": self.error,
            "created": self.created,
            "cancel_requested": self.cancel_requested,
            "models": [m.to_dict() for m in self.models],
            "skipped": [dict(s) for s in self.skipped],
        }


class PrepareQueue:
    """FIFO of Prepare jobs with a single runner (see the module doc)."""

    def __init__(self, weights_root: Path) -> None:
        self.root = Path(weights_root)
        self._jobs: dict[str, PrepareJob] = {}
        self._waiting: list[PrepareJob] = []
        self._wake = asyncio.Event()
        self._runner: asyncio.Task[None] | None = None
        # packages were installed since start: the server must be restarted
        self.restart_required = False

    # -- public API ---------------------------------------------------------

    def submit(
        self,
        models: list[Key] | Literal["all"],
        data: dict[str, Any],
        packages: bool = False,
        plans: dict[Key, pkgs.PackagePlan] | None = None,
    ) -> PrepareJob:
        """Queue a Prepare of `models` (keys from the status rows) or of every
        model of the config (call from the running event loop). With
        `packages`, the job installs the named models' per-model extras
        instead (`install_packages`): `plans` are the server's plans of
        those models (`install_plans`, called from a thread first), each must
        be "install"; the job's fresh dry-run must give the same plan.

        A model with an unknown source (`unknown_sources`), or whose only
        missing files are user-provided (`user_provided_only`), is never
        queued: named explicitly it raises `UnknownSource` /
        `UserProvidedOnly`; with "all" it is left out and listed in the job's
        `skipped` with its message (the manual-download hint).

        Raises:
            ConfigError: the configuration is invalid.
            UnknownModel: a key is not one of the config's models.
            UnknownSource: a named model (or, with "all", every model that
                is not ready) has an unknown source.
            UserProvidedOnly: a named model (or, with "all", every model
                that is not ready) only misses user-provided files.
            NotInstallable: `packages` for a model without a per-model extra
                (the base stack is never installed by the server).
            ValueError: no model selected, or `packages` with "all".
        """
        rows = config_rows(data)
        skipped: list[dict[str, Any]] = []
        if packages and models == "all":
            raise ValueError("Install packages names its models (not 'all')")
        if models == "all":
            chosen, errors = [], []
            for row in rows:
                why = self._not_preparable(row)
                if why is not None:
                    errors.append(why[0])
                    skipped.append({**row.head(), "message": why[1]})
                else:
                    chosen.append(row)
            if skipped and not chosen:
                names = "; ".join(
                    f"{_where_of(s)} {s['name']}: {s['message']}" for s in skipped
                )
                raise errors[0](names)
        else:
            by_key = {row.key: row for row in rows}
            chosen, seen = [], set()
            for key in models:
                key = (key[0], key[1])
                if key not in by_key:
                    stage, index = key
                    where = stage if index is None else f"{stage}[{index}]"
                    raise UnknownModel(f"{where} is not a model of this config")
                row = by_key[key]
                if packages:
                    _check_installable(row, (plans or {}).get(key))
                elif (why := self._not_preparable(row)) is not None:
                    raise why[0](f"{_where(row)} {row.name}: {why[1]}")
                if key not in seen:
                    seen.add(key)
                    chosen.append(row)
        if not chosen:
            raise ValueError("no model to prepare")
        job = PrepareJob(uuid.uuid4().hex[:12], [ModelProgress(r) for r in chosen])
        job.kind = "packages" if packages else "weights"
        if packages:
            imports = config_imports(rows)
            for model in job.models:
                model.expected = (plans or {}).get(model.row.key)
                model.imports = imports
        job.skipped = skipped
        job.created = datetime.now().astimezone().isoformat(timespec="seconds")
        self._jobs[job.id] = job
        self._append(job, {"event": "queued"})
        self._waiting.append(job)
        self._ensure_runner()
        self._wake.set()
        self._forget_old()
        return job

    def _not_preparable(self, row: ModelRow) -> tuple[type[ValueError], str] | None:
        """(error class, message) when a weights Prepare must not take `row`:
        an unknown source, or only user-provided files missing (the message
        is the row's, with the manual-download hint)."""
        if row.req is None:
            return None
        if unknown_sources(row.req, self.root):
            return UnknownSource, UNKNOWN_SOURCE
        status = check(row.req, self.root)
        if status.state == "missing" and user_provided_only(row.req, status, self.root):
            hint = redact(status.message, self.root) or "provide the missing file"
            return UserProvidedOnly, hint
        return None

    def install_plans(
        self, models: list[Key], data: dict[str, Any]
    ) -> dict[Key, pkgs.PackagePlan]:
        """The server's packages plans of `models` (as the status rows show
        them: config-wide import names, cached dry-run). Call from a thread.

        Raises:
            ConfigError: the configuration is invalid.
            UnknownModel: a key is not one of the config's models.
        """
        rows = config_rows(data)
        imports = config_imports(rows)
        by_key = {row.key: row for row in rows}
        plans = {}
        for key in models:
            key = (key[0], key[1])
            row = by_key.get(key)
            if row is None:
                stage, index = key
                where = stage if index is None else f"{stage}[{index}]"
                raise UnknownModel(f"{where} is not a model of this config")
            if row.req is not None:
                plans[key] = pkgs.plan_packages(row.req, imports=imports)
        return plans

    def installing(self) -> bool:
        """An Install packages job is queued or running."""
        return any(
            j.kind == "packages" and j.status not in TERMINAL
            for j in self._jobs.values()
        )

    def get(self, job_id: str) -> PrepareJob:
        """The job, or KeyError."""
        return self._jobs[job_id]

    def recent(self) -> list[PrepareJob]:
        """Unfinished jobs and the latest finished ones, newest first (a
        reloaded page re-attaches to the unfinished ones)."""
        return list(reversed(self._jobs.values()))

    def position(self, job: PrepareJob) -> int | None:
        """Place in the waiting line (0 = starts next); None once started."""
        return self._waiting.index(job) if job in self._waiting else None

    def cancel(self, job_id: str) -> None:
        """Cancel a queued or running job (no-op once it ended). A running one
        ends `cancelled` when the engine notices (between chunks)."""
        job = self._jobs[job_id]
        if job.status in TERMINAL:
            return
        job.cancel_requested = True
        job.cancel_event.set()
        if job in self._waiting:
            self._waiting.remove(job)
            self._finish(job)
        else:
            self._append(job, {"event": "cancelling"})

    def preparing(self) -> list[Requirements]:
        """Requirements of the model being prepared right now (if any)."""
        cur = next(
            (j.current for j in self._jobs.values() if j.current is not None), None
        )
        return [] if cur is None or cur.row.req is None else [cur.row.req]

    async def events(
        self, job_id: str, start: int = 0
    ) -> AsyncIterator[tuple[int, dict[str, Any]]]:
        """`(index, event)` of every event of the job from `start`; ends after
        its `end` event."""
        job = self._jobs[job_id]
        seen = max(0, min(start, len(job.events)))
        while True:
            changed = job.changed
            batch = job.events[seen:]
            for i, event in enumerate(batch, seen):
                yield i, event
            seen += len(batch)
            if job.closed and seen >= len(job.events):
                return
            if len(job.events) == seen:
                await changed.wait()

    async def aclose(self) -> None:
        """Server shutdown: cancel everything; unfinished jobs end `cancelled`."""
        for job in self._jobs.values():
            if job.status not in TERMINAL:
                job.cancel_requested = True
                job.cancel_event.set()
        if self._runner is not None:
            self._runner.cancel()
            try:
                await self._runner
            except asyncio.CancelledError:
                pass
            self._runner = None
        self._waiting.clear()
        for job in self._jobs.values():
            if not job.closed:
                job.error = job.error or SHUTDOWN_ERROR
                self._finish(job)

    # -- internals ----------------------------------------------------------

    def _append(self, job: PrepareJob, event: dict[str, Any]) -> None:
        """Add an event and wake the subscribers (loop thread only)."""
        job.events.append(event)
        job.changed.set()
        job.changed = asyncio.Event()

    def _finish(self, job: PrepareJob) -> None:
        if job.closed:
            return
        if job.cancel_requested:
            job.status = "cancelled"
            job.error = job.error or "cancelled"
        else:
            bad = [m for m in job.models if not _succeeded(job, m)]
            job.status = "failed" if bad else "done"
            if bad:
                job.error = "; ".join(
                    f"{_where(m.row)} {m.row.name}: {m.message or m.state}" for m in bad
                )
        for m in job.models:
            if m.state in ("queued", "preparing"):
                m.state = "cancelled"
        job.current = None
        self._append(job, {"event": "end", "status": job.status, "error": job.error})
        job.closed = True
        job.changed.set()
        log.info(
            "prepare %s: %s%s",
            job.id,
            job.status,
            f" ({job.error})" if job.error else "",
        )

    def _forget_old(self) -> None:
        finished = [j for j in self._jobs.values() if j.closed]
        for job in finished[:-RECENT_FINISHED]:
            del self._jobs[job.id]

    def _ensure_runner(self) -> None:
        if self._runner is None or self._runner.done():
            self._runner = asyncio.get_running_loop().create_task(self._run_forever())

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
            except Exception as exc:  # noqa: BLE001 - a bug must not stall the queue
                job.error = client_text(f"{type(exc).__name__}: {exc}", self.root)
                self._finish(job)

    async def _execute(self, job: PrepareJob) -> None:
        loop = asyncio.get_running_loop()
        job.status = "running"
        self._append(job, {"event": "running"})
        for model in job.models:
            if job.cancel_requested:
                break
            job.current = model
            model.state = "preparing"
            self._append(job, {"event": "model", **model.to_dict()})
            if model.row.req is None:
                model.state, model.message = (
                    "failed",
                    redact(model.row.error, self.root),
                )
            elif job.kind == "packages":
                await self._install(job, model, loop)
            else:
                progress = self._progress(job, model, loop)
                req, root, cancel = model.row.req, self.root, job.cancel_event

                def run(req=req, root=root, cancel=cancel, progress=progress):
                    # The web never fetches a source the adapter does not
                    # vouch for, even if it went missing after queueing.
                    status = prepare(req, root, progress, cancel, allow_unknown=False)
                    manual = status.state == "missing" and user_provided_only(
                        req, status, root
                    )
                    return status, manual

                try:
                    status, model.manual = await asyncio.to_thread(run)
                except Exception as exc:  # noqa: BLE001 - reported, never a crash
                    model.state = "failed"
                    model.message = client_text(f"{type(exc).__name__}: {exc}", root)
                else:
                    view = status_view(status, self.root)
                    model.state = view["state"]
                    model.layers, model.missing = view["layers"], view["missing"]
                    model.size_bytes, model.message = (
                        view["size_bytes"],
                        view["message"],
                    )
            job.current = None
            self._append(job, {"event": "model", **model.to_dict()})
            log.info(
                "prepare %s: %s %s -> %s",
                job.id,
                _where(model.row),
                model.row.name,
                model.state,
            )
        self._finish(job)

    async def _install(self, job: PrepareJob, model: ModelProgress, loop) -> None:
        """Install the packages of one model (fresh dry-run, then `uv sync`)."""
        stage, index = model.row.key

        def deliver(line: str) -> None:
            if job.output_lines >= OUTPUT_LIMIT:
                return
            job.output_lines += 1
            self._append(
                job,
                {
                    "event": "output",
                    "stage": stage,
                    "index": index,
                    "line": redact(line, self.root),
                },
            )

        def on_line(line: str) -> None:  # uv reader thread
            if not loop.is_closed():
                loop.call_soon_threadsafe(deliver, line)

        assert model.row.req is not None
        try:
            outcome = await asyncio.to_thread(
                install_packages,
                model.row.req,
                on_line,
                job.cancel_event,
                root=self.root,
                imports=model.imports,
                expected=model.expected,
            )
        except Exception as exc:  # noqa: BLE001 - reported, never a crash
            model.state = "failed"
            model.message = client_text(f"{type(exc).__name__}: {exc}", self.root)
            return
        view = status_view(outcome.status, self.root)
        model.state = view["state"]
        model.layers, model.missing = view["layers"], view["missing"]
        model.size_bytes, model.message = view["size_bytes"], view["message"]
        model.installed = outcome.installed
        if outcome.installed:
            self.restart_required = True
            model.message = INSTALLED_NOTE

    def _progress(self, job: PrepareJob, model: ModelProgress, loop):
        """Engine progress callback (any thread): throttled, then handed to
        the loop thread."""
        last: dict[str, float] = {}
        lock = threading.Lock()
        stage, index = model.row.key

        def deliver(label: str, done: int, total: int | None) -> None:
            model.item, model.done, model.total = label, done, total
            self._append(
                job,
                {
                    "event": "progress",
                    "stage": stage,
                    "index": index,
                    "item": label,
                    "done": done,
                    "total": total,
                },
            )

        def progress(label: str, done: int, total: int | None) -> None:
            now = time.monotonic()
            final = done == 0 or (total is not None and done >= total)
            with lock:
                if not final and now - last.get(label, -1e9) < PROGRESS_INTERVAL_S:
                    return
                last[label] = now
            if not loop.is_closed():
                loop.call_soon_threadsafe(deliver, label, done, total)

        return progress


def _check_installable(row: ModelRow, plan: pkgs.PackagePlan | None) -> None:
    """Raise `NotInstallable` unless the server may install the packages of
    `row`: a per-model extra (never the base stack) whose server-side plan is
    "install" (a clean dry-run)."""
    extra = row.req.extra if row.req is not None else None
    if extra is None:
        raise NotInstallable(
            f"{_where(row)} {row.name}: no extra declares its packages"
        )
    if extra in pkgs.BASE_EXTRAS:
        raise NotInstallable(
            f"{_where(row)} {row.name}: the base Python stack is never installed "
            f"by the server: run `{pkgs.BASE_COMMAND}` once and restart it"
        )
    where = f"{_where(row)} {row.name}"
    if plan is None:
        raise NotInstallable(f"{where}: no packages plan for this model")
    if plan.action == "base":
        raise NotInstallable(
            f"{where}: the base Python stack is not installed and the server "
            f"never installs it: run `{pkgs.BASE_COMMAND}` once and restart it"
        )
    if plan.action == "ready":
        raise NotInstallable(f"{where}: its packages are already installed")
    if plan.action != "install":
        raise NotInstallable(f"{where}: {plan.message or plan.action}")


def _succeeded(job: PrepareJob, model: ModelProgress) -> bool:
    """A Prepare succeeds when the model is ready, or when everything it
    could fetch is there and only user-provided files are missing (the row
    then says which); Install packages when its packages layer is ready (its
    weights may still be missing)."""
    if job.kind == "packages":
        return model.layers.get("packages") == "ready"
    return model.state == "ready" or model.manual


def _where(row: ModelRow) -> str:
    return row.stage if row.index is None else f"{row.stage}[{row.index}]"


def _where_of(head: dict[str, Any]) -> str:
    stage, index = head["stage"], head["index"]
    return stage if index is None else f"{stage}[{index}]"
