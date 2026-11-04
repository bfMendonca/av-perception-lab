"""Packages layer: detect the installed extras, dry-run, then install.

A component declares the import names it needs (`Requirements.packages`) and
the `pyproject.toml` extra that provides them (`Requirements.extra`). This
module never imports a declared package:

- `detect_extras` reads `[project.optional-dependencies]` (tomllib) and keeps
  the extras whose distributions are installed (`importlib.metadata`) and
  whose import names, as the components declare them, are found by the
  top-level `find_spec`;
- `dry_run` runs `uv sync --dry-run` with `--extra` for *every* detected extra
  plus the new one and reports the resolver's reason when it fails; it
  changes nothing (no lock written, no package installed);
- `install` runs the same command without `--dry-run` in its own process
  group (streamed output, killed as a whole on cancel).

The command is never a bare `uv sync` (which would remove every extra not
named): it always names every detected extra and passes `--inexact` (nothing
extraneous is removed) and `--locked` (the lock is not rewritten). A dry-run
that would uninstall or replace a package counts as a conflict. The base
stack (`BASE_EXTRAS`, today the `infer` extra) is never installed here: the
plan reports the one-time command (`BASE_COMMAND`).
"""

import importlib
import importlib.metadata
import os
import re
import shlex
import signal
import subprocess
import sys
import threading
import time
import tomllib
from collections import deque
from collections.abc import Callable, Iterable, Mapping
from concurrent.futures import Future
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from av_perception_lab.core import requirements as rq
from av_perception_lab.core.provision.engine import (
    ProvisionCancelled,
    ProvisionError,
)

BASE_EXTRAS = frozenset({"infer"})  # the GPU stack every current adapter needs
BASE_COMMAND = "uv sync --extra infer --extra web"  # one-time, documented (README)
CLI_BASE_COMMAND = "uv sync --extra infer"  # the CLI alone does not need `web`
DOCS = "docs/model_environments.md"
DRY_RUN_TIMEOUT_S = 180.0
INSTALL_TIMEOUT_S = 3600.0
POLL_S = 0.2
MESSAGE_LIMIT = 1500  # characters of resolver output kept in a message

# PEP 685 normalised extra names; nothing that could be read as an option.
_EXTRA_RE = re.compile(r"[a-z0-9](?:[a-z0-9._-]*[a-z0-9])?")
_DIST_RE = re.compile(r"\s*([A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?)")
_ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")
# uv's own progress lines, not part of a resolver reason
_NOISE = (
    "Resolved ",
    "Prepared ",
    "Audited ",
    "Using ",
    "Checked ",
    "Would ",
    "Discovered ",
)


def default_pyproject() -> Path:
    """The repository's `pyproject.toml`."""
    return Path(__file__).resolve().parents[4] / "pyproject.toml"


def default_uv() -> str:
    """`AVPL_UV` (a path to the `uv` executable) or `uv` on the PATH."""
    return os.environ.get("AVPL_UV") or "uv"


def _marker_applies(spec: str) -> bool:
    if ";" not in spec:
        return True
    try:
        from packaging.markers import Marker
    except ImportError:  # no `packaging`: count the entry as needed
        return True
    try:
        return Marker(spec.split(";", 1)[1]).evaluate()
    except Exception:  # noqa: BLE001 - an unreadable marker counts as needed
        return True


def declared_extras(pyproject: Path | None = None) -> dict[str, tuple[str, ...]]:
    """`{extra: distribution names}` of `[project.optional-dependencies]`.

    Raises:
        OSError: the file cannot be read.
        ValueError: it is not valid TOML (`tomllib.TOMLDecodeError`).
    """
    path = default_pyproject() if pyproject is None else Path(pyproject)
    with path.open("rb") as handle:
        data = tomllib.load(handle)
    table = data.get("project", {}).get("optional-dependencies", {})
    out: dict[str, tuple[str, ...]] = {}
    for extra, specs in table.items():
        names = []
        for spec in specs if isinstance(specs, list) else []:
            match = _DIST_RE.match(str(spec))
            if match and _marker_applies(str(spec)):
                names.append(match[1])
        out[str(extra)] = tuple(names)
    return out


def _installed(distribution: str) -> bool:
    try:
        importlib.metadata.distribution(distribution)
    except importlib.metadata.PackageNotFoundError:
        return False
    except (OSError, ValueError):
        return False
    return True


def imports_of(reqs: Iterable[rq.Requirements]) -> dict[str, tuple[str, ...]]:
    """`{extra: import names}` declared by `reqs` (merged per extra)."""
    out: dict[str, list[str]] = {}
    for req in reqs:
        if req.extra:
            names = out.setdefault(req.extra, [])
            names += [p for p in req.packages if p not in names]
    return {k: tuple(v) for k, v in out.items()}


def detect_extras(
    pyproject: Path | None = None,
    imports: Mapping[str, Iterable[str]] | None = None,
) -> set[str]:
    """Extras of `pyproject` that are installed: every declared distribution
    has metadata and every import name `imports[extra]` (declared by the
    components) is found by `find_spec`. Nothing is imported.

    Raises:
        OSError, ValueError: `pyproject` cannot be read.
    """
    importlib.invalidate_caches()  # an install made by this process is seen
    imports = imports or {}
    found = set()
    for extra, distributions in declared_extras(pyproject).items():
        if not all(_installed(d) for d in distributions):
            continue
        if all(rq._top_level_present(n) for n in imports.get(extra, ())):
            found.add(extra)
    return found


def sync_command(
    extra: str, detected: Iterable[str], uv: str = "uv", *, dry_run: bool
) -> list[str]:
    """`uv sync` naming every detected extra plus `extra`, `--inexact` and
    `--locked` (never a bare sync, never a removal); `--dry-run` when asked.

    Raises:
        ValueError: an extra name is not a normalised extra name.
    """
    extras = sorted({*detected, extra})
    for name in extras:
        if not _EXTRA_RE.fullmatch(name):
            raise ValueError(f"invalid extra name {name!r}")
    argv = [uv, "--color", "never", "sync", "--locked", "--inexact"]
    for name in extras:
        argv += ["--extra", name]
    if dry_run:
        argv.append("--dry-run")
    return argv


def command_text(argv: list[str]) -> str:
    """The command as shown to a person (`uv`, not the executable's path)."""
    return shlex.join(["uv", *argv[1:]])


def _subprocess_env() -> dict[str, str]:
    env = dict(os.environ)
    env["NO_COLOR"] = "1"
    # The environment that was inspected is the one that changes: this
    # interpreter's venv unless the caller chose one.
    if "UV_PROJECT_ENVIRONMENT" not in env and sys.prefix != sys.base_prefix:
        env["UV_PROJECT_ENVIRONMENT"] = sys.prefix
    return env


# Credentials that may appear in uv output (index URLs, tokens).
_SECRETS = (
    (re.compile(r"(?<=://)[^/\s@'\"]+@"), "***@"),  # scheme://user:pass@host
    (
        re.compile(
            r"(?i)([?&;](?:[a-z0-9_-]*(?:token|password|passwd|secret|key|sig|"
            r"signature|auth|credential)[a-z0-9_-]*)=)[^&\s'\"]+"
        ),
        r"\1***",
    ),
    # an Authorization header (any scheme), then a lone `Bearer <token>`; prose
    # such as "basic requirements" or "token count" is left alone
    (
        re.compile(
            r"(?i)\b(authorization[ \t]*[:=][ \t]*"
            r"(?:(?:bearer|basic|token|digest)[ \t]+)?)[^\s'\"]+"
        ),
        r"\1***",
    ),
    (re.compile(r"(?i)\b(bearer[ \t]+)[A-Za-z0-9._~+/=-]{8,}"), r"\1***"),
    # well-known token shapes (Hugging Face, GitHub, GitLab, PyPI)
    (
        re.compile(r"\b(?:hf_|gh[pousr]_|github_pat_|glpat-|pypi-)[A-Za-z0-9_-]{16,}"),
        "***",
    ),
)


def scrub(text: str) -> str:
    """`text` without credentials: `user:pass@` in URLs, token-like query
    values, `Authorization:` values, `Bearer <token>` and well-known token
    shapes become `***`."""
    for pattern, repl in _SECRETS:
        text = pattern.sub(repl, text)
    return text


def relative(text: str, project: Path) -> str:
    """`text` with credentials scrubbed, the project folder shown as `.` and
    the home as `~`."""
    out = rq.redact_root(scrub(text), project, ".") or ""
    return rq.redact_root(out, Path.home(), "~") or ""


def resolver_reason(output: str) -> str:
    """The part of `uv` output that explains a failure (from the first
    `error`/`×` line; progress lines dropped), at most `MESSAGE_LIMIT` chars."""
    lines = [_ANSI_RE.sub("", line).rstrip() for line in output.splitlines()]
    lines = [line for line in lines if line.strip()]
    start = next(
        (
            i
            for i, line in enumerate(lines)
            if re.match(r"\s*(error|×|x\s)", line, re.IGNORECASE)
        ),
        None,
    )
    kept = lines[start:] if start is not None else lines
    kept = [line for line in kept if not line.lstrip().startswith(_NOISE)]
    text = "\n".join(kept).strip() or "uv failed without a message"
    return text if len(text) <= MESSAGE_LIMIT else text[: MESSAGE_LIMIT - 1] + "…"


# DryRunResult.kind: "ok"; "resolver" (uv refused: a conflict); "replace" (the
# plan removes/replaces packages: a conflict); "invalid" (bad extra name);
# "unavailable" (uv not found or not executable) and "timeout" are transient
# tooling failures: never a conflict, never cached.
TRANSIENT = frozenset({"unavailable", "timeout", "cancelled"})


@dataclass(frozen=True, slots=True)
class DryRunResult:
    """`ok` when the resolution is clean and removes nothing; `message` is the
    planned change or the reason (project-relative); `command` is the install
    command it validates; `changes` the `+ name==version` lines; `kind` says
    which outcome it is (see `TRANSIENT`)."""

    ok: bool
    message: str
    command: str = ""
    changes: tuple[str, ...] = ()
    kind: str = "ok"


def dry_run(
    extra: str,
    detected: Iterable[str],
    uv: str = "uv",
    *,
    pyproject: Path | None = None,
    timeout: float | None = None,
    cancel: threading.Event | None = None,
) -> DryRunResult:
    """Run `uv sync --dry-run` for `detected` plus `extra` and report.

    uv runs in its own process group, killed as a whole on timeout or when
    `cancel` is set. Never raises for a uv failure: a resolver error or a
    plan that removes/replaces packages is `ok=False` (a conflict); a missing
    executable, a timeout or a cancel is `ok=False` with a `TRANSIENT` kind.
    """
    project = (default_pyproject() if pyproject is None else Path(pyproject)).parent
    try:
        argv = sync_command(extra, detected, uv, dry_run=True)
    except ValueError as exc:
        return DryRunResult(False, str(exc), kind="invalid")
    command = command_text(argv[:-1])  # what Install would run
    timeout = DRY_RUN_TIMEOUT_S if timeout is None else timeout
    try:
        proc = subprocess.Popen(
            argv,
            cwd=project,
            env=_subprocess_env(),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,  # its own process group, killed as a whole
        )
    except OSError as exc:
        return DryRunResult(
            False,
            f"uv is not available ({exc.strerror or exc}): install uv or set "
            "AVPL_UV to the uv executable (e.g. ~/.local/bin/uv) and restart",
            command,
            kind="unavailable",
        )
    deadline = time.monotonic() + timeout
    try:
        while True:
            try:
                stdout, stderr = proc.communicate(timeout=POLL_S)
                break
            except subprocess.TimeoutExpired:
                if cancel is not None and cancel.is_set():
                    _kill(proc)
                    return DryRunResult(False, "cancelled", command, kind="cancelled")
                if time.monotonic() > deadline:
                    _kill(proc)
                    return DryRunResult(
                        False,
                        f"uv dry-run timed out after {timeout:g} s (retried on "
                        "the next check)",
                        command,
                        kind="timeout",
                    )
    finally:
        if proc.poll() is None:  # KeyboardInterrupt or another error
            _kill(proc)
    text = f"{stdout}\n{stderr}"
    if proc.returncode != 0:
        return DryRunResult(
            False, relative(resolver_reason(text), project), command, kind="resolver"
        )
    plan = [_ANSI_RE.sub("", line).strip() for line in text.splitlines()]
    removed = [line[2:].strip() for line in plan if line.startswith("- ")]
    added = tuple(line[2:].strip() for line in plan if line.startswith("+ "))
    if removed:
        return DryRunResult(
            False,
            relative(
                "installing it would remove or replace installed packages: "
                f"{', '.join(removed)}; the environment is left untouched",
                project,
            ),
            command,
            kind="replace",
        )
    message = (
        relative(f"would install {', '.join(added)}", project)
        if added
        else "nothing to install"
    )
    return DryRunResult(True, message, command, added)


def _kill(proc: subprocess.Popen) -> None:
    """Kill `uv` and every process it spawned (builds, downloads); the pipes
    are closed without reading them (a stray holder cannot block us)."""
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except (AttributeError, ProcessLookupError, PermissionError):
        proc.kill()
    for pipe in (proc.stdout, proc.stderr):
        if pipe is not None:
            try:
                pipe.close()
            except OSError:
                pass
    proc.wait()


def install(
    extra: str,
    detected: Iterable[str],
    uv: str = "uv",
    *,
    pyproject: Path | None = None,
    on_line: Callable[[str], None] = lambda line: None,
    cancel: threading.Event | None = None,
    timeout: float = INSTALL_TIMEOUT_S,
) -> None:
    """Install `extra` next to `detected`: the dry-run command without
    `--dry-run`. Run a clean `dry_run` first (`engine.install_packages` does).
    Output lines go to `on_line` (project-relative); `cancel` kills the
    process group.

    Raises:
        ProvisionError: a base extra, the base stack not detected, an extra
            the pyproject does not declare, or uv failed (with its reason).
        ProvisionCancelled: `cancel` was set.
    """
    detected = set(detected)
    path = default_pyproject() if pyproject is None else Path(pyproject)
    project = path.parent
    if extra in BASE_EXTRAS:
        raise ProvisionError(
            f"the base stack (extra {extra!r}) is never installed here: "
            f"run `{BASE_COMMAND}` once"
        )
    if not BASE_EXTRAS <= detected:
        raise ProvisionError(
            f"the base stack is not installed: run `{BASE_COMMAND}` once first"
        )
    try:
        declared = declared_extras(path)
    except (OSError, ValueError) as exc:
        raise ProvisionError(f"cannot read pyproject.toml ({exc})") from None
    if extra not in declared:
        raise ProvisionError(f"extra {extra!r} is not declared in pyproject.toml")
    try:
        argv = sync_command(extra, detected, uv, dry_run=False)
    except ValueError as exc:
        raise ProvisionError(str(exc)) from None
    cancel = cancel or threading.Event()
    try:
        proc = subprocess.Popen(
            argv,
            cwd=project,
            env=_subprocess_env(),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            start_new_session=True,  # its own process group, killed as a whole
        )
    except OSError as exc:
        raise ProvisionError(
            f"uv is not available ({exc.strerror or exc}): install uv or set "
            "AVPL_UV to the uv executable"
        ) from None
    tail: deque[str] = deque(maxlen=40)

    def read() -> None:
        assert proc.stdout is not None
        try:
            for raw in proc.stdout:
                line = relative(_ANSI_RE.sub("", raw).rstrip(), project)
                if line:
                    tail.append(line)
                    on_line(line)
        except (OSError, ValueError):  # the pipe was closed by a kill
            pass

    reader = threading.Thread(target=read, name="uv-install-output", daemon=True)
    reader.start()
    deadline = time.monotonic() + timeout
    try:
        while True:
            try:
                proc.wait(timeout=POLL_S)
                break
            except subprocess.TimeoutExpired:
                if cancel.is_set():
                    _kill(proc)
                    raise ProvisionCancelled() from None
                if time.monotonic() > deadline:
                    _kill(proc)
                    raise ProvisionError(
                        f"uv sync timed out after {timeout:g} s"
                    ) from None
    finally:
        if proc.poll() is None:  # KeyboardInterrupt or another error
            _kill(proc)
        reader.join(timeout=5)
    if proc.returncode != 0:
        raise ProvisionError(f"uv sync failed: {resolver_reason(chr(10).join(tail))}")
    importlib.invalidate_caches()
    clear_cache()


Action = Literal["ready", "base", "install", "conflict", "invalid", "error"]


@dataclass(frozen=True, slots=True)
class PackagePlan:
    """What the packages layer of one component needs.

    `ready`: every import name is found. `base`: the base stack (or the base
    a per-model extra needs) is missing; `command` is the one-time command,
    never run here. `install`: a per-model `extra` whose dry-run is clean;
    `command` is what Install runs, `extras` every extra it names. `conflict`:
    the dry-run failed; `message` is the resolver's reason. `invalid`: the
    declaration cannot be acted on (no extra, an undeclared extra). `error`:
    the dry-run could not run (uv unavailable, timeout, cancelled): not a
    conflict, asked again on the next check.
    """

    action: Action
    extra: str | None = None
    command: str | None = None
    message: str | None = None
    extras: tuple[str, ...] = ()


_CACHE: dict[tuple, DryRunResult] = {}
_INFLIGHT: dict[tuple, Future] = {}
_CACHE_LOCK = threading.Lock()
_CACHE_SIZE = 32


def clear_cache() -> None:
    """Forget the cached dry-runs (after an install); runs in flight finish
    for their callers but are not stored."""
    with _CACHE_LOCK:
        _CACHE.clear()
        _INFLIGHT.clear()


def _mtime(path: Path) -> int:
    try:
        return path.stat().st_mtime_ns
    except OSError:
        return 0


def cached_dry_run(
    extra: str, detected: Iterable[str], uv: str, pyproject: Path
) -> DryRunResult:
    """`dry_run`, remembered per (extra, detected extras, uv, pyproject and
    lock modification times): the status route asks often. Concurrent callers
    with the same key share one uv run (single flight); a `TRANSIENT`
    result (uv unavailable, timeout) is returned but never stored."""
    detected = frozenset(detected)
    key = (
        extra,
        detected,
        uv,
        str(pyproject),
        _mtime(pyproject),
        _mtime(pyproject.with_name("uv.lock")),
    )
    with _CACHE_LOCK:
        if key in _CACHE:
            return _CACHE[key]
        future = _INFLIGHT.get(key)
        owner = future is None
        if owner:
            future = _INFLIGHT[key] = Future()
    if not owner:
        return future.result()
    try:
        result = dry_run(extra, detected, uv, pyproject=pyproject)
    except BaseException as exc:
        with _CACHE_LOCK:
            if _INFLIGHT.get(key) is future:
                del _INFLIGHT[key]
        future.set_exception(exc)
        raise
    with _CACHE_LOCK:
        if _INFLIGHT.get(key) is future:
            del _INFLIGHT[key]
            if result.kind not in TRANSIENT:
                if len(_CACHE) >= _CACHE_SIZE:
                    _CACHE.clear()
                _CACHE[key] = result
    future.set_result(result)
    return result


def missing_packages(req: rq.Requirements) -> list[str]:
    """Import names of `req` that `find_spec` does not find."""
    return [p for p in req.packages if not rq._top_level_present(p)]


def plan_packages(
    req: rq.Requirements,
    *,
    pyproject: Path | None = None,
    uv: str | None = None,
    imports: Mapping[str, Iterable[str]] | None = None,
    fresh: bool = False,
    cancel: threading.Event | None = None,
) -> PackagePlan:
    """The packages plan of `req` (see `PackagePlan`). Runs a dry-run only
    for a missing per-model extra while the base stack is installed (cached
    unless `fresh`; a fresh one stops when `cancel` is set); never installs
    anything. `imports` are the import names per extra of the whole config:
    the same mapping must be used for the plan shown and the plan run."""
    missing = missing_packages(req)
    if not missing:
        return PackagePlan("ready", req.extra)
    names = ", ".join(missing)
    extra = req.extra
    if extra is None:
        return PackagePlan(
            "invalid",
            None,
            message=f"package(s) {names} not installed and no extra declares them",
        )
    if extra in BASE_EXTRAS:
        return PackagePlan(
            "base",
            extra,
            BASE_COMMAND,
            f"the base Python stack (extra {extra!r}) is not installed "
            f"(missing {names}); the server never installs it",
        )
    path = default_pyproject() if pyproject is None else Path(pyproject)
    try:
        declared = declared_extras(path)
    except (OSError, ValueError) as exc:
        return PackagePlan(
            "invalid", extra, message=f"cannot read pyproject.toml ({exc})"
        )
    if extra not in declared:
        return PackagePlan(
            "invalid",
            extra,
            message=f"extra {extra!r} is not declared in pyproject.toml",
        )
    merged = {k: tuple(v) for k, v in (imports or {}).items()}
    merged[extra] = tuple(dict.fromkeys((*merged.get(extra, ()), *req.packages)))
    try:
        detected = detect_extras(path, merged)
    except (OSError, ValueError) as exc:
        return PackagePlan(
            "invalid", extra, message=f"cannot read pyproject.toml ({exc})"
        )
    detected.discard(extra)
    if not BASE_EXTRAS <= detected:
        return PackagePlan(
            "base",
            extra,
            BASE_COMMAND,
            f"extra {extra!r} needs the base Python stack, which is not installed",
        )
    uv = default_uv() if uv is None else uv
    result = (
        dry_run(extra, detected, uv, pyproject=path, cancel=cancel)
        if fresh
        else cached_dry_run(extra, detected, uv, path)
    )
    extras = tuple(sorted({*detected, extra}))
    if result.ok:
        return PackagePlan("install", extra, result.command, result.message, extras)
    if result.kind in TRANSIENT:
        return PackagePlan(
            "error",
            extra,
            None,
            f"cannot check extra {extra!r}: {result.message}",
            extras,
        )
    return PackagePlan(
        "conflict",
        extra,
        None,
        f"installing extra {extra!r} conflicts with the current environment: "
        f"{result.message} (see {DOCS})",
        extras,
    )
