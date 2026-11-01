"""Request guards of the web server (no auth: these keep other sites out).

- `allowed_hosts`: the `Host` values `TrustedHostMiddleware` accepts (the
  loopback names, the bound host and `--allowed-host`), so a DNS-rebinding
  page, whose requests carry its own host name, is refused.
- `OriginGuard`: a state-changing request (POST/PUT/PATCH/DELETE) whose
  `Origin` is not the server's own (nor a `--dev-origin`) is a 403; browsers
  send `Origin` on every cross-site POST, uploads and body-less ones included.
- `BodyLimit`: the JSON bodies of state-changing requests are capped
  (`MAX_BODY_BYTES`); the upload route has its own, streamed limits.
- `path_problems`: the filesystem paths of a run configuration (discovered by
  `schema.config_paths`, resolved like the adapter opens them) may not point
  into the folders clients write (library, run outputs) and, with
  `--asset-root`, must lie inside one or inside the weights root (where
  Prepare writes only what the adapters' code declares).
- `weights_root_problem`: the weights root may not be `/`, the home folder,
  contain the project or the active environment, or overlap a folder
  clients write.
- `file_headers`: served files never run as a page of the app's origin.
"""

import os
from collections.abc import Collection, Iterable
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from fastapi import HTTPException
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from det_depth_distance.core import requirements
from det_depth_distance.core.requirements import weights_root
from det_depth_distance.web import schema
from det_depth_distance.web.paths import WebPaths

LOOPBACK_HOSTS = ("127.0.0.1", "localhost", "[::1]")
WILDCARD_HOSTS = ("0.0.0.0", "::", "")  # bind-all: not a name a client sends
UNSAFE_METHODS = frozenset({"POST", "PUT", "PATCH", "DELETE"})
MAX_BODY_BYTES = 1 << 20
# Run-contract paths the server checks or sets itself (submit_run binds and
# contains source.video, JobQueue.submit forces output.out_root).
SERVER_CHECKED_PATHS = frozenset({"source.video", "output.out_root"})
# Media types a served file may be shown inline with (anything else, e.g.
# HTML/SVG/JS a future sink might write, is a download).
INLINE_MEDIA = frozenset(
    {
        "image/png",
        "image/jpeg",
        "image/webp",
        "image/gif",
        "text/csv",
        "text/plain",
        "application/json",
        "application/x-ndjson",
        "application/yaml",
    }
)
_FILE_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "Content-Security-Policy": "sandbox",
}


def is_loopback(host: str) -> bool:
    """Whether a bind address only accepts local connections."""
    return host.strip("[]") in ("127.0.0.1", "localhost", "::1") or host.startswith(
        "127."
    )


def _host_pattern(host: str) -> str:
    """`Host`-header form of a host name (IPv6 in brackets, lower case)."""
    host = host.strip().lower()
    if ":" in host and not host.startswith("["):
        return f"[{host}]"
    return host


def allowed_hosts(bind_host: str, extra: Iterable[str] = ()) -> list[str]:
    """Host names the server answers to: loopback, the bind host (unless it
    binds every interface) and the `--allowed-host` values."""
    names = [*LOOPBACK_HOSTS]
    if bind_host not in WILDCARD_HOSTS:
        names.append(bind_host)
    names += list(extra)
    out: list[str] = []
    for name in names:
        pattern = _host_pattern(name)
        if pattern and pattern not in out:
            out.append(pattern)
    return out


def _origin_netloc(origin: str) -> str | None:
    parts = urlsplit(origin)
    if parts.scheme not in ("http", "https") or not parts.netloc:
        return None
    return parts.netloc.lower()


class OriginGuard:
    """403 for a state-changing request from another origin.

    Same-origin requests (the `Origin` host:port equals the `Host` header)
    and requests without `Origin` (curl, scripts; a browser always sends it
    cross-site) pass; `dev_origins` (e.g. the Vite dev server) pass too.
    """

    def __init__(self, app: ASGIApp, dev_origins: Collection[str] = ()) -> None:
        self.app = app
        self.dev = {n for o in dev_origins if (n := _origin_netloc(o))}

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http" and scope["method"] in UNSAFE_METHODS:
            headers = {
                k.decode("latin-1"): v.decode("latin-1") for k, v in scope["headers"]
            }
            origin = headers.get("origin")
            host = headers.get("host", "").lower()
            if origin is not None:
                netloc = _origin_netloc(origin)
                refused = netloc is None or (netloc != host and netloc not in self.dev)
            else:
                refused = headers.get("sec-fetch-site") == "cross-site"
            if refused:
                response = JSONResponse(
                    {"detail": "cross-origin request refused"}, status_code=403
                )
                await response(scope, receive, send)
                return
        await self.app(scope, receive, send)


class BodyLimit:
    """413 for a state-changing request whose body passes `limit` bytes
    (by `Content-Length`, or counted as it streams); `exempt` paths (the
    upload) are not limited here."""

    def __init__(
        self, app: ASGIApp, limit: int = MAX_BODY_BYTES, exempt: Collection[str] = ()
    ) -> None:
        self.app = app
        self.limit = limit
        self.exempt = frozenset(exempt)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if (
            scope["type"] != "http"
            or scope["method"] not in UNSAFE_METHODS
            or scope["path"] in self.exempt
        ):
            await self.app(scope, receive, send)
            return
        too_large_detail = f"request body larger than {self.limit >> 10} KiB"
        too_large = JSONResponse({"detail": too_large_detail}, status_code=413)
        for key, value in scope["headers"]:
            if key == b"content-length":
                try:
                    declared = int(value)
                except ValueError:
                    declared = 0
                if declared > self.limit:
                    await too_large(scope, receive, send)
                    return
        received = 0

        async def limited_receive() -> Message:
            # Raised while the route reads its body: FastAPI re-raises an
            # HTTPException from there and answers it (413).
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self.limit:
                    raise HTTPException(413, too_large_detail)
            return message

        await self.app(scope, limited_receive, send)


def resolve(value: Path | str) -> Path:
    """A config path as the worker opens it: `~` expanded, relative to the
    working directory (the repo root), symlinks followed."""
    return (Path.cwd() / os.path.expanduser(str(value))).resolve()


def within(path: Path, roots: Iterable[Path]) -> bool:
    """`path` is one of `roots` or inside one (all resolved)."""
    return any(path == r or path.is_relative_to(r) for r in roots)


def client_writable_roots(paths: WebPaths) -> list[Path]:
    """Folders whose files a client can create (uploads, staging, outputs)."""
    return [paths.library_root.resolve(), paths.out_root.resolve()]


def path_problems(config: dict[str, Any], paths: WebPaths) -> list[dict[str, str]]:
    """Errors (`{"path", "message"}`) of the path fields of a valid config:
    none may resolve inside the library or out_root (a client could have put
    a checkpoint or a Python package there), and with `paths.asset_roots`
    each must resolve inside one of them or inside the weights root."""
    writable = client_writable_roots(paths)
    assets = [r.resolve() for r in paths.asset_roots]
    if assets:
        assets.append(weights_root().resolve())
    errors = []
    for dotted, value in schema.config_paths(config):
        if dotted in SERVER_CHECKED_PATHS:
            continue
        real = resolve(value)
        if within(real, writable):
            message = (
                "must not point into the video library or the run outputs "
                "(folders clients write)"
            )
        elif assets and not within(real, assets):
            message = "is outside the asset folders of the server (--asset-root)"
        else:
            continue
        errors.append({"path": dotted, "message": f"{value}: {message}"})
    return errors


def weights_root_problem(root: Path, paths: WebPaths) -> str | None:
    """Why `root` cannot be the weights root (None when it can): never `/`,
    the home folder or a folder containing the project or the active Python
    environment (`core.requirements.weights_root_problem`); and it may not be,
    contain or lie inside a folder clients write (an upload would then count
    as a prepared checkpoint, or a download land among the uploads)."""
    problem = requirements.weights_root_problem(root)
    if problem is not None:
        return problem
    real = Path(root).expanduser().resolve()
    for folder in client_writable_roots(paths):
        if within(real, [folder]) or within(folder, [real]):
            return (
                f"weights root {root} overlaps {folder} (the video library or "
                "the run outputs, folders clients write): use another folder"
            )
    return None


def file_headers(media_type: str, filename: str) -> dict[str, str]:
    """Headers of a served file: never sniffed, sandboxed if opened as a
    document, and a download unless its type is a plain media type."""
    headers = dict(_FILE_HEADERS)
    if not (media_type.startswith("video/") or media_type in INLINE_MEDIA):
        safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in filename)
        headers["Content-Disposition"] = f'attachment; filename="{safe or "file"}"'
    return headers
