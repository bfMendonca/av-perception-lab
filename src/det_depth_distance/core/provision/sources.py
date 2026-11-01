"""Built-in sources: Hugging Face, plain URL, Google Drive, git clone, user-provided.

Every write path is resolved through `core.requirements.contained` before
anything is written, so a target, a `.part` file or a clone folder cannot
leave the weights root. Only the standard library is imported here;
`huggingface_hub` is imported inside the Hugging Face source.

- `HuggingFaceAsset`: `huggingface_hub.snapshot_download` into
  `<root>/<target>` as a cache dir (the layout `HuggingFaceAsset.present`
  reads), restricted to the declared files/patterns. The hub keeps its own
  `.incomplete` blobs (never ready) and resumes them.
- `UrlAsset`: streams to `<target>.part`, then renames atomically. A file is
  accepted only if the declared size or `sha256` matches, or the server's
  full length was received; otherwise it fails ("cannot verify download
  integrity"). A `.part` is resumed (`Range` + `If-Range` with the validator
  saved in `<target>.part.meta`) only when size or `sha256` is declared and
  the server answers 206 at the right offset; anything else restarts. A
  mismatch deletes the `.part`; a cancel keeps it for resume. Redirects go
  only to http(s), never from https down to http. An existing target is never
  replaced (a file there that is not the declared one is the user's). The
  body is bounded by the expected size (the server's length, else the
  declared one): more bytes fail and remove the `.part`; free disk space is
  checked before the first byte is written.
- `GoogleDriveAsset`: the same download code as `UrlAsset` (one base class,
  `_HttpFileSource`) from the fixed endpoint `DRIVE_DOWNLOAD` with
  `id=<file_id>&export=download&confirm=t`. Redirects stay on https and on
  the Google hosts Drive downloads use (`DRIVE_HOSTS`, `*.googleusercontent.com`);
  an HTML answer (the virus-scan confirmation or the quota page) is refused
  before anything is kept, never saved as the file.
- `CodeRepo`: `git clone --no-checkout` into `<name>.part`, `switch --detach`
  to the pinned commit (hex only, after `--end-of-options`), rename. Each git
  command is killed (with its helpers) on cancel or after `GIT_TIMEOUT_S`. An existing clone at another commit is fetched and
  checked out, never deleted; a folder that is not a clone is left alone.
- `UserProvided`: never downloads; raises with the declared hint.
"""

import hashlib
import http.client
import os
import shutil
import signal
import subprocess
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from det_depth_distance.core.provision.engine import (
    ItemProgress,
    ProvisionCancelled,
    ProvisionError,
    register_source,
)
from det_depth_distance.core.requirements import (
    CODE_DIR,
    Asset,
    CodeRepo,
    GoogleDriveAsset,
    HuggingFaceAsset,
    UrlAsset,
    UserProvided,
    contained,
)

CHUNK = 1 << 16  # bytes per read; the cancel event is checked between reads
HTTP_TIMEOUT_S = 30.0
GIT = "git"  # executable
GIT_POLL_S = 0.1  # how often a running git checks the cancel event
GIT_TIMEOUT_S = 900.0  # per git command (a stalled clone or fetch fails)


def _check_cancel(cancel: threading.Event) -> None:
    if cancel.is_set():
        raise ProvisionCancelled()


# --------------------------------------------------------------- URL

UNVERIFIABLE = (
    "cannot verify download integrity: declare size_bytes or sha256 for this asset"
)


def _content_range(header: str | None) -> tuple[int | None, int | None]:
    """(start, total) of a `Content-Range: bytes a-b/n` header."""
    if not header or not header.startswith("bytes "):
        return None, None
    span, _, tail = header[6:].partition("/")
    first = span.split("-", 1)[0]
    start = int(first) if first.isdigit() else None
    return start, int(tail) if tail.isdigit() else None


def _validator(headers) -> str | None:
    """A strong validator usable in `If-Range` (a weak ETag is not)."""
    etag = headers.get("ETag")
    if etag and not etag.startswith("W/"):
        return etag
    return headers.get("Last-Modified")


def _http_reason(exc: Exception, where: str, gated: bool = False) -> str:
    if isinstance(exc, urllib.error.HTTPError):
        if exc.code == 404:
            return f"not found (HTTP 404) at {where}"
        if exc.code in (401, 403):
            hint = (
                " - the file is gated: accept the licence on the source site"
                if gated
                else ""
            )
            return f"access denied (HTTP {exc.code}) at {where}{hint}"
        return f"HTTP {exc.code} {exc.reason} at {where}"
    reason = getattr(exc, "reason", exc)
    return f"offline or unreachable: cannot download {where} ({reason})"


class _SafeRedirect(urllib.request.HTTPRedirectHandler):
    """Follow redirects only to http(s), never from https down to http.
    Subclasses narrow it with `_refuse(new_url)` (a reason, or None)."""

    def _refuse(self, new: str) -> str | None:
        return None

    def http_error_302(self, req, fp, code, msg, headers):
        location = headers.get("Location") or headers.get("URI")
        if location:
            new = urllib.parse.urljoin(req.full_url, location)
            # The narrower policy first: its message names no URL (a Drive
            # redirect may carry a signed query).
            why = self._refuse(new)
            if why:
                fp.close()
                raise ProvisionError(f"refused redirect: {why}")
            scheme = urllib.parse.urlsplit(new).scheme.lower()
            old = urllib.parse.urlsplit(req.full_url).scheme.lower()
            if scheme not in ("http", "https") or (old == "https" and scheme == "http"):
                fp.close()
                raise ProvisionError(
                    f"refused redirect from {req.full_url} to {new} "
                    "(only http(s), no https -> http downgrade)"
                )
        return super().http_error_302(req, fp, code, msg, headers)

    http_error_301 = http_error_303 = http_error_307 = http_error_308 = http_error_302


_OPENER = urllib.request.build_opener(_SafeRedirect)


def _open(request: urllib.request.Request, opener=None):
    """Open `request` with `opener` (default: the plain URL policy). Every
    download goes through here (tests replace it to forbid the network)."""
    return (opener or _OPENER).open(request, timeout=HTTP_TIMEOUT_S)


class _HttpFileSource:
    """One file over HTTP(S) with resume and integrity checks (`urllib`);
    subclasses say where (`_url`, `_where`), how redirects are followed
    (`_opener`) and what a wrong answer looks like (`_check_response`,
    `_check_first_chunk`).

    A file is accepted only when its integrity is known: the declared
    `size_bytes` or `sha256` matches, or the server's `Content-Length` (or
    `Content-Range` total) was received in full. A `.part` is resumed only
    when `size_bytes` or `sha256` is declared and the server answers 206 at
    the expected offset to an `If-Range` request carrying the validator
    (ETag / Last-Modified) saved in `<target>.part.meta` when the `.part` was
    started; anything else restarts from zero.
    """

    def _url(self, asset: Asset) -> str:
        raise NotImplementedError

    def _where(self, asset: Asset) -> str:
        """How messages name the file (default: its URL)."""
        return self._url(asset)

    def _opener(self):
        return None  # the plain URL redirect policy

    def _reason(self, exc: Exception, asset: Asset) -> str:
        return _http_reason(exc, self._where(asset), asset.gated)

    def _check_response(self, asset: Asset, response) -> None:
        """Raise ProvisionError for an answer that is not the file."""

    def _check_first_chunk(self, asset: Asset, chunk: bytes) -> None:
        """Raise ProvisionError when the first bytes show it is not the file."""

    def prepare(
        self,
        asset: Asset,
        root: Path,
        progress: ItemProgress,
        cancel: threading.Event,
    ) -> None:
        url = self._url(asset)
        target = contained(root, asset.target)
        part = contained(root, asset.target + ".part")
        meta = contained(root, asset.target + ".part.meta")
        self._refuse_existing(asset, target)
        target.parent.mkdir(parents=True, exist_ok=True)
        if part.is_dir() or meta.is_dir():
            raise ProvisionError(f"{part} is a folder, expected a partial file")
        try:
            server_total = self._download(asset, url, part, meta, progress, cancel)
        except (ProvisionCancelled, ProvisionError):
            raise
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            raise ProvisionError(self._reason(exc, asset)) from None
        except http.client.HTTPException as exc:  # connection dropped mid-body
            raise ProvisionError(
                f"download of {self._where(asset)} interrupted "
                f"({type(exc).__name__}); run Prepare again to resume"
            ) from None
        self._verify(asset, part, meta, server_total)
        self._refuse_existing(asset, target)  # appeared during the download
        os.replace(part, target)
        meta.unlink(missing_ok=True)

    @staticmethod
    def _refuse_existing(asset: Asset, target: Path) -> None:
        """Never replace a file (or anything) already at the target."""
        if target.exists() or target.is_symlink():
            raise ProvisionError(
                f"{asset.target} already exists and is not the declared file "
                "(another size or sha256); it is never replaced: move or delete "
                "it, or point the config at another file"
            )

    @staticmethod
    def _check_disk(part: Path, needed: int | None) -> None:
        """ProvisionError when the folder of `part` has less than `needed`
        bytes free (unknown sizes are not checked)."""
        if not needed or needed <= 0:
            return
        try:
            free = shutil.disk_usage(part.parent).free
        except OSError:
            return
        if free < needed:
            raise ProvisionError(
                f"not enough free disk space: {needed / 1e6:.1f} MB needed, "
                f"{free / 1e6:.1f} MB free on the weights root's disk"
            )

    @staticmethod
    def _restart(part: Path, meta: Path) -> None:
        part.unlink(missing_ok=True)
        meta.unlink(missing_ok=True)

    def _download(
        self,
        asset: Asset,
        url: str,
        part: Path,
        meta: Path,
        progress: ItemProgress,
        cancel: threading.Event,
    ) -> int | None:
        """Fill `part`; returns the full size announced by the server, if any."""
        sha256 = getattr(asset, "sha256", None)
        where = self._where(asset)
        resumable = asset.size_bytes is not None or sha256 is not None
        offset = part.stat().st_size if part.is_file() else 0
        validator = meta.read_text().strip() if meta.is_file() else ""
        if offset and (
            not resumable
            or not validator
            or (asset.size_bytes is not None and offset >= asset.size_bytes)
        ):
            # Unverifiable, unvalidated or too long: never trust it, restart.
            self._restart(part, meta)
            offset = 0
        headers = {"User-Agent": "det-depth-distance"}
        if offset:
            headers["Range"] = f"bytes={offset}-"
            headers["If-Range"] = validator
        request = urllib.request.Request(url, headers=headers)
        _check_cancel(cancel)
        try:
            response = _open(request, self._opener())
        except urllib.error.HTTPError as exc:
            if exc.code != 416 or not offset:
                raise
            # Range not satisfiable: the .part is not a prefix we can trust.
            self._restart(part, meta)
            return self._download(asset, url, part, meta, progress, cancel)
        with response:
            self._check_response(asset, response)
            start, range_total = _content_range(response.headers.get("Content-Range"))
            if response.status == 206:
                if not offset or start != offset:
                    # Not the range we asked for: drop the .part, restart.
                    response.close()
                    self._restart(part, meta)
                    if not offset:
                        raise ProvisionError(
                            f"server sent a partial answer to a full request: {where}"
                        )
                    return self._download(asset, url, part, meta, progress, cancel)
                mode = "ab"
            else:
                # 200: a fresh file, or the server ignored Range / If-Range
                # failed (the remote changed): restart cleanly.
                offset, mode, range_total = 0, "wb", None
                new_validator = _validator(response.headers)
                if new_validator:
                    meta.write_text(new_validator)
                else:
                    meta.unlink(missing_ok=True)
            length = response.headers.get("Content-Length")
            server_total = range_total
            if server_total is None and length is not None and length.isdigit():
                server_total = offset + int(length)
            total = server_total if server_total is not None else asset.size_bytes
            # The most bytes accepted: the smaller of what the server announced
            # and what is declared (a stream with neither is bounded by the disk).
            limit = min(
                (n for n in (server_total, asset.size_bytes) if n is not None),
                default=None,
            )
            self._check_disk(part, None if total is None else total - offset)
            done = offset
            progress(done, total)
            first = True
            with part.open(mode) as out:
                while True:
                    _check_cancel(cancel)
                    chunk = response.read1(CHUNK)
                    if not chunk:
                        break
                    if first:
                        first = False
                        try:
                            self._check_first_chunk(asset, chunk)
                        except ProvisionError:
                            out.close()
                            self._restart(part, meta)
                            raise
                    if limit is not None and done + len(chunk) > limit:
                        out.close()
                        self._restart(part, meta)
                        raise ProvisionError(
                            f"download of {where} is larger than expected "
                            f"(more than {limit} bytes); partial file removed"
                        )
                    out.write(chunk)
                    done += len(chunk)
                    progress(done, total)
                out.flush()
                os.fsync(out.fileno())
        if server_total is not None and done != server_total:
            # Connection closed early: keep the .part, the next run resumes.
            raise ProvisionError(
                f"download of {where} interrupted at {done} of {server_total} "
                "bytes; run Prepare again to resume"
            )
        return server_total

    @staticmethod
    def _verify(asset: Asset, part: Path, meta: Path, server_total: int | None) -> None:
        size = part.stat().st_size
        sha256 = getattr(asset, "sha256", None)
        if asset.size_bytes is not None and size != asset.size_bytes:
            _HttpFileSource._restart(part, meta)
            raise ProvisionError(
                f"size mismatch: {size} bytes, expected {asset.size_bytes} "
                "(partial file removed)"
            )
        if sha256 is not None:
            digest = hashlib.sha256()
            with part.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1 << 20), b""):
                    digest.update(chunk)
            got = digest.hexdigest()
            if got != sha256.lower():
                _HttpFileSource._restart(part, meta)
                raise ProvisionError(
                    f"sha256 mismatch: expected {sha256.lower()}, got {got} "
                    "(partial file removed)"
                )
            return
        if asset.size_bytes is None and server_total != size:
            _HttpFileSource._restart(part, meta)
            raise ProvisionError(UNVERIFIABLE)


@register_source(UrlAsset)
class UrlSource(_HttpFileSource):
    """Plain HTTP(S) download of `UrlAsset.url` (see `_HttpFileSource`)."""

    def _url(self, asset: UrlAsset) -> str:
        if not asset.url.lower().startswith(("http://", "https://")):
            raise ProvisionError(f"unsupported URL scheme: {asset.url}")
        return asset.url


# --------------------------------------------------------------- Google Drive

DRIVE_DOWNLOAD = "https://drive.usercontent.google.com/download"
# Hosts a Drive download may be redirected to (exact names, then suffixes).
DRIVE_HOSTS = ("drive.usercontent.google.com", "drive.google.com")
DRIVE_HOST_SUFFIXES = (".googleusercontent.com",)
_HTML_STARTS = (b"<!doctype html", b"<html", b"<head", b"<body", b"<?xml")


class _DriveRedirect(_SafeRedirect):
    """`_SafeRedirect` plus: https only (`schemes`) and Google hosts only."""

    def __init__(self, hosts, suffixes, schemes) -> None:
        super().__init__()
        self.hosts, self.suffixes, self.schemes = hosts, suffixes, schemes

    def _refuse(self, new: str) -> str | None:
        parts = urllib.parse.urlsplit(new)
        host = (parts.hostname or "").lower()
        if parts.scheme.lower() not in self.schemes:
            return (
                f"Google Drive downloads stay on {'/'.join(self.schemes)} "
                f"(redirect to {parts.scheme or '?'}://{host or '?'})"
            )
        if host in self.hosts or host.endswith(self.suffixes):
            return None
        return f"host {host or '?'} is not a Google Drive download host"


@register_source(GoogleDriveAsset)
class GoogleDriveSource(_HttpFileSource):
    """A public Google Drive file through the direct download endpoint
    (`DRIVE_DOWNLOAD?id=<id>&export=download&confirm=t`, which skips the
    virus-scan page for files Drive does not scan), with the URL source's
    resume and integrity rules. An HTML answer is the confirmation or quota
    page, not the file: it fails with a readable message and nothing is kept.
    `endpoint`, `hosts`, `host_suffixes` and `schemes` are class attributes
    so tests can point the source at a local server."""

    endpoint = DRIVE_DOWNLOAD
    hosts = DRIVE_HOSTS
    host_suffixes = DRIVE_HOST_SUFFIXES
    schemes: tuple[str, ...] = ("https",)

    def _url(self, asset: GoogleDriveAsset) -> str:
        query = urllib.parse.urlencode(
            {"id": asset.file_id, "export": "download", "confirm": "t"}
        )
        return f"{self.endpoint}?{query}"

    def _where(self, asset: GoogleDriveAsset) -> str:
        return f"Google Drive file {asset.file_id}"

    def _opener(self):
        return urllib.request.build_opener(
            _DriveRedirect(self.hosts, self.host_suffixes, self.schemes)
        )

    def _manual(self, asset: GoogleDriveAsset) -> str:
        where = asset.manual or f"https://drive.google.com/file/d/{asset.file_id}"
        return f"download it manually from {where}"

    def _reason(self, exc: Exception, asset: GoogleDriveAsset) -> str:
        reason = super()._reason(exc, asset)
        if isinstance(exc, urllib.error.HTTPError) and exc.code in (403, 429):
            reason += f" (Drive quota or access limit: try again later or {self._manual(asset)})"
        return reason

    def _not_the_file(self, asset: GoogleDriveAsset) -> ProvisionError:
        return ProvisionError(
            "Google Drive returned a confirmation or quota page instead of the "
            f"file; try again later or {self._manual(asset)}"
        )

    def _check_response(self, asset: GoogleDriveAsset, response) -> None:
        kind = (response.headers.get("Content-Type") or "").lower()
        if kind.startswith(("text/html", "application/xhtml")):
            raise self._not_the_file(asset)

    def _check_first_chunk(self, asset: GoogleDriveAsset, chunk: bytes) -> None:
        if chunk[:512].lstrip().lower().startswith(_HTML_STARTS):
            raise self._not_the_file(asset)


# --------------------------------------------------------------- Hugging Face


def _hf_reason(exc: Exception, asset: HuggingFaceAsset) -> str:
    """A readable reason for a `huggingface_hub` error (matched by class name,
    so a renamed or missing class in another hub version is not fatal)."""
    names = {cls.__name__ for cls in type(exc).__mro__}
    page = f"https://huggingface.co/{asset.repo}"
    status = getattr(getattr(exc, "response", None), "status_code", None)
    if "GatedRepoError" in names or (asset.gated and status in (401, 403)):
        return (
            f"gated - accept the licence at {page} and set HF_TOKEN "
            "(or run `hf auth login`)"
        )
    if names & {
        "LocalEntryNotFoundError",
        "OfflineModeIsEnabled",
        "IncompleteSnapshotError",
    }:
        return f"offline or unreachable: cannot reach huggingface.co for {asset.repo}"
    if "RepositoryNotFoundError" in names:
        return f"repository {asset.repo} not found (HTTP 404) at {page}"
    if "RevisionNotFoundError" in names:
        return f"revision {asset.revision!r} of {asset.repo} not found (HTTP 404)"
    if "EntryNotFoundError" in names:
        return f"file not found in {asset.repo} (HTTP 404)"
    if "HfHubHTTPError" in names:
        return f"Hugging Face error for {asset.repo}: {exc}"
    if isinstance(exc, OSError):
        return f"offline or unreachable: {asset.repo} ({exc})"
    return f"{type(exc).__name__} while downloading {asset.repo}: {exc}"


def _hf_progress_class(progress: ItemProgress, cancel: threading.Event):
    """A `tqdm_class` for `snapshot_download`: silent, forwards the bytes of
    its "reconstruct" bar to `progress` and raises when `cancel` is set."""
    from huggingface_hub.utils import tqdm as hf_tqdm

    lock = threading.Lock()

    class _Bar(hf_tqdm):
        def __init__(self, *args, **kwargs):
            self._bytes = kwargs.get("name") == "huggingface_hub.snapshot_download"
            self._done = 0
            kwargs["disable"] = True
            super().__init__(*args, **kwargs)

        def update(self, n=1):
            _check_cancel(cancel)
            result = super().update(n)
            if self._bytes and n:
                with lock:
                    self._done += int(n)
                    progress(self._done, int(self.total) if self.total else None)
            return result

    return _Bar


@register_source(HuggingFaceAsset)
class HuggingFaceSource:
    """`snapshot_download` into the cache layout `HuggingFaceAsset` reads."""

    def prepare(
        self,
        asset: HuggingFaceAsset,
        root: Path,
        progress: ItemProgress,
        cancel: threading.Event,
    ) -> None:
        cache_dir = contained(root, asset.target)
        patterns = [*asset.files, *asset.patterns] or None
        _check_cancel(cancel)
        try:
            import huggingface_hub

            huggingface_hub.snapshot_download(
                asset.repo,
                revision=asset.revision,
                cache_dir=cache_dir,
                allow_patterns=patterns,
                tqdm_class=_hf_progress_class(progress, cancel),
            )
        except ProvisionCancelled:
            raise
        except ImportError as exc:
            raise ProvisionError(f"huggingface_hub is not installed ({exc})") from None
        except Exception as exc:  # noqa: BLE001 - the hub error zoo, mapped
            raise ProvisionError(_hf_reason(exc, asset)) from None
        _check_cancel(cancel)


# --------------------------------------------------------------- git


def _git_env() -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    env["GIT_TERMINAL_PROMPT"] = "0"  # never wait for credentials
    env["GIT_ALLOW_PROTOCOL"] = "file:git:http:https:ssh"  # no ext:: transport
    return env


def _kill(proc: subprocess.Popen) -> None:
    """Kill git and the helpers it spawned (remote-https, upload-pack...), which
    would otherwise keep the pipes open and block `communicate`."""
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except (AttributeError, ProcessLookupError, PermissionError):
        proc.kill()
    proc.communicate()


def _git(args: list[str], cancel: threading.Event, what: str) -> str:
    """Run git, killing it when `cancel` is set or after `GIT_TIMEOUT_S`;
    ProvisionError on failure or timeout."""
    try:
        proc = subprocess.Popen(
            [GIT, *args],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            stdin=subprocess.DEVNULL,
            text=True,
            env=_git_env(),
            start_new_session=True,  # its own process group, killed as a whole
        )
    except OSError as exc:
        raise ProvisionError(f"git is not available ({exc})") from None
    deadline = time.monotonic() + GIT_TIMEOUT_S
    try:
        while True:
            try:
                out, err = proc.communicate(timeout=GIT_POLL_S)
                break
            except subprocess.TimeoutExpired:
                if cancel.is_set():
                    _kill(proc)
                    raise ProvisionCancelled() from None
                if time.monotonic() > deadline:
                    _kill(proc)
                    raise ProvisionError(
                        f"{what} timed out after {GIT_TIMEOUT_S:g} s"
                    ) from None
    finally:
        if proc.poll() is None:  # KeyboardInterrupt or another error
            _kill(proc)
    if proc.returncode != 0:
        lines = [line for line in err.strip().splitlines() if line.strip()]
        detail = lines[-1] if lines else f"exit code {proc.returncode}"
        raise ProvisionError(f"{what} failed: {detail}")
    return out.strip()


# The pin is validated by CodeRepo (hex only); `--end-of-options` is a second
# guard so a value can never be parsed as a git option.
def _has_commit(path: Path, commit: str, cancel: threading.Event) -> bool:
    try:
        _git(
            [
                "-C",
                str(path),
                "cat-file",
                "-e",
                "--end-of-options",
                f"{commit}^{{commit}}",
            ],
            cancel,
            "cat-file",
        )
    except ProvisionError:
        return False
    return True


def _checkout(path: Path, repo: CodeRepo, cancel: threading.Event) -> None:
    if not _has_commit(path, repo.commit, cancel):
        _git(["-C", str(path), "fetch", "--quiet", "origin"], cancel, "git fetch")
    if not _has_commit(path, repo.commit, cancel):
        # A commit outside every branch: ask for it directly (servers may refuse).
        _git(
            [
                "-C",
                str(path),
                "fetch",
                "--quiet",
                "--end-of-options",
                "origin",
                repo.commit,
            ],
            cancel,
            f"git fetch of commit {repo.commit}",
        )
    _git(
        [
            "-C",
            str(path),
            "switch",  # unlike checkout, switch honours --end-of-options
            "--quiet",
            "--detach",
            "--end-of-options",
            repo.commit,
        ],
        cancel,
        f"git switch to {repo.commit}",
    )


@register_source(CodeRepo)
class GitSource:
    """Pinned clone with the `git` CLI under `<root>/_code/<name>`."""

    def prepare(
        self,
        repo: CodeRepo,
        root: Path,
        progress: ItemProgress,
        cancel: threading.Event,
    ) -> None:
        path = repo.path(root)
        progress(0, None)
        if path.exists():
            if not (path / ".git").exists():
                raise ProvisionError(
                    f"{path} exists but is not a git clone; move it away and retry"
                )
            _checkout(path, repo, cancel)  # never deleted, only moved to the pin
            return
        tmp = contained(root, f"{CODE_DIR}/{repo.name}.part")
        if tmp.exists():  # our own interrupted clone
            shutil.rmtree(tmp)
        tmp.parent.mkdir(parents=True, exist_ok=True)
        try:
            _git(
                ["clone", "--quiet", "--no-checkout", "--", repo.url, str(tmp)],
                cancel,
                f"git clone of {repo.url}",
            )
            _checkout(tmp, repo, cancel)
        except BaseException:
            shutil.rmtree(tmp, ignore_errors=True)
            raise
        os.replace(tmp, path)
        progress(1, 1)


# --------------------------------------------------------------- user-provided


@register_source(UserProvided)
class UserProvidedSource:
    """Never downloads: the file can only come from the user."""

    def prepare(
        self,
        asset: UserProvided,
        root: Path,
        progress: ItemProgress,
        cancel: threading.Event,
    ) -> None:
        where = contained(root, asset.target)
        raise ProvisionError(
            f"provide this file: {asset.hint} (expected at {where}, or set the "
            f"option {asset.option!r})"
        )
