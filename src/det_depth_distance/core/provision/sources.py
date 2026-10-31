"""Built-in sources: Hugging Face, plain URL, git clone, user-provided.

Every write path is resolved through `core.requirements.contained` before
anything is written, so a target, a `.part` file or a clone folder cannot
leave the weights root. Only the standard library is imported here;
`huggingface_hub` is imported inside the Hugging Face source.

- `HuggingFaceAsset`: `huggingface_hub.snapshot_download` into
  `<root>/<target>` as a cache dir (the layout `HuggingFaceAsset.present`
  reads), restricted to the declared files/patterns. The hub keeps its own
  `.incomplete` blobs (never ready) and resumes them.
- `UrlAsset`: streams to `<target>.part`, resumes with a `Range` header when
  the server answers 206 (a 200 restarts cleanly), checks the size and
  `sha256` when declared (a mismatch deletes the `.part`, so the next run
  restarts), then renames atomically. A cancel keeps the `.part` for resume.
- `CodeRepo`: `git clone --no-checkout` into `<name>.part`, `checkout` of the
  pinned commit, rename. An existing clone at another commit is fetched and
  checked out, never deleted; a folder that is not a clone is left alone.
- `UserProvided`: never downloads; raises with the declared hint.

Google Drive is not implemented: a Drive source would be one more class
registered with `register_source` for a `GoogleDriveAsset` kind.
"""

import hashlib
import http.client
import os
import shutil
import subprocess
import threading
import urllib.error
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
    CodeRepo,
    HuggingFaceAsset,
    UrlAsset,
    UserProvided,
    contained,
)

CHUNK = 1 << 16  # bytes per read; the cancel event is checked between reads
HTTP_TIMEOUT_S = 30.0
GIT_POLL_S = 0.1


def _check_cancel(cancel: threading.Event) -> None:
    if cancel.is_set():
        raise ProvisionCancelled()


# --------------------------------------------------------------- URL


def _content_range_start(header: str | None) -> int | None:
    """Start offset of a `Content-Range: bytes a-b/n` header."""
    if not header or not header.startswith("bytes "):
        return None
    try:
        return int(header[6:].split("-", 1)[0])
    except ValueError:
        return None


def _content_range_total(header: str | None) -> int | None:
    if not header or "/" not in header:
        return None
    tail = header.rsplit("/", 1)[1]
    return int(tail) if tail.isdigit() else None


def _http_reason(exc: Exception, asset: UrlAsset) -> str:
    if isinstance(exc, urllib.error.HTTPError):
        if exc.code == 404:
            return f"not found (HTTP 404) at {asset.url}"
        if exc.code in (401, 403):
            hint = (
                " - the file is gated: accept the licence on the source site"
                if asset.gated
                else ""
            )
            return f"access denied (HTTP {exc.code}) at {asset.url}{hint}"
        return f"HTTP {exc.code} {exc.reason} at {asset.url}"
    reason = getattr(exc, "reason", exc)
    return f"offline or unreachable: cannot download {asset.url} ({reason})"


@register_source(UrlAsset)
class UrlSource:
    """Plain HTTP(S) download with resume and checksum (stdlib `urllib`)."""

    def prepare(
        self,
        asset: UrlAsset,
        root: Path,
        progress: ItemProgress,
        cancel: threading.Event,
    ) -> None:
        if not asset.url.lower().startswith(("http://", "https://")):
            raise ProvisionError(f"unsupported URL scheme: {asset.url}")
        target = contained(root, asset.target)
        part = contained(root, asset.target + ".part")
        target.parent.mkdir(parents=True, exist_ok=True)
        if part.is_dir():
            raise ProvisionError(f"{part} is a folder, expected a partial file")
        try:
            self._download(asset, part, progress, cancel)
        except ProvisionCancelled:
            raise
        except ProvisionError:
            raise
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            raise ProvisionError(_http_reason(exc, asset)) from None
        except http.client.HTTPException as exc:  # connection dropped mid-body
            raise ProvisionError(
                f"download of {asset.url} interrupted ({type(exc).__name__}); "
                "run Prepare again to resume"
            ) from None
        self._verify(asset, part)
        os.replace(part, target)

    def _download(
        self,
        asset: UrlAsset,
        part: Path,
        progress: ItemProgress,
        cancel: threading.Event,
    ) -> None:
        offset = part.stat().st_size if part.is_file() else 0
        if asset.size_bytes is not None and offset > asset.size_bytes:
            offset = 0  # longer than the file: stale, restart
        headers = {"User-Agent": "det-depth-distance"}
        if offset:
            headers["Range"] = f"bytes={offset}-"
        request = urllib.request.Request(asset.url, headers=headers)
        _check_cancel(cancel)
        try:
            response = urllib.request.urlopen(request, timeout=HTTP_TIMEOUT_S)
        except urllib.error.HTTPError as exc:
            if exc.code != 416 or not offset:
                raise
            # Range not satisfiable: the .part is not a prefix we can trust.
            part.unlink()
            return self._download(asset, part, progress, cancel)
        with response:
            mode = "wb"
            if offset and response.status == 206:
                start = _content_range_start(response.headers.get("Content-Range"))
                if start != offset:
                    # Not the range we asked for: drop the .part, restart.
                    response.close()
                    part.unlink()
                    return self._download(asset, part, progress, cancel)
                mode = "ab"
            else:
                offset = 0  # 200: the server ignored Range, restart cleanly
            length = response.headers.get("Content-Length")
            total = _content_range_total(response.headers.get("Content-Range"))
            if total is None and length is not None and length.isdigit():
                total = offset + int(length)
            if total is None:
                total = asset.size_bytes
            done = offset
            progress(done, total)
            with part.open(mode) as out:
                while True:
                    _check_cancel(cancel)
                    chunk = response.read1(CHUNK)
                    if not chunk:
                        break
                    out.write(chunk)
                    done += len(chunk)
                    progress(done, total)
                out.flush()
                os.fsync(out.fileno())
        if total is not None and done != total:
            # Connection closed early: keep the .part, the next run resumes.
            raise ProvisionError(
                f"download of {asset.url} interrupted at {done} of {total} bytes; "
                "run Prepare again to resume"
            )

    @staticmethod
    def _verify(asset: UrlAsset, part: Path) -> None:
        size = part.stat().st_size
        if asset.size_bytes is not None and size != asset.size_bytes:
            part.unlink()
            raise ProvisionError(
                f"size mismatch: {size} bytes, expected {asset.size_bytes} "
                "(partial file removed)"
            )
        if asset.sha256 is not None:
            digest = hashlib.sha256()
            with part.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1 << 20), b""):
                    digest.update(chunk)
            got = digest.hexdigest()
            if got != asset.sha256.lower():
                part.unlink()
                raise ProvisionError(
                    f"sha256 mismatch: expected {asset.sha256.lower()}, got {got} "
                    "(partial file removed)"
                )


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


def _git(args: list[str], cancel: threading.Event, what: str) -> str:
    """Run git, killing it when `cancel` is set; ProvisionError on failure."""
    try:
        proc = subprocess.Popen(
            ["git", *args],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            stdin=subprocess.DEVNULL,
            text=True,
            env=_git_env(),
        )
    except OSError as exc:
        raise ProvisionError(f"git is not available ({exc})") from None
    try:
        while True:
            try:
                out, err = proc.communicate(timeout=GIT_POLL_S)
                break
            except subprocess.TimeoutExpired:
                if cancel.is_set():
                    proc.kill()
                    proc.communicate()
                    raise ProvisionCancelled() from None
    finally:
        if proc.poll() is None:  # KeyboardInterrupt or another error
            proc.kill()
            proc.communicate()
    if proc.returncode != 0:
        lines = [line for line in err.strip().splitlines() if line.strip()]
        detail = lines[-1] if lines else f"exit code {proc.returncode}"
        raise ProvisionError(f"{what} failed: {detail}")
    return out.strip()


def _has_commit(path: Path, commit: str, cancel: threading.Event) -> bool:
    try:
        _git(
            ["-C", str(path), "cat-file", "-e", f"{commit}^{{commit}}"],
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
            ["-C", str(path), "fetch", "--quiet", "origin", repo.commit],
            cancel,
            f"git fetch of commit {repo.commit}",
        )
    _git(
        ["-C", str(path), "checkout", "--quiet", "--detach", repo.commit],
        cancel,
        f"git checkout {repo.commit}",
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
