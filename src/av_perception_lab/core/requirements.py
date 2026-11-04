"""What a component needs to run, and a status check that never imports torch.

A component class declares a `Requirements` (see `Component.requirements`) in
three layers: *assets* (weights and other files under the weights root),
*code* (pinned git clones under `<root>/_code/<name>`) and *packages* (import
names provided by a `pyproject.toml` extra). `check` aggregates the layers into
a `ModelStatus` using only the file system, `git rev-parse` and
`importlib.util.find_spec`, so the server and the UI work on a fresh clone
without torch.

Asset kinds are registered classes (`register_asset_kind`): each implements
`present(root) -> AssetState`, so a new kind (a new source type) is one class
and never an edit of `check`. Kinds are pure stdlib; the Hugging Face kind
reads the HF cache layout on disk (`models--org--name/snapshots/<rev>/...`)
itself instead of importing `huggingface_hub`. Every path is resolved and
confined to the weights root (`..`, absolute targets and symlinks that leave
it raise `UnsafePathError`); a `.part` / `.incomplete` file is never ready.
An existing file that is not the declared one (another size or sha256: a
fine-tuned checkpoint saved under a release name) is "invalid", never
"partial": nothing downloads over it.
"""

import fnmatch
import hashlib
import importlib.util
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path, PurePath
from typing import ClassVar, Literal
from urllib.parse import urlsplit

CODE_DIR = "_code"  # clones live in <weights root>/_code/<name>
WEIGHTS_ROOT_ENV = "AVPL_WEIGHTS_ROOT"
_PARTIAL_SUFFIXES = (".part", ".incomplete")

AssetStateName = Literal["ready", "missing", "partial", "invalid"]


class UnsafePathError(ValueError):
    """A target or folder name that would leave the weights root."""


@dataclass(frozen=True, slots=True)
class AssetState:
    """Result of `Asset.present`: "ready", "missing" (absent), "partial"
    (still downloading: a `.part`, an unfinished snapshot or clone) or
    "invalid" (an unsafe path, or an existing file that is not the declared
    one: it is never replaced, the user moves it)."""

    state: AssetStateName
    detail: str = ""


def contained(root: Path, relative: str) -> Path:
    """`root / relative` resolved (symlinks included) and checked to stay inside
    the resolved `root`; raises `UnsafePathError` otherwise."""
    rel = PurePath(relative)
    if not relative or rel.is_absolute() or ".." in rel.parts:
        raise UnsafePathError(f"unsafe path {relative!r}: must stay inside the root")
    base = Path(root).resolve()
    resolved = (base / rel).resolve()
    if resolved == base or not resolved.is_relative_to(base):
        raise UnsafePathError(f"unsafe path {relative!r}: resolves outside the root")
    return resolved


@dataclass(frozen=True, kw_only=True)
class Asset:
    """A file or folder a model needs, `target` relative to the weights root.

    Subclasses are registered with `register_asset_kind` and implement
    `present`; `downloadable` is False for kinds the engine must never fetch
    (it still calls their source, which raises "provide this file: ...", and
    reports the item as missing, not failed). `known` fails closed: False
    unless the adapter's code sets it, which it does only when both the source
    (a table in its code) and the target (the canonical location under the
    weights root) are its own; anything derived from a user option (a
    Hugging Face repo id typed in the YAML, a `weights` path or `hf_cache`
    folder of the client's choice) stays False. The CLI prepares both; the
    web server only prepares known sources. `known` does not apply to kinds
    that are not downloadable.
    """

    kind_name: ClassVar[str] = ""  # set by register_asset_kind
    downloadable: ClassVar[bool] = True

    key: str
    target: str
    size_bytes: int | None = None
    license: str | None = None
    gated: bool = False
    known: bool = False

    def path(self, root: Path) -> Path:
        """Absolute, confined location of `target`."""
        return contained(root, self.target)

    def present(self, root: Path) -> AssetState:
        raise NotImplementedError(f"{type(self).__name__}.present")

    def describe_source(self) -> str:
        """One human-readable line naming where the asset comes from (shown
        by the UI); kinds override it, the default is the kind name."""
        return self.kind_name or type(self).__name__


_ASSET_KINDS: dict[str, type[Asset]] = {}


def register_asset_kind(name: str):
    """Class decorator registering an `Asset` subclass under `name`."""

    def deco(cls: type[Asset]) -> type[Asset]:
        if name in _ASSET_KINDS:
            raise ValueError(f"asset kind {name!r} is already registered")
        cls.kind_name = name
        _ASSET_KINDS[name] = cls
        return cls

    return deco


def asset_kinds() -> dict[str, type[Asset]]:
    """Registered asset kinds by name (a copy)."""
    return dict(_ASSET_KINDS)


_REVISION_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
# A pin is a commit id (abbreviated or full; SHA-1 or SHA-256): a branch or tag
# moves, and a leading "-" would be read as a git option.
_COMMIT_RE = re.compile(r"[0-9a-fA-F]{7,64}")


def _check_revision(value: str, what: str = "revision") -> None:
    """Branch, tag or commit names only: no separators, no `..`, not absolute."""
    if not _REVISION_RE.fullmatch(value) or ".." in value:
        raise UnsafePathError(f"unsafe {what} {value!r}")


# What to do about an existing file that is not the declared one.
_MOVE_IT = "move or delete it, or point the config at another file"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _single_file_state(asset: Asset, sha256: str | None, root: Path) -> AssetState:
    """`present` of a single downloaded file (URL, Google Drive): an existing
    target must match the declared size and sha256 (else "invalid": it is the
    user's, never downloaded over); a `.part` alone is "partial"."""
    path = asset.path(root)
    # An existing target is checked first: one that is not the declared
    # file is the user's, never "partial" (which would download over it).
    if path.exists() or path.is_symlink():
        if not path.is_file():
            return AssetState(
                "invalid",
                f"{asset.key}: {asset.target} exists and is not a file; {_MOVE_IT}",
            )
        size = path.stat().st_size
        if asset.size_bytes is not None and size != asset.size_bytes:
            return AssetState(
                "invalid",
                f"{asset.key}: exists with a different size at {asset.target} "
                f"({size} bytes, expected {asset.size_bytes}); {_MOVE_IT}",
            )
        if sha256 is not None and _sha256(path) != sha256.lower():
            return AssetState(
                "invalid",
                f"{asset.key}: exists with a different sha256 at "
                f"{asset.target}; {_MOVE_IT}",
            )
        return AssetState("ready")
    if path.with_name(path.name + ".part").exists():
        return AssetState("partial", f"{asset.key}: unfinished download (.part)")
    return AssetState("missing", f"{asset.key}: {asset.target} not found")


@register_asset_kind("url")
@dataclass(frozen=True, kw_only=True)
class UrlAsset(Asset):
    """One file fetched from `url`; `size_bytes` and `sha256`, when known,
    must match for the file to count as ready."""

    url: str
    sha256: str | None = None

    def describe_source(self) -> str:
        return urlsplit(self.url).hostname or self.url

    def present(self, root: Path) -> AssetState:
        return _single_file_state(self, self.sha256, root)


# A Drive file id: letters, digits, "_" and "-" only (never a URL or a query).
_DRIVE_ID_RE = re.compile(r"[A-Za-z0-9_-]{10,}")


@register_asset_kind("gdrive")
@dataclass(frozen=True, kw_only=True)
class GoogleDriveAsset(Asset):
    """One public Google Drive file, by `file_id`; downloaded like a URL
    asset (see `provision.sources.GoogleDriveSource`) and ready under the
    same rules: `size_bytes` and `sha256`, when declared, must match.
    `manual` is shown when Drive answers with a confirmation or quota page
    (where to download the file by hand, e.g. the upstream folder)."""

    file_id: str
    sha256: str | None = None
    manual: str | None = None

    def __post_init__(self) -> None:
        if not _DRIVE_ID_RE.fullmatch(self.file_id):
            raise UnsafePathError(
                f"{self.key}: Google Drive file id {self.file_id!r} must be 10 or "
                "more letters, digits, '_' or '-'"
            )

    def describe_source(self) -> str:
        return f"Google Drive file {self.file_id}"

    def present(self, root: Path) -> AssetState:
        return _single_file_state(self, self.sha256, root)


@register_asset_kind("user")
@dataclass(frozen=True, kw_only=True)
class UserProvided(Asset):
    """A file or folder only the user can supply (never downloaded);
    `option` names the YAML option that overrides its location."""

    downloadable: ClassVar[bool] = False

    option: str
    hint: str

    def describe_source(self) -> str:
        return self.hint

    def present(self, root: Path) -> AssetState:
        if self.path(root).exists():
            return AssetState("ready")
        return AssetState("missing", f"{self.key}: {self.hint}")


@register_asset_kind("huggingface")
@dataclass(frozen=True, kw_only=True)
class HuggingFaceAsset(Asset):
    """Files of a Hugging Face repo in a cache folder (`target`, like
    `weights/hf`). `files` are exact names, `patterns` fnmatch globs; both
    empty means any file of the snapshot. `size_bytes` is a lower bound for
    the files found. The cache layout is read directly (no hub import)."""

    repo: str
    files: tuple[str, ...] = ()
    patterns: tuple[str, ...] = ()
    revision: str | None = None

    def __post_init__(self) -> None:
        if self.revision is not None:
            _check_revision(self.revision)

    def describe_source(self) -> str:
        return f"Hugging Face · {self.repo}"

    def _snapshots(self, repo_dir: Path) -> list[Path]:
        snaps = repo_dir / "snapshots"
        if self.revision is not None:
            _check_revision(self.revision)
            ref = repo_dir / "refs" / self.revision
            rev = ref.read_text().strip() if ref.is_file() else self.revision
            _check_revision(rev, "ref content")  # a hostile ref file is not trusted
            snap = snaps / rev
            if not snap.is_dir():
                return []
            if not snap.resolve().is_relative_to(repo_dir.resolve()):
                raise UnsafePathError(f"snapshot {rev!r} resolves outside the cache")
            return [snap]
        return (
            sorted(p for p in snaps.glob("*") if p.is_dir()) if snaps.is_dir() else []
        )

    def _satisfied(self, snap: Path) -> tuple[bool, list[str], int]:
        found = {
            p.relative_to(snap).as_posix(): p
            for p in snap.rglob("*")
            if p.is_file()  # follows symlinks: a dangling blob link is not a file
        }
        missing = [f for f in self.files if f not in found]
        missing += [p for p in self.patterns if not fnmatch.filter(found, p)]
        if not self.files and not self.patterns and not found:
            missing.append("*")
        everything = not (self.files or self.patterns)
        wanted = [
            n
            for n in found
            if everything
            or n in self.files
            or any(fnmatch.fnmatch(n, p) for p in self.patterns)
        ]
        return not missing, missing, sum(found[n].stat().st_size for n in wanted)

    def present(self, root: Path) -> AssetState:
        repo_dir = self.path(root) / ("models--" + self.repo.replace("/", "--"))
        if not repo_dir.is_dir():
            return AssetState(
                "missing", f"{self.key}: {self.repo} not in {self.target}"
            )
        for p in repo_dir.rglob("*"):
            if p.name.endswith(_PARTIAL_SUFFIXES):
                return AssetState(
                    "partial", f"{self.key}: unfinished download ({p.name})"
                )
        best: list[str] | None = None
        for snap in self._snapshots(repo_dir):
            ok, missing, size = self._satisfied(snap)
            if ok:
                if self.size_bytes is not None and size < self.size_bytes:
                    return AssetState(
                        "partial",
                        f"{self.key}: {size} bytes, expected {self.size_bytes}",
                    )
                return AssetState("ready")
            best = missing if best is None or len(missing) < len(best) else best
        if best is None:
            return AssetState("missing", f"{self.key}: no snapshot of {self.repo}")
        return AssetState("partial", f"{self.key}: missing {', '.join(best)}")


@dataclass(frozen=True, kw_only=True)
class CodeRepo:
    """A git clone pinned to `commit` (7-64 hex characters) at
    `<root>/_code/<name>`. `known` fails closed like `Asset.known`: True only
    for a repo and folder fixed in the adapter's code."""

    name: str
    url: str
    commit: str
    known: bool = False

    def __post_init__(self) -> None:
        n = self.name
        if (
            not n
            or n in (".", "..")
            or "/" in n
            or "\\" in n
            or "\0" in n
            or PurePath(n).is_absolute()
        ):
            raise UnsafePathError(f"unsafe code folder name {n!r}: one plain component")
        if not _COMMIT_RE.fullmatch(self.commit):
            raise UnsafePathError(
                f"code {n}: commit {self.commit!r} must be 7-64 hex characters "
                "(a pinned commit id, not a branch or tag)"
            )

    def path(self, root: Path) -> Path:
        return contained(root, f"{CODE_DIR}/{self.name}")

    def describe_source(self) -> str:
        """`git · host/path` (never the URL's credentials or query)."""
        parts = urlsplit(self.url)
        if not parts.hostname:
            return "git"
        port = f":{parts.port}" if parts.port else ""
        return f"git · {parts.hostname}{port}{parts.path}"

    def present(self, root: Path) -> AssetState:
        path = self.path(root)
        if not path.is_dir():
            return AssetState("missing", f"code {self.name}: not cloned")
        if not (path / ".git").exists():
            return AssetState("partial", f"code {self.name}: not a git clone")
        env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
        try:
            out = subprocess.run(
                ["git", "-C", str(path), "rev-parse", "HEAD"],
                check=False,
                capture_output=True,
                text=True,
                timeout=30,
                env=env,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            return AssetState("partial", f"code {self.name}: git failed ({exc})")
        head = out.stdout.strip()
        if out.returncode != 0 or not head:
            return AssetState("partial", f"code {self.name}: cannot read HEAD")
        pinned = self.commit.lower()
        if head == pinned or (len(pinned) >= 7 and head.startswith(pinned)):
            return AssetState("ready")
        return AssetState(
            "partial", f"code {self.name}: at {head[:12]}, pinned {pinned[:12]}"
        )


@dataclass(frozen=True, kw_only=True)
class Requirements:
    """Everything a component needs; `packages` are import names and `extra`
    the `pyproject` extra that provides them."""

    assets: tuple[Asset, ...] = ()
    code: tuple[CodeRepo, ...] = ()
    packages: tuple[str, ...] = ()
    extra: str | None = None


_explicit_root: Path | None = None


def set_weights_root(path: Path | None) -> None:
    """Fix the weights root for this process (None restores the defaults)."""
    global _explicit_root
    _explicit_root = None if path is None else Path(path)


def weights_root() -> Path:
    """Explicit setter, else `AVPL_WEIGHTS_ROOT`, else the repo `weights/`."""
    if _explicit_root is not None:
        return _explicit_root
    env = os.environ.get(WEIGHTS_ROOT_ENV)
    if env:
        return Path(env)
    return Path(__file__).resolve().parents[3] / "weights"


def _is_within(path: Path, folder: Path) -> bool:
    return path == folder or path.is_relative_to(folder)


def weights_root_problem(root: Path, others: tuple[Path, ...] = ()) -> str | None:
    """Why `root` cannot be the weights root (None when it can). Prepare
    writes and the status walks everything under it, so it may not be `/`,
    the home folder, or contain the project folder, the active Python
    environment or any of `others` (the run outputs, the video library)."""
    real = Path(root).expanduser().resolve()
    if real == Path(real.anchor):
        return f"weights root {root} is the file system root: use a dedicated folder"
    if real == Path.home().resolve():
        return f"weights root {root} is the home folder: use a dedicated folder"
    project = Path(__file__).resolve().parents[3]
    guarded = [
        (project, "the project folder"),
        (Path(sys.prefix).resolve(), "the active Python environment"),
        *((Path(o).expanduser().resolve(), str(o)) for o in others),
    ]
    for folder, what in guarded:
        if _is_within(folder, real):
            return (
                f"weights root {root} contains {what}: use a dedicated folder "
                "(for example weights/ inside the project)"
            )
    return None


# A path ends at the end of the text, a separator or a delimiter (a "." only
# when it ends a sentence); it starts at the beginning or after a delimiter.
_PATH_BEFORE = r"(?<![^\s'\"(\[=:,;`<>{}])"
_PATH_AFTER = r"(?=$|[/\s'\"),;:\]`<>{}]|\.(?:\s|$))"


def redact_root(
    text: str | None, root: Path, label: str = "<weights root>"
) -> str | None:
    """`text` with the weights root's absolute path (as given made absolute,
    and resolved) replaced by `label`: one pass, only where the whole path
    appears (`/x/weights-old` is not `/x/weights`); a relative form such as
    the bare word `weights` is never touched."""
    if not text:
        return text
    base = Path(root).expanduser()
    forms = {str(base.absolute()), str(base.resolve())} - {"/", ""}
    if not forms:
        return text
    alternatives = "|".join(re.escape(f) for f in sorted(forms, key=len, reverse=True))
    return re.sub(f"{_PATH_BEFORE}(?:{alternatives}){_PATH_AFTER}", label, text)


@dataclass(frozen=True, slots=True)
class ModelStatus:
    """Readiness of a component. `check` only produces "ready", "missing" and
    "failed" (unsafe declaration); "conflict" and "preparing" are set by the
    provisioning layers. `layers` maps assets/code/packages to
    ready|missing|partial|invalid."""

    state: Literal["ready", "missing", "conflict", "failed", "preparing"]
    layers: dict[str, str]
    missing: tuple[str, ...]
    size_bytes: int | None
    message: str | None


_RANK = ("ready", "missing", "partial", "invalid")


def _worst(states: list[str]) -> str:
    return max(states, key=_RANK.index, default="ready")


def _top_level_present(name: str) -> bool:
    # find_spec("a.b") imports "a"; the top-level name never imports anything.
    try:
        return importlib.util.find_spec(name.split(".")[0]) is not None
    except (ImportError, ValueError):
        return False


def check(req: Requirements, root: Path | None = None) -> ModelStatus:
    """Readiness of `req` under `root` (default `weights_root()`), without
    importing torch or any declared package."""
    root = weights_root() if root is None else Path(root)
    missing: list[str] = []
    details: list[str] = []
    unknown_size = True
    size = 0

    def run(label: str, probe) -> str:
        nonlocal unknown_size, size
        try:
            result = probe()
        except (UnsafePathError, ValueError) as exc:
            result = AssetState("invalid", f"{label}: {exc}")
        except (OSError, RuntimeError) as exc:  # unreadable, symlink loop...
            result = AssetState("partial", f"{label}: cannot inspect ({exc})")
        if result.state != "ready":
            missing.append(label)
            if result.detail:
                details.append(result.detail)
        return result.state

    asset_states = []
    for asset in req.assets:
        state = run(asset.key, lambda a=asset: a.present(root))
        asset_states.append(state)
        if state != "ready" and asset.size_bytes is not None:
            unknown_size = False
            size += asset.size_bytes
    code_states = [run(f"code:{c.name}", lambda c=c: c.present(root)) for c in req.code]
    package_states = []
    for pkg in req.packages:
        found = _top_level_present(pkg)
        package_states.append("ready" if found else "missing")
        if not found:
            hint = f" (extra {req.extra!r})" if req.extra else ""
            missing.append(f"package:{pkg}")
            details.append(f"package {pkg} is not installed{hint}")

    layers = {
        "assets": _worst(asset_states),
        "code": _worst(code_states),
        "packages": _worst(package_states),
    }
    worst = _worst(list(layers.values()))
    state = (
        "ready" if worst == "ready" else "failed" if worst == "invalid" else "missing"
    )
    return ModelStatus(
        state=state,
        layers=layers,
        missing=tuple(missing),
        size_bytes=None if unknown_size else size,
        message="; ".join(details) or None,
    )
