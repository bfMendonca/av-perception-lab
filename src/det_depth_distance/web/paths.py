"""Filesystem roots the web server may touch, and the traversal guard.

Ids and paths coming from the client are only ever resolved through
`safe_join`, so they cannot leave their root (`../`, absolute paths,
symlinks).
"""

from dataclasses import dataclass
from pathlib import Path

# Under out_root: one JSON record per web job (written by web.jobs, read by
# web.history, which must not import the queue).
JOBS_DIR = ".web_jobs"
# Environment variable the queue passes the library video id to the worker in.
VIDEO_ID_ENV = "DDD_WEB_VIDEO_ID"


@dataclass(frozen=True, slots=True)
class WebPaths:
    """Roots of the server: run folders, uploaded videos, importable folders
    and the folders a run's other files (weights, labels...) may come from."""

    out_root: Path  # run folders (output.out_root of the configs)
    library_root: Path  # uploaded videos
    data_roots: tuple[Path, ...] = ()  # server folders videos may be imported from
    # --asset-root: when set, every other path field of a run config must
    # resolve inside one of them (empty = anywhere outside the client-writable
    # library and out_root).
    asset_roots: tuple[Path, ...] = ()


def safe_join(root: Path, rel: str) -> Path:
    """`root / rel`, resolved; `rel` must be relative and stay inside `root`.

    Raises:
        ValueError: `rel` is absolute or resolves (through `..` or a symlink)
            outside `root`.
    """
    if Path(rel).is_absolute():
        raise ValueError(f"{rel!r}: absolute paths are not allowed")
    base = root.resolve()
    target = (base / rel).resolve()
    if target != base and not target.is_relative_to(base):
        raise ValueError(f"{rel!r}: resolves outside {root}")
    return target
