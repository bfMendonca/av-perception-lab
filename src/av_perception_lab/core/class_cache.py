"""Load-time class cache: a hint of the classes a checkpoint emits.

YOLO and RF-DETR checkpoints are torch pickles, so their class names are only
known after loading. Each successful load records them as a small JSON manifest
under `<weights root>/.classes/<key>.json`; a later run can then declare them
without importing any framework. The key hashes the weights file's resolved
path, size and mtime, so a replaced file is a miss. The cache is only a hint
(the loaded model's `class_names` stays authoritative): a corrupt file is a
miss and a write failure is logged and ignored."""

import hashlib
import json
import logging
from collections.abc import Sequence
from pathlib import Path

from av_perception_lab.core.requirements import weights_root

log = logging.getLogger(__name__)

CACHE_DIR = ".classes"


def cache_key(weights: Path) -> str | None:
    """Hash of the resolved path, size and `st_mtime_ns`; None if the file is missing."""
    try:
        path = Path(weights).expanduser().resolve()
        stat = path.stat()
    except OSError:
        return None
    raw = f"{path}\0{stat.st_size}\0{stat.st_mtime_ns}"
    return hashlib.sha256(raw.encode()).hexdigest()[:32]


def _cache_file(key: str) -> Path:
    return weights_root() / CACHE_DIR / f"{key}.json"


def read(weights: Path) -> tuple[str, ...] | None:
    """Class names recorded for this exact weights file, or None (never raises)."""
    key = cache_key(weights)
    if key is None:
        return None
    try:
        data = json.loads(_cache_file(key).read_text(encoding="utf-8"))
        names = data["names"]
        if not isinstance(names, list) or not all(isinstance(n, str) for n in names):
            return None
        return tuple(names)
    except (OSError, ValueError, KeyError, TypeError):
        return None


def write(weights: Path, names: Sequence[str]) -> None:
    """Record `names` for this weights file; logs and returns on `OSError`."""
    key = cache_key(weights)
    if key is None:
        return
    target = _cache_file(key)
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(".tmp")
        tmp.write_text(json.dumps({"names": list(names)}), encoding="utf-8")
        tmp.replace(target)
    except OSError as exc:
        log.warning("class cache not written (%s): %s", target, exc)
