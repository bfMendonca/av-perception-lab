"""Provisioning engine: bring a `Requirements` to ready with registered sources.

A *source* knows how to obtain one asset kind (or `CodeRepo`) under the
weights root; it is registered for that class with `register_source`, so a
new source kind (Google Drive, an S3 bucket...) is one class plus one
decorator and never an edit of `prepare`. Built-in sources live in
`sources.py` and are registered when the package is imported.

`prepare` is framework-free and runs in the calling thread (a web layer wraps
it in its own thread): it checks first (a ready requirement is a no-op, no
source is called), runs the source of every non-ready code repo and asset,
then checks again. Sources raise `ProvisionError` with a readable reason
(offline, HTTP 404, gated, sha256 mismatch); `prepare` never raises for those
and returns a `ModelStatus` whose state is "failed" with the reasons joined in
`message`. Cancellation is a `threading.Event` checked by the sources between
chunks: `prepare` then stops and returns the current status with a message
starting with "cancelled" (a partial download is never ready, see
`core.requirements`). The packages layer is reported, not installed.
"""

import threading
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any, Protocol

from det_depth_distance.core.requirements import (
    AssetState,
    ModelStatus,
    Requirements,
    UnsafePathError,
    check,
    weights_root,
)

# progress(done, total) of one item, in bytes when known; total None = unknown
ItemProgress = Callable[[int, int | None], None]
# progress(label, done, total) of a whole prepare; label = asset key or code:<name>
Progress = Callable[[str, int, int | None], None]


class ProvisionError(Exception):
    """A readable reason why an asset or clone could not be prepared."""


class ProvisionCancelled(Exception):
    """The cancel event was set while a source was working."""


class Source(Protocol):
    """Obtains one item under `root`; raises `ProvisionError` on failure and
    `ProvisionCancelled` when `cancel` is set (checked between chunks)."""

    def prepare(
        self,
        asset: Any,
        root: Path,
        progress: ItemProgress,
        cancel: threading.Event,
    ) -> None: ...


_SOURCES: dict[type, type] = {}


def register_source(asset_cls: type):
    """Class decorator registering a `Source` class for `asset_cls` (an
    `Asset` kind or `CodeRepo`); raises ValueError if one is registered."""

    def deco(source_cls: type) -> type:
        if asset_cls in _SOURCES:
            raise ValueError(
                f"a source for {asset_cls.__name__} is already registered "
                f"({_SOURCES[asset_cls].__name__})"
            )
        _SOURCES[asset_cls] = source_cls
        return source_cls

    return deco


def source_for(item: Any) -> Source | None:
    """A new source for `item`'s class, or for its nearest registered base."""
    for cls in type(item).__mro__:
        if cls in _SOURCES:
            return _SOURCES[cls]()
    return None


def _present(item: Any, root: Path) -> AssetState:
    try:
        return item.present(root)
    except (UnsafePathError, ValueError) as exc:
        return AssetState("invalid", str(exc))
    except (OSError, RuntimeError) as exc:
        return AssetState("partial", f"cannot inspect ({exc})")


def prepare(
    req: Requirements,
    root: Path | None,
    progress: Progress,
    cancel: threading.Event,
) -> ModelStatus:
    """Obtain every missing code repo and asset of `req` under `root`
    (default `weights_root()`) and return the resulting status.

    Idempotent: a ready requirement returns at once and a ready item is
    skipped. Failures never raise: the state is "failed" and `message` lists
    the reasons; a cancel returns the status reached so far with a message
    starting with "cancelled".
    """
    root = weights_root() if root is None else Path(root)
    before = check(req, root)
    if before.state == "ready":
        return before
    items: list[tuple[str, Any]] = [(f"code:{c.name}", c) for c in req.code]
    items += [(a.key, a) for a in req.assets]
    failures: list[str] = []
    cancelled = False
    for label, item in items:
        if cancel.is_set():
            cancelled = True
            break
        state = _present(item, root)
        if state.state == "ready":
            continue
        if state.state == "invalid":
            failures.append(f"{label}: {state.detail}")
            continue
        source = source_for(item)
        if source is None:
            kind = getattr(item, "kind_name", "") or type(item).__name__
            failures.append(
                f"{label}: no source registered for asset kind {kind!r} "
                "(add one with register_source)"
            )
            continue
        try:
            progress(label, 0, getattr(item, "size_bytes", None))
            source.prepare(
                item,
                root,
                lambda done, total, n=label: progress(n, done, total),
                cancel,
            )
        except ProvisionCancelled:
            cancelled = True
            break
        except ProvisionError as exc:
            failures.append(f"{label}: {exc}")
            continue
        except (OSError, ValueError) as exc:  # disk full, permissions, unsafe path
            failures.append(f"{label}: {type(exc).__name__}: {exc}")
            continue
        after = _present(item, root)
        if after.state != "ready":
            failures.append(
                f"{label}: still not ready after download ({after.detail or after.state})"
            )
    status = check(req, root)
    if cancelled:
        reason = "; ".join(["cancelled", *failures])
        return replace(
            status, message=f"{reason}; {status.message}" if status.message else reason
        )
    if failures:
        return replace(status, state="failed", message="; ".join(failures))
    return status
