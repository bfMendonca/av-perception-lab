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
`message`. An item whose kind is not downloadable (`Asset.downloadable`
False, e.g. `UserProvided`) is not a failure: its source's reason ("provide
this file: ...") goes into `message` and the state stays "missing". With
`allow_unknown=False` (the web server) an item whose source the adapter does
not vouch for (`known` False) is never fetched: it is reported as a failure
("unknown source"), checked when the job runs, not only when it was queued.
Cancellation is a `threading.Event` checked by the sources between
chunks: `prepare` then stops and returns the current status with a message
starting with "cancelled" (a partial download is never ready, see
`core.requirements`). The packages layer is reported, not installed, by
`prepare`: `install_packages` handles it (see `packages`): a fresh dry-run,
then, only when it is clean, the install of the per-model extra; a failed
dry-run is the state "conflict" (`with_package_plan`).
"""

import threading
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from det_depth_distance.core.requirements import (
    AssetState,
    ModelStatus,
    Requirements,
    UnsafePathError,
    check,
    weights_root,
)

if TYPE_CHECKING:
    from det_depth_distance.core.provision.packages import PackagePlan

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


UNKNOWN_SOURCE = "unknown source: prepare it from the CLI (--prepare)"


def prepare(
    req: Requirements,
    root: Path | None,
    progress: Progress,
    cancel: threading.Event,
    *,
    allow_unknown: bool = False,
) -> ModelStatus:
    """Obtain every missing code repo and asset of `req` under `root`
    (default `weights_root()`) and return the resulting status.

    Idempotent: a ready requirement returns at once and a ready item is
    skipped. Failures never raise: the state is "failed" and `message` lists
    the reasons; a cancel returns the status reached so far with a message
    starting with "cancelled". A non-downloadable item adds its source's
    reason to `message` without failing. `allow_unknown` (the CLI passes
    True) lets items with `known` False be fetched; otherwise they fail with
    `UNKNOWN_SOURCE` and nothing is downloaded for them.
    """
    root = weights_root() if root is None else Path(root)
    before = check(req, root)
    if before.state == "ready":
        return before
    items: list[tuple[str, Any]] = [(f"code:{c.name}", c) for c in req.code]
    items += [(a.key, a) for a in req.assets]
    failures: list[str] = []
    manual: list[str] = []
    cancelled = False
    for label, item in items:
        if cancel.is_set():
            cancelled = True
            break
        state = _present(item, root)
        if state.state == "ready":
            continue
        if state.state == "invalid":
            # Asset details already start with the key ("weights: exists...").
            detail = state.detail or state.state
            labelled = detail.startswith(f"{label}: ")
            failures.append(detail if labelled else f"{label}: {detail}")
            continue
        downloadable = getattr(type(item), "downloadable", True)
        if downloadable and not allow_unknown and not getattr(item, "known", False):
            failures.append(f"{label}: {UNKNOWN_SOURCE}")
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
            (failures if downloadable else manual).append(f"{label}: {exc}")
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
        return replace(status, state="failed", message="; ".join(failures + manual))
    if manual and status.state != "ready":
        return replace(status, message="; ".join(manual))
    return status


def with_package_plan(status: ModelStatus, plan: "PackagePlan") -> ModelStatus:
    """`status` with the packages plan applied: a "conflict" plan turns a
    "missing" model into "conflict"; an "invalid" or "error" one (the
    dry-run could not run: uv unavailable, timeout) into "failed", with the
    plan's message first."""
    if plan.action not in ("conflict", "invalid", "error") or status.state not in (
        "missing",
        "conflict",
    ):
        return status
    state = "conflict" if plan.action == "conflict" else "failed"
    parts = [plan.message, status.message]
    return replace(status, state=state, message="; ".join(p for p in parts if p))


@dataclass(frozen=True, slots=True)
class PackagesOutcome:
    """Result of `install_packages`: the status afterwards, the plan the
    fresh dry-run gave and whether `uv sync` ran to completion."""

    status: ModelStatus
    plan: "PackagePlan"
    installed: bool


def install_packages(
    req: Requirements,
    on_line: Callable[[str], None],
    cancel: threading.Event,
    *,
    root: Path | None = None,
    pyproject: Path | None = None,
    uv: str | None = None,
    imports: Any = None,
    expected: "PackagePlan | None" = None,
) -> PackagesOutcome:
    """Install the per-model extra of `req` when its packages are missing:
    a fresh dry-run first (with the same config-wide `imports` the shown plan
    used); only a clean one ("install") is followed by the install (output
    lines to `on_line`). With `expected` (the plan the user confirmed), the
    fresh plan must be "install" with the same extras and command, else
    nothing runs ("failed": the environment changed). The base stack is
    never installed (plan "base"); a failed dry-run returns the state
    "conflict" and changes nothing. Never raises for uv failures: "failed"
    with the reason; a cancel returns the status with a message starting
    with "cancelled".
    """
    # packages imports this module (its errors): imported here, not on top.
    from det_depth_distance.core.provision import packages

    root = weights_root() if root is None else Path(root)
    plan = packages.plan_packages(
        req, pyproject=pyproject, uv=uv, imports=imports, fresh=True, cancel=cancel
    )
    if cancel.is_set():
        status = check(req, root)
        return PackagesOutcome(
            replace(status, message="cancelled before uv sync started"), plan, False
        )
    if expected is not None and (
        expected.action != "install"
        or plan.action != "install"
        or plan.extras != expected.extras
        or plan.command != expected.command
    ):
        if plan.action == "error":
            # A transient tooling failure (uv missing, timeout): say why.
            why = (
                f"{plan.message or 'the dry-run could not run'}; nothing was installed"
            )
        else:
            why = (
                "the environment changed since the install was confirmed "
                f"(now: {plan.action}"
                + (f", `{plan.command}`" if plan.command else "")
                + (f": {plan.message}" if plan.message else "")
                + "): review the model again"
            )
        status = replace(check(req, root), state="failed", message=why)
        return PackagesOutcome(status, plan, False)
    if plan.action != "install" or plan.extra is None:
        return PackagesOutcome(with_package_plan(check(req, root), plan), plan, False)
    on_line(f"$ {plan.command}")
    try:
        packages.install(
            plan.extra,
            set(plan.extras) - {plan.extra},
            packages.default_uv() if uv is None else uv,
            pyproject=pyproject,
            on_line=on_line,
            cancel=cancel,
        )
    except ProvisionCancelled:
        status = check(req, root)
        reason = "cancelled: uv sync was stopped"
        message = f"{reason}; {status.message}" if status.message else reason
        return PackagesOutcome(replace(status, message=message), plan, False)
    except ProvisionError as exc:
        status = replace(check(req, root), state="failed", message=str(exc))
        return PackagesOutcome(status, plan, False)
    return PackagesOutcome(check(req, root), plan, True)
