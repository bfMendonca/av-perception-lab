"""Worker: `python -m det_depth_distance.web.worker CONFIG.yaml`.

Runs `cli.run` in its own process (the only place the models, hence torch,
are loaded) and reports on stdout, one JSON object per line:
`started` (run_dir), `frame` (n), `done` (run_dir, frames) or `error`
(message). Exit code 0 on success, 2 on `ConfigError`, 1 on anything else.
The normal output of `run` goes to stderr so stdout stays machine-readable.

`run()` creates the run folder itself, so it is discovered on the first
frame as the new folder of `output.out_root`. The worker writes into it the
effective `config.yaml`, `web_run.json` (`status`, `video_id` from the
`DDD_WEB_VIDEO_ID` environment variable, `started_at`) and `preview.jpg`
(rewritten at most every `PREVIEW_PERIOD_S`).
"""

import argparse
import contextlib
import json
import os
import sys
import time
import traceback
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

from det_depth_distance import cli
from det_depth_distance.config import load_config
from det_depth_distance.core.options import ConfigError

PREVIEW_PERIOD_S = 0.5
VIDEO_ID_ENV = "DDD_WEB_VIDEO_ID"


def _folders(out_root: Path) -> set[Path]:
    return {p for p in out_root.iterdir() if p.is_dir()} if out_root.is_dir() else set()


def _write_status(run_dir: Path, status: str, started_at: str) -> None:
    (run_dir / "web_run.json").write_text(
        json.dumps(
            {
                "status": status,
                "video_id": os.environ.get(VIDEO_ID_ENV) or None,
                "started_at": started_at,
            }
        )
    )


def _write_preview(run_dir: Path, rgb: np.ndarray) -> None:
    """Atomic JPEG so a reader never sees a half-written file."""
    tmp = run_dir / "preview.tmp.jpg"
    if cv2.imwrite(str(tmp), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)):
        os.replace(tmp, run_dir / "preview.jpg")


def main(argv: Sequence[str] | None = None) -> int:
    """Worker entry point; returns the exit code."""
    parser = argparse.ArgumentParser(prog="det_depth_distance.web.worker")
    parser.add_argument("config", type=Path, help="run configuration YAML")
    args = parser.parse_args(argv)
    out = sys.stdout

    def emit(**event: Any) -> None:
        out.write(json.dumps(event) + "\n")
        out.flush()

    try:
        cfg = load_config(args.config)
    except ConfigError as exc:
        emit(event="error", message=str(exc))
        return 2
    out_root = Path(cfg.output.out_root)
    before = _folders(out_root)
    started_at = datetime.now().astimezone().isoformat(timespec="seconds")
    state: dict[str, Any] = {"run_dir": None, "frames": 0, "preview_at": -1e9}

    def discover() -> Path | None:
        """The run folder `run()` created (newest new folder of out_root)."""
        if state["run_dir"] is None:
            new = _folders(out_root) - before
            if new:
                run_dir = max(new, key=lambda p: p.stat().st_mtime)
                state["run_dir"] = run_dir
                (run_dir / "config.yaml").write_text(
                    yaml.safe_dump(cfg.to_dict(), sort_keys=False, allow_unicode=True)
                )
                _write_status(run_dir, "running", started_at)
                emit(event="started", run_dir=str(run_dir))
        return state["run_dir"]

    def on_frame(index: int, image: np.ndarray) -> None:
        run_dir = discover()
        state["frames"] += 1
        now = time.monotonic()
        if run_dir is not None and now - state["preview_at"] >= PREVIEW_PERIOD_S:
            state["preview_at"] = now
            _write_preview(run_dir, image)
        emit(event="frame", n=state["frames"])

    def fail(message: str, code: int) -> int:
        run_dir = discover()
        if run_dir is not None:
            _write_status(run_dir, "failed", started_at)
        emit(event="error", message=message)
        return code

    try:
        with contextlib.redirect_stdout(sys.stderr):
            run_dir = cli.run(cfg, on_frame)
    except ConfigError as exc:
        return fail(str(exc), 2)
    except Exception as exc:  # noqa: BLE001 - any failure is reported
        traceback.print_exc()
        return fail(f"{type(exc).__name__}: {exc}", 1)
    _write_status(run_dir, "done", started_at)
    emit(event="done", run_dir=str(run_dir), frames=state["frames"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
