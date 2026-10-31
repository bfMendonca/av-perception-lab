"""Stand-in for det_depth_distance.web.worker in JobQueue tests (no models).

Usage: python fake_worker.py CONFIG.yaml. The config's `fake` mapping picks
the behaviour: mode ok | sleep | crash, `mark` (file that records start/end),
`pid_file`, `name`.
"""

import json
import os
import sys
import time
from pathlib import Path

import yaml


def emit(**event):
    print(json.dumps(event), flush=True)


def main() -> int:
    cfg = yaml.safe_load(Path(sys.argv[1]).read_text())
    # Validated configs cannot carry a `fake` key: output.tag picks the mode then.
    tag = (cfg.get("output") or {}).get("tag")
    fake = cfg.get("fake") or ({"mode": tag} if tag in ("sleep", "crash") else {})
    name = fake.get("name", "job")
    mode = fake.get("mode", "ok")
    if fake.get("pid_file"):
        Path(fake["pid_file"]).write_text(str(os.getpid()))
    if fake.get("mark"):
        with open(fake["mark"], "a") as fh:
            fh.write(f"start {name}\n")
    if mode == "crash_early":  # e.g. CUDA OOM while loading: no folder yet
        print("RuntimeError: CUDA out of memory (load)", file=sys.stderr, flush=True)
        return 1
    if mode == "sleep_early":
        time.sleep(60)
    out = cfg["output"]
    run_dir = Path(out["out_root"]) / f"20260101_000000_{name}_{out['tag']}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "web_run.json").write_text(json.dumps({"status": "running"}))
    emit(
        event="started",
        run_dir=str(run_dir.with_name("foreign") if mode == "wrongstart" else run_dir),
    )
    if mode == "partial_stderr":  # no newline: must not touch the event stream
        sys.stderr.write("downloading 50%\r")
        sys.stderr.flush()
        time.sleep(0.2)
    if mode == "crash":
        print("RuntimeError: CUDA out of memory", file=sys.stderr, flush=True)
        return 1
    emit(event="frame", n=1)
    if mode == "sleep":
        time.sleep(60)
    time.sleep(float(fake.get("seconds", 0.2)))
    emit(event="frame", n=2)
    (run_dir / "web_run.json").write_text(json.dumps({"status": "done"}))
    if fake.get("mark"):
        with open(fake["mark"], "a") as fh:
            fh.write(f"end {name}\n")
    emit(event="done", run_dir=str(run_dir), frames=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
