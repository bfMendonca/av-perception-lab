"""Hand-built run folders for the compare tests (CSV with chosen columns)."""

import csv
import json
from pathlib import Path


def write_csv(path: Path, header: list[str], rows: list[list]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        writer.writerows(rows)


def make_run(out: Path, name: str, header, rows, metrics=None, config=None) -> Path:
    run = out / name
    write_csv(run / "detections.csv", header, rows)
    if metrics is not None:
        (run / "metrics.json").write_text(json.dumps(metrics))
    if config is not None:
        (run / "config.yaml").write_text(json.dumps(config))  # JSON is YAML
    return run


# Columns of a pre-web run (task_0001/0002 era): no track_id, no smoothing.
OLD_HEADER = ["frame", "t", "class_name", "distance_m", "source", "raw_label"]
NEW_HEADER = [*OLD_HEADER, "track_id", "distance_smooth_m"]
