"""The refactored pipeline must reproduce task_0001 on IMG_4021 (frames 0-59).

Reference: the first 60 frames of task_0001's full run (frozen in Passo 0 under
_testbed/regression/, or $DDD_REGRESSION_REF), proven bit-identical across
three task_0001 runs. Tolerances only absorb the CSV's decimal representation;
a failure means the refactor changed results - investigate before relaxing.
The run happens in a fresh interpreter so that no other test of the session
(e.g. the YOLO adapter importing ultralytics) can influence it.
"""

import csv
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import pytest

from det_depth_distance.distance import STATS_ROW_FIELDS

REPO = Path(__file__).resolve().parents[2]
REF_CSV = Path(
    os.environ.get(
        "DDD_REGRESSION_REF",
        REPO / "_testbed" / "regression" / "task_0001_IMG_4021_60f.csv",
    )
)
REF_METRICS = REF_CSV.with_name("task_0001_IMG_4021_metrics.json")
REF_FOCAL = 1390.407941545759

EXACT = ("frame", "class_name", "class_id", "t", "confidence")
BOX = ("x0", "y0", "x1", "y1")
INTEGERS = ("map_c0", "map_r0", "map_c1", "map_r1", "n_box_px", "n_valid", "n_inliers")
FLOATS = tuple(f for f in STATS_ROW_FIELDS if f not in INTEGERS)


def _run(out_root: Path) -> None:
    argv = [
        "--config",
        "configs/rfdetr_da3.yaml",
        "--set",
        "source.max_frames=60",
        "--set",
        f"output.out_root={out_root}",
        "--set",
        "render.layout=overlay",
    ]
    code = (
        "import sys\n"
        "from det_depth_distance.cli import main\n"
        f"code = main({argv!r})\n"
        "assert 'ultralytics' not in sys.modules, 'ultralytics imported by an RF-DETR run'\n"
        "sys.exit(code)\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO,
        check=False,
        capture_output=True,
        text=True,
    )
    assert out.returncode == 0, out.stderr[-3000:]


def _read(path: Path) -> list[dict[str, str]]:
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def _close(a: str, b: str, rel: float) -> bool:
    if a == b:
        return True
    if not a or not b:
        return False
    x, y = float(a), float(b)
    if math.isnan(x) or math.isnan(y):
        return math.isnan(x) and math.isnan(y)
    return math.isclose(x, y, rel_tol=rel, abs_tol=0.0)


def test_regression_against_task_0001(tmp_path, video):
    if not REF_CSV.is_file() or not REF_METRICS.is_file():
        pytest.skip(f"missing regression reference {REF_CSV} (task_0002 Passo 0)")
    _run(tmp_path)
    (run_dir,) = tmp_path.iterdir()
    ref, new = _read(REF_CSV), _read(run_dir / "detections.csv")

    assert len(new) == len(ref) == 102
    for i, (r, n) in enumerate(zip(ref, new, strict=True)):
        for col in EXACT + INTEGERS:
            assert n[col] == r[col], (i, col, r[col], n[col])
        for col in BOX:
            assert abs(float(n[col]) - float(r[col])) <= 1e-3, (i, col, r[col], n[col])
        for col in FLOATS:
            assert _close(n[col], r[col], 1e-6), (i, col, r[col], n[col])
        assert n["source"] == "rfdetr-large" and n["raw_label"] == n["class_name"]

    metrics = json.loads((run_dir / "metrics.json").read_text())
    reference = json.loads(REF_METRICS.read_text())
    assert math.isclose(metrics["focal_frame_px"], REF_FOCAL, rel_tol=1e-9)
    assert metrics["focal_map_intrinsics"] == reference["focal_map_intrinsics"]
    assert metrics["boxes_total"] == 102
