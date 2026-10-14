"""det-depth-distance-evaluate --refs REFS.yaml RUN [RUN ...]: error report
of one or more runs against field references (eval.json, eval.md, plots).

With one run the report goes to <run>/eval_<references stem>/ unless --out
is given; several runs need --out (side-by-side comparison). Runs that
undistorted their frames get their reference pixels undistorted with the K
and distortion recorded in their metrics.json (camera.calibration), not with
the calibration file, so recalibrating later does not change old evaluations.
Configuration errors print "error: ..." and exit with code 2.
"""

import argparse
import sys
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

from det_depth_distance.camera import Undistorter
from det_depth_distance.core.options import ConfigError
from det_depth_distance.evaluation.evaluate import EvalOptions, evaluate
from det_depth_distance.evaluation.matching import PointsTransform
from det_depth_distance.evaluation.references import load_references
from det_depth_distance.evaluation.report import write_report
from det_depth_distance.evaluation.run_data import ESTIMATORS, RunData, load_run
from det_depth_distance.evaluation.stats import DEFAULT_BINS

# Rebuilt vs recorded undistorted K (px): JSON floats round-trip exactly, so any
# real difference means another OpenCV behaviour or an edited metrics.json.
K_TOLERANCE_PX = 1e-6


def _floats(text: str) -> tuple[float, ...]:
    try:
        return tuple(float(v) for v in text.split(","))
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"expected comma-separated numbers, got {text!r}"
        ) from None


def build_parser() -> argparse.ArgumentParser:
    """--refs, RUN..., --out, --estimators, --bins, --level, --iou-min, --plot, --force."""
    p = argparse.ArgumentParser(
        prog="det-depth-distance-evaluate",
        description="Compare the distances of one or more runs with field reference "
        "distances (see docs/refs/protocolo_calibracao_e_referencias.md).",
    )
    p.add_argument("runs", nargs="+", type=Path, metavar="RUN", help="run folder(s)")
    p.add_argument(
        "--refs", type=Path, required=True, help="reference YAML of the video"
    )
    p.add_argument(
        "--out", type=Path, help="report folder (required with several runs)"
    )
    p.add_argument(
        "--estimators",
        type=lambda s: tuple(s.split(",")),
        default=ESTIMATORS,
        help=f"comma-separated subset of {','.join(ESTIMATORS)} (all present by default)",
    )
    p.add_argument(
        "--bins",
        type=_floats,
        default=DEFAULT_BINS,
        help="reference distance bin edges in m (default 0,5,10,20,30,50; last bin open)",
    )
    p.add_argument("--level", choices=("reference", "frame"), default="reference")
    p.add_argument(
        "--iou-min", type=float, default=0.3, help="box selector minimum IoU (0.3)"
    )
    p.add_argument(
        "--plot", action="store_true", help="also write PNG figures (extra eval)"
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="evaluate runs of another video than the references'",
    )
    return p


def _points_transform(run: RunData) -> PointsTransform | None:
    """Undistortion of reference pixels for a run made with camera.calibration.undistort.

    Rebuilt from the calibration recorded in the run's metrics.json (never from
    the file, which may have been recalibrated since); the rebuilt undistorted
    K must equal the one the run recorded.

    Raises:
        ConfigError: the rebuilt undistorted K differs from the recorded one.
    """
    recorded = run.undistort_calibration
    if recorded is None:
        return None
    calibration, alpha = recorded
    undistorter = Undistorter(calibration, alpha=alpha)
    expected = run.metrics["camera"]["undistort"].get("intrinsics")
    rebuilt = asdict(undistorter.output.intrinsics)
    if expected is not None and any(
        key not in expected or abs(rebuilt[key] - expected[key]) > K_TOLERANCE_PX
        for key in rebuilt
    ):
        raise ConfigError(
            f"run {run.name}: the undistorted K rebuilt from metrics.json "
            f"({rebuilt}) differs from the recorded one ({expected})"
        )
    return undistorter.undistort_points


def main(argv: Sequence[str] | None = None) -> int:
    """Console entry point (det-depth-distance-evaluate)."""
    args = build_parser().parse_args(argv)
    try:
        if args.out is None and len(args.runs) > 1:
            raise ConfigError("--out is required with several runs")
        options = EvalOptions(
            estimators=args.estimators,
            level=args.level,
            bins=args.bins,
            iou_min=args.iou_min,
            force=args.force,
        )
        refs = load_references(args.refs)
        runs = [load_run(path) for path in args.runs]
        transforms = {run.name: t for run in runs if (t := _points_transform(run))}
        result = evaluate(refs, runs, options, points_transforms=transforms)
        out = args.out or args.runs[0] / f"eval_{args.refs.stem}"
        paths = write_report(result, out, plot=args.plot)
    # User/configuration errors only; bugs keep their traceback.
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    for warning in result.warnings:
        print(f"[warn] {warning}", file=sys.stderr)
    for path in paths:
        print(f"[saved] {path}")
    return 0
