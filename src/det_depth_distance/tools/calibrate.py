"""det-depth-distance-calibrate: calibrate | board | inspect.

calibrate VIDEO|--images GLOB writes an OpenCV calibration file; board
renders a printable target; inspect prints a video's camera metadata, the
focal range implied by its 35 mm equivalent and, with --calibration, whether
the calibration matches the recording camera.

Video frames are decoded by VideoSource, exactly as pipeline runs decode
them (upright, same size), so the calibration applies to the same pixels.
Configuration errors print "error: ..." and exit with code 2.
"""

import argparse
import glob
import sys
from collections.abc import Iterator, Sequence
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from det_depth_distance.calibration import (
    CalibrationOptions,
    CalibrationReport,
    TargetSpec,
    View,
    calibrate_views,
    detect_views,
    focal_px_from_35mm,
    make_target,
    select_views,
)
from det_depth_distance.camera import (
    Calibration,
    check_camera_metadata,
    load_calibration,
    save_calibration,
)
from det_depth_distance.config import SourceConfig
from det_depth_distance.core.options import ConfigError
from det_depth_distance.io.source import VideoSource
from det_depth_distance.runtime import git_commit, package_version

# Acceptance criteria of docs/refs/protocolo_calibracao_e_referencias.md. Below
# MIN_COVERAGE the distortion at the frame edges is poorly constrained.
MAX_RMS_PX = 0.5
MIN_COVERAGE = 0.6
MAX_FX_STD_REL = 0.005  # std of fx as a fraction of fx
COVERAGE_HINT = "otherwise show the target near the frame edges and corners"


def _add_target_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--target", choices=("chessboard", "charuco"), default="chessboard")
    p.add_argument(
        "--cols",
        type=int,
        default=9,
        help="chessboard: inner corners per row; charuco: squares per row (9)",
    )
    p.add_argument(
        "--rows",
        type=int,
        default=6,
        help="chessboard: inner corners per column; charuco: squares per column (6)",
    )
    p.add_argument(
        "--square-mm",
        type=float,
        default=25.0,
        help="square side measured on the print, in mm (25)",
    )
    p.add_argument("--marker-mm", type=float, help="charuco marker side, in mm")
    p.add_argument(
        "--dictionary", default="DICT_5X5_100", help="charuco ArUco dictionary"
    )


def build_parser() -> argparse.ArgumentParser:
    """Subcommands calibrate | board | inspect."""
    p = argparse.ArgumentParser(
        prog="det-depth-distance-calibrate",
        description="Camera calibration with a chessboard or ChArUco target "
        "(see docs/refs/protocolo_calibracao_e_referencias.md).",
    )
    sub = p.add_subparsers(dest="command", required=True)

    c = sub.add_parser(
        "calibrate", help="calibrate from a video or images of the target"
    )
    c.add_argument("video", nargs="?", type=Path, help="calibration video")
    c.add_argument("--images", help="glob of target images instead of a video")
    c.add_argument(
        "--out", type=Path, required=True, help="calibration file (.yaml/.json)"
    )
    _add_target_args(c)
    c.add_argument(
        "--every-s", type=float, default=0.5, help="video: one frame every N s (0.5)"
    )
    c.add_argument("--start", type=float, default=0.0, help="video: start (s)")
    c.add_argument(
        "--duration",
        type=float,
        default=0.0,
        help="video: duration (s, 0 = to the end)",
    )
    c.add_argument("--min-views", type=int, default=12)
    c.add_argument("--max-views", type=int, default=60)
    c.add_argument("--no-fix-k3", action="store_true", help="also fit k3")
    c.add_argument("--fix-aspect", action="store_true", help="force fx = fy")
    c.add_argument(
        "--note",
        default="",
        help="recording notes kept in the file (e.g. stabilisation mode, lens)",
    )
    c.add_argument(
        "--debug-dir", type=Path, help="save every detected view with its corners"
    )

    b = sub.add_parser("board", help="render a printable target (PNG)")
    b.add_argument("--out", type=Path, required=True, help="PNG file")
    _add_target_args(b)
    b.add_argument("--dpi", type=float, default=300.0)
    b.add_argument("--margin-mm", type=float, default=10.0)

    i = sub.add_parser("inspect", help="camera metadata of a video")
    i.add_argument("video", type=Path)
    i.add_argument(
        "--calibration", type=Path, help="check this calibration against the video"
    )
    return p


def _spec(args: argparse.Namespace) -> TargetSpec:
    return TargetSpec(
        type=args.target,
        cols=args.cols,
        rows=args.rows,
        square_mm=args.square_mm,
        marker_mm=args.marker_mm,
        dictionary=args.dictionary,
    )


def _image_frames(pattern: str) -> tuple[list[tuple[int, np.ndarray]], tuple[int, int]]:
    """(index, grayscale image) of every file matching `pattern`, and their size."""
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise ConfigError(f"--images: no file matches {pattern!r}")
    frames, size = [], None
    for index, path in enumerate(paths):
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ConfigError(f"--images: cannot read {path}")
        if size is None:
            size = image.shape[1], image.shape[0]
        elif (image.shape[1], image.shape[0]) != size:
            raise ConfigError(
                f"--images: {path} is not {size[0]}x{size[1]} like the first image"
            )
        frames.append((index, image))
    return frames, size


def _video_frames(
    args: argparse.Namespace,
) -> tuple[VideoSource, Iterator[tuple[int, np.ndarray]]]:
    base = VideoSource(
        SourceConfig(video=args.video, start=args.start, duration=args.duration)
    )
    stride = max(1, round(base.info.fps * args.every_s))
    source = VideoSource(replace(base.cfg, stride=stride))
    return source, ((frame.index, frame.rgb) for frame in source)


def _debug_writer(directory: Path):
    directory.mkdir(parents=True, exist_ok=True)

    def write(view: View, image: np.ndarray) -> None:
        bgr = cv2.cvtColor(
            image, cv2.COLOR_RGB2BGR if image.ndim == 3 else cv2.COLOR_GRAY2BGR
        )
        for x, y in view.image_points:
            cv2.circle(bgr, (round(x), round(y)), 4, (0, 0, 255), 1, cv2.LINE_AA)
        cv2.imwrite(str(directory / f"view_{view.index:06d}.png"), bgr)

    return write


def _print_report(report: CalibrationReport, n_examined: int, n_detected: int) -> None:
    cal = report.calibration
    k = cal.camera_matrix()
    std = cal.std
    print(
        f"[views] {n_examined} examined, {n_detected} with the target, "
        f"{len(report.used_views)} used, {len(report.rejected_views)} rejected"
    )
    print(
        f"[K] fx {k[0, 0]:.2f} ± {std.get('fx', float('nan')):.2f}, "
        f"fy {k[1, 1]:.2f} ± {std.get('fy', float('nan')):.2f}, "
        f"cx {k[0, 2]:.2f} ± {std.get('cx', float('nan')):.2f}, "
        f"cy {k[1, 2]:.2f} ± {std.get('cy', float('nan')):.2f} px (OpenCV convention)"
    )
    print(f"[dist] {', '.join(f'{d:.5f}' for d in cal.dist)}")
    print(f"[fit] RMS {cal.rms_px:.3f} px, coverage {report.coverage:.2f}")
    for passed, text in _acceptance(report):
        if passed:
            print(f"[accept] ok: {text}")
        else:
            print(f"[warn] not accepted: {text}", file=sys.stderr)


def _acceptance(report: CalibrationReport) -> list[tuple[bool, str]]:
    """(passed, description) of each acceptance criterion of the protocol."""
    cal = report.calibration
    fx = cal.intrinsics.fx
    fx_std = cal.std.get("fx", float("nan"))
    return [
        (cal.rms_px < MAX_RMS_PX, f"RMS {cal.rms_px:.3f} px < {MAX_RMS_PX} px"),
        (
            report.coverage >= MIN_COVERAGE,
            f"coverage {report.coverage:.2f} >= {MIN_COVERAGE} ({COVERAGE_HINT})",
        ),
        (
            fx_std / fx < MAX_FX_STD_REL,
            f"std of fx {100 * fx_std / fx:.2f} % < {100 * MAX_FX_STD_REL:g} % of fx",
        ),
    ]


def _print_35mm(
    camera: dict, width: int, height: int, focal: float | None = None
) -> None:
    f35 = camera.get("focal_length_35mm")
    if f35 is None:
        return
    implied = focal_px_from_35mm(float(f35), width, height)
    line = f"[35mm] {f35} mm equivalent -> {implied:.1f} px (diagonal convention; diagnostic only)"
    if focal is not None:
        line += f"; calibrated focal / implied = {focal / implied:.3f}"
    print(line)


def _calibrate(args: argparse.Namespace) -> int:
    if (args.video is None) == (args.images is None):
        raise ConfigError("calibrate: give a video or --images, not both")
    target = make_target(_spec(args))
    options = CalibrationOptions(
        fix_k3=not args.no_fix_k3,
        fix_aspect=args.fix_aspect,
        min_views=args.min_views,
        max_views=args.max_views,
    )
    on_view = _debug_writer(args.debug_dir) if args.debug_dir else None
    if args.video is not None:
        source, frames = _video_frames(args)
        size = source.frame_size
        camera = dict(source.camera_metadata())
        recording = {
            "video": source.info.path.name,
            "fps": source.info.fps,
            "every_s": args.every_s,
        }
    else:
        frames, size = _image_frames(args.images)
        camera, recording = {}, {"images": args.images}
    if args.note:
        recording["notes"] = args.note
    views, n_examined = detect_views(frames, target, on_view=on_view)
    selected = select_views(views, options.max_views)
    report = calibrate_views(
        selected,
        size,
        options,
        meta={
            "target": target.spec.describe(),
            "recording": recording,
            "n_candidates": n_examined,
            "n_detected": len(views),
            "created": datetime.now().isoformat(timespec="seconds"),
            "tool": {
                "name": "det-depth-distance-calibrate",
                "version": package_version("det-depth-distance"),
                "commit": git_commit(),
            },
        },
        camera=camera,
    )
    _print_report(report, n_examined, len(views))
    k = report.calibration.intrinsics
    _print_35mm(camera, *size, focal=(k.fx + k.fy) / 2)
    save_calibration(report.calibration, args.out)
    print(f"[saved] {args.out}")
    return 0


def _board(args: argparse.Namespace) -> int:
    target = make_target(_spec(args))
    image = target.image(args.dpi / 25.4, args.margin_mm)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    try:
        # The resolution is stored in the file (PNG pHYs) so that printing at
        # 100 % gives the real square size; cv2.imwrite cannot store it.
        Image.fromarray(image).save(args.out, dpi=(args.dpi, args.dpi))
    except (OSError, ValueError) as exc:
        raise ConfigError(f"board: cannot write {args.out} ({exc})") from None
    h, w = image.shape
    print(
        f"[board] {w}x{h} px = {w / args.dpi * 25.4:.1f}x{h / args.dpi * 25.4:.1f} mm at "
        f"{args.dpi:g} dpi -> {args.out}; print at 100 % and measure the squares"
    )
    return 0


def _inspect(args: argparse.Namespace) -> int:
    source = VideoSource(SourceConfig(video=args.video))
    w, h = source.frame_size
    camera = dict(source.camera_metadata())
    print(
        f"[video] {source.info.path.name}: {w}x{h} @ {source.info.fps:.3f} fps, "
        f"{source.info.n_frames} frames"
    )
    if not camera:
        print("[camera] no QuickTime camera metadata")
    for key, value in camera.items():
        print(f"[camera] {key}: {value}")
    _print_35mm(camera, w, h)
    if args.calibration is None:
        return 0
    calibration: Calibration = load_calibration(args.calibration).scaled_to(w, h)
    check = check_camera_metadata(calibration.camera, camera, "warn")
    k = calibration.intrinsics
    print(
        f"[calibration] {args.calibration}: fx {k.fx:.2f}, fy {k.fy:.2f} px at {w}x{h}; "
        f"camera check: {check['status']}"
    )
    return 1 if check["status"] == "mismatch" else 0


def main(argv: Sequence[str] | None = None) -> int:
    """Console entry point (det-depth-distance-calibrate)."""
    args = build_parser().parse_args(argv)
    command = {"calibrate": _calibrate, "board": _board, "inspect": _inspect}[
        args.command
    ]
    try:
        return command(args)
    # User/configuration errors only; bugs keep their traceback.
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
