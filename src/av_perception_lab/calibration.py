"""Camera calibration from views of a planar target (chessboard or ChArUco).

Views come from a video decoded exactly like pipeline runs (upright, same
crop) or from images. cv2.aruco ships with opencv-python, but the legacy
calibrateCameraCharuco does not: ChArUco corners are matched to board
points with Board.matchImagePoints and calibrated with calibrateCamera like
the chessboard. k3 is fixed by default: with tens of views it is poorly
constrained and trades off against k1/k2.

Image points are in OpenCV's pixel convention (pixel i centred at i);
`camera.Calibration.from_opencv` converts the result to the project's
continuous convention.
"""

from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import cv2
import numpy as np

from av_perception_lab.camera import Calibration
from av_perception_lab.core.options import ConfigError

TargetType = Literal["chessboard", "charuco"]
# Names of stdIntrinsics entries of calibrateCameraExtended, in order.
_STD_NAMES = ("fx", "fy", "cx", "cy", "k1", "k2", "p1", "p2", "k3")
# Diagonal of the 36 x 24 mm full frame that defines "35 mm equivalent".
FULL_FRAME_DIAGONAL_MM = float(np.hypot(36.0, 24.0))
# ChArUco views with fewer interpolated corners, or with collinear ones, are
# dropped (unstable or undefined pose).
MIN_CHARUCO_CORNERS = 6


@dataclass(frozen=True, slots=True)
class TargetSpec:
    """A printed planar target. Built from CLI arguments, so it raises
    ConfigError directly."""

    type: TargetType = "chessboard"
    cols: int = 9  # chessboard: inner corners per row; charuco: squares per row
    rows: int = 6  # chessboard: inner corners per column; charuco: squares per column
    square_mm: float = 25.0  # measured on the print, not the nominal size
    marker_mm: float | None = None  # charuco only, < square_mm
    dictionary: str = "DICT_5X5_100"  # cv2.aruco.DICT_* name (charuco)

    def __post_init__(self) -> None:
        if self.type not in ("chessboard", "charuco"):
            raise ConfigError(f"target: unknown type {self.type!r}")
        if self.cols < 2 or self.rows < 2:
            raise ConfigError("target: cols and rows must be >= 2")
        if self.square_mm <= 0:
            raise ConfigError("target: square_mm must be > 0")
        if self.type == "charuco":
            if self.marker_mm is None or not 0 < self.marker_mm < self.square_mm:
                raise ConfigError("target: charuco needs 0 < marker_mm < square_mm")
            if not self.dictionary.startswith("DICT_") or not hasattr(
                cv2.aruco, self.dictionary
            ):
                raise ConfigError(
                    f"target: unknown ArUco dictionary {self.dictionary!r}"
                )
        elif self.marker_mm is not None:
            raise ConfigError("target: marker_mm is only used by charuco targets")

    def describe(self) -> dict[str, Any]:
        """Calibration-file `target` entry."""
        out: dict[str, Any] = {
            "type": self.type,
            "cols": self.cols,
            "rows": self.rows,
            "square_mm": self.square_mm,
        }
        if self.type == "charuco":
            out |= {"marker_mm": self.marker_mm, "dictionary": self.dictionary}
        return out


@dataclass(frozen=True, slots=True)
class View:
    """Target corners found in one frame or image."""

    index: int  # source frame index (or image index)
    object_points: np.ndarray  # (K, 3) float32, metres, plane z = 0
    image_points: np.ndarray  # (K, 2) float32, pixels, OpenCV convention


def _gray(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image


class CalibrationTarget(ABC):
    """Detects a target in an image and renders it for printing.

    A new target (e.g. asymmetric circles) is a subclass plus an entry in TARGETS.
    """

    def __init__(self, spec: TargetSpec) -> None:
        self.spec = spec

    @abstractmethod
    def detect(self, image: np.ndarray, index: int) -> View | None:
        """Corners of the target in an RGB or grayscale image; None if not found."""

    @abstractmethod
    def image(self, px_per_mm: float, margin_mm: float = 10.0) -> np.ndarray:
        """Grayscale uint8 rendering of the target with a white margin, for printing."""


class ChessboardTarget(CalibrationTarget):
    """Classic chessboard; `cols` x `rows` inner corners."""

    def __init__(self, spec: TargetSpec) -> None:
        super().__init__(spec)
        grid = np.mgrid[0 : spec.cols, 0 : spec.rows].T.reshape(-1, 2)
        # Row-major, x fastest: the order findChessboardCorners reports.
        self._object = np.zeros((len(grid), 3), np.float32)
        self._object[:, :2] = grid * (spec.square_mm / 1000.0)

    def detect(self, image: np.ndarray, index: int) -> View | None:
        gray = _gray(image)
        size = (self.spec.cols, self.spec.rows)
        # ACCURACY refines corners (RMS 0.04 vs 0.06 px on rendered boards) at
        # ~3x the time; EXHAUSTIVE would cost another 2-3x for no gain there.
        found, corners = cv2.findChessboardCornersSB(
            gray, size, flags=cv2.CALIB_CB_ACCURACY
        )
        if not found:
            # The sector-based detector misses some blurred or small boards
            # that the classic one finds; refine those corners explicitly.
            found, corners = cv2.findChessboardCorners(
                gray,
                size,
                flags=cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE,
            )
            if not found:
                return None
            corners = cv2.cornerSubPix(
                gray,
                corners,
                (5, 5),
                (-1, -1),
                (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 1e-3),
            )
        return View(index, self._object, corners.reshape(-1, 2).astype(np.float32))

    def image(self, px_per_mm: float, margin_mm: float = 10.0) -> np.ndarray:
        s = self.spec
        square = s.square_mm * px_per_mm
        margin = round(margin_mm * px_per_mm)
        nx, ny = s.cols + 1, s.rows + 1  # squares
        width = round(nx * square) + 2 * margin
        height = round(ny * square) + 2 * margin
        out = np.full((height, width), 255, np.uint8)
        for j in range(ny):
            for i in range(nx):
                if (i + j) % 2 == 0:
                    x0, y0 = margin + round(i * square), margin + round(j * square)
                    x1, y1 = (
                        margin + round((i + 1) * square),
                        margin + round((j + 1) * square),
                    )
                    out[y0:y1, x0:x1] = 0
        return out


class CharucoTarget(CalibrationTarget):
    """ChArUco board; `cols` x `rows` squares. Partially visible boards still
    give views, which helps cover the frame corners."""

    def __init__(self, spec: TargetSpec) -> None:
        super().__init__(spec)
        dictionary = cv2.aruco.getPredefinedDictionary(
            getattr(cv2.aruco, spec.dictionary)
        )
        self.board = cv2.aruco.CharucoBoard(
            (spec.cols, spec.rows),
            spec.square_mm / 1000.0,
            spec.marker_mm / 1000.0,
            dictionary,
        )
        self._detector = cv2.aruco.CharucoDetector(self.board)

    def detect(self, image: np.ndarray, index: int) -> View | None:
        corners, ids, _, _ = self._detector.detectBoard(_gray(image))
        if ids is None or len(ids) < MIN_CHARUCO_CORNERS:
            return None
        obj, img = self.board.matchImagePoints(corners, ids)
        obj = obj.reshape(-1, 3).astype(np.float32)
        # A partial board can show a single row or column of corners: no
        # homography exists for it and calibrateCamera aborts the whole fit.
        if np.linalg.matrix_rank(obj[:, :2] - obj[:, :2].mean(0), tol=1e-9) < 2:
            return None
        return View(index, obj, img.reshape(-1, 2).astype(np.float32))

    def image(self, px_per_mm: float, margin_mm: float = 10.0) -> np.ndarray:
        s = self.spec
        margin = round(margin_mm * px_per_mm)
        width = round(s.cols * s.square_mm * px_per_mm) + 2 * margin
        height = round(s.rows * s.square_mm * px_per_mm) + 2 * margin
        return self.board.generateImage((width, height), marginSize=margin)


TARGETS: dict[str, type[CalibrationTarget]] = {
    "chessboard": ChessboardTarget,
    "charuco": CharucoTarget,
}


def make_target(spec: TargetSpec) -> CalibrationTarget:
    """The detector/renderer of `spec.type`."""
    return TARGETS[spec.type](spec)


@dataclass(frozen=True, slots=True)
class CalibrationOptions:
    """Fit settings. Built from CLI arguments, so it raises ConfigError directly."""

    fix_k3: bool = True
    fix_aspect: bool = False  # fx = fy (square pixels)
    min_views: int = 12
    max_views: int = 60
    outlier_factor: float = 3.0  # drop views with RMS > max(f * median, 1 px)
    grid: tuple[int, int] = (8, 6)  # coverage cells (x, y)

    def __post_init__(self) -> None:
        if self.min_views < 3:
            raise ConfigError("min_views must be >= 3")
        if self.max_views < self.min_views:
            raise ConfigError("max_views must be >= min_views")
        if self.outlier_factor <= 1:
            raise ConfigError("outlier_factor must be > 1")
        if min(self.grid) < 1:
            raise ConfigError("grid cells must be >= 1")


@dataclass(frozen=True, slots=True)
class CalibrationReport:
    """Result of `calibrate_views`."""

    calibration: Calibration
    used_views: tuple[int, ...]  # View.index of the views in the final fit
    rejected_views: tuple[int, ...]  # outliers dropped before the refit
    per_view_rms: tuple[float, ...]  # px, aligned with used_views
    coverage: float  # [0, 1], fraction of grid cells holding a corner


def detect_views(
    frames: Iterable[tuple[int, np.ndarray]],
    target: CalibrationTarget,
    *,
    on_view: Callable[[View, np.ndarray], None] | None = None,
) -> tuple[list[View], int]:
    """Detect the target in (index, RGB or grayscale image) pairs.

    `on_view(view, image)` is called for every detection (debug output).
    Returns (views, number of images examined).
    """
    views: list[View] = []
    n = 0
    for index, image in frames:
        n += 1
        view = target.detect(image, index)
        if view is not None:
            views.append(view)
            if on_view is not None:
                on_view(view, image)
    return views, n


def select_views(views: Sequence[View], max_views: int) -> list[View]:
    """At most `max_views` views, evenly spread over the input order (time)."""
    if len(views) <= max_views:
        return list(views)
    keep = np.unique(np.linspace(0, len(views) - 1, max_views).round().astype(int))
    return [views[i] for i in keep]


def coverage(
    views: Sequence[View], image_size: tuple[int, int], grid: tuple[int, int]
) -> float:
    """Fraction of the cells of a `grid` (x, y) over the image that hold at least
    one corner of some view: low coverage leaves the distortion at the frame
    edges unconstrained."""
    w, h = image_size
    gx, gy = grid
    hit = np.zeros((gy, gx), bool)
    for view in views:
        pts = view.image_points
        cx = np.clip((pts[:, 0] + 0.5) * gx // w, 0, gx - 1).astype(int)
        cy = np.clip((pts[:, 1] + 0.5) * gy // h, 0, gy - 1).astype(int)
        hit[cy, cx] = True
    return float(hit.mean())


def _fit(
    views: Sequence[View], image_size: tuple[int, int], flags: int
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(rms, K, dist, std of the intrinsics, per-view RMS) of one OpenCV fit."""
    # Identity start: with CALIB_FIX_ASPECT_RATIO only its fx/fy ratio (1) is used.
    rms, k, dist, _, _, std, _, per_view = cv2.calibrateCameraExtended(
        [v.object_points for v in views],
        [v.image_points for v in views],
        image_size,
        np.eye(3),
        None,
        flags=flags,
    )
    return rms, k, dist, std.ravel(), per_view.ravel()


def calibrate_views(
    views: Sequence[View],
    image_size: tuple[int, int],
    options: CalibrationOptions | None = None,
    *,
    meta: Mapping[str, Any] | None = None,
    camera: Mapping[str, Any] | None = None,
) -> CalibrationReport:
    """Calibrate from views, reject outlier views once and recalibrate.

    Views whose RMS reprojection error exceeds max(outlier_factor * median,
    1 px) are dropped (motion blur, wrong corners). Standard deviations of the
    intrinsics come from calibrateCameraExtended. `meta` and `camera` go into
    the Calibration (calibration file); n_views, coverage and per_view_rms are
    added to `meta`. `options` defaults to CalibrationOptions().

    Raises:
        ConfigError: fewer than min_views usable views.
    """
    options = options or CalibrationOptions()
    if len(views) < options.min_views:
        raise ConfigError(
            f"{len(views)} views with the target, at least {options.min_views} needed"
        )
    flags = (cv2.CALIB_FIX_K3 if options.fix_k3 else 0) | (
        cv2.CALIB_FIX_ASPECT_RATIO if options.fix_aspect else 0
    )
    used = list(views)
    rms, k, dist, std, per_view = _fit(used, image_size, flags)
    limit = max(options.outlier_factor * float(np.median(per_view)), 1.0)
    outliers = per_view > limit
    rejected = tuple(v.index for v, bad in zip(used, outliers, strict=True) if bad)
    if rejected:
        used = [v for v, bad in zip(used, outliers, strict=True) if not bad]
        if len(used) < options.min_views:
            raise ConfigError(
                f"{len(used)} views left after dropping {len(rejected)} outliers, "
                f"at least {options.min_views} needed"
            )
        rms, k, dist, std, per_view = _fit(used, image_size, flags)
    n_std = len(_STD_NAMES) if not options.fix_k3 else len(_STD_NAMES) - 1
    # With k3 fixed it stays 0 and is not written (k1 k2 p1 p2).
    coeffs = dist.ravel()[:4] if options.fix_k3 else dist.ravel()[:5]
    cov = coverage(used, image_size, options.grid)
    calibration = Calibration.from_opencv(
        k,
        coeffs,
        image_size,
        rms_px=float(rms),
        std={name: float(v) for name, v in zip(_STD_NAMES[:n_std], std[:n_std])},
        camera=dict(camera or {}),
        meta={
            **(meta or {}),
            "n_views": len(used),
            "coverage": cov,
            "per_view_rms": {
                "median": float(np.median(per_view)),
                "max": float(per_view.max()),
            },
        },
    )
    return CalibrationReport(
        calibration,
        tuple(v.index for v in used),
        rejected,
        tuple(float(e) for e in per_view),
        cov,
    )


def focal_px_from_35mm(focal_35mm: float, width: int, height: int) -> float:
    """Focal length in pixels implied by a 35 mm-equivalent focal, using the
    diagonal definition (43.27 mm). Diagnostic only: phones report it for the
    sensor, not for the cropped and stabilised video frame (on the task_0007
    iPhone videos it gives 1222 px against 1390/1503 px estimated)."""
    return focal_35mm * float(np.hypot(width, height)) / FULL_FRAME_DIAGONAL_MM
