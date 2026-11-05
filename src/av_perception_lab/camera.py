"""Camera intrinsics providers (frame-resolution pinhole K).

The focal length belongs to the camera, not to the depth model: providers
return intrinsics at frame resolution and depth adapters rescale them to
their map with `MapGeometry.scale_intrinsics`. Depth models that predict K
per frame declare `needs_intrinsics == False` and need no provider.

A camera calibration file (OpenCV FileStorage YAML/JSON: image size,
camera_matrix, distortion_coefficients, optional std and capture metadata)
is the preferred source: `build_camera` loads it, rescales it to the video
(same aspect ratio only), checks the recording camera against the video's
QuickTime metadata, optionally undistorts frames, and returns a
`CalibratedIntrinsics` provider.

Pixel convention: intrinsics here use continuous coordinates (pixel i is
centred at i + 0.5, as box edges and MapGeometry do); OpenCV centres pixel i
at i, so cx/cy differ by 0.5 and are converted only in Calibration.
"""

import sys
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from av_perception_lab.adapters.depth.da3 import estimate_focal_frame_px
from av_perception_lab.config import (
    CalibrationConfig,
    CameraConfig,
    EstimateConfig,
    LensCheck,
    RectifyConfig,
)
from av_perception_lab.core.options import ConfigError
from av_perception_lab.core.types import CameraIntrinsics
from av_perception_lab.io.mcap_reader import CameraInfoMsg
from av_perception_lab.io.source import FrameSource, FrameTransform

# Valid lengths of an OpenCV distortion vector (k1 k2 p1 p2 [k3 [k4 k5 k6 [s1..s4 [tx ty]]]]).
DIST_COUNTS: frozenset[int] = frozenset({4, 5, 8, 12, 14})
# Metadata compared between the calibration recording and the video (check_lens).
CAMERA_CHECK_KEYS: tuple[str, ...] = ("model", "lens_model", "focal_length_35mm")
# Calibration file keys read into Calibration fields; any other key goes to `meta`.
_FIELD_KEYS = frozenset(
    {
        "image_width",
        "image_height",
        "camera_matrix",
        "distortion_coefficients",
        "rms_px",
        "std_intrinsics",
        "camera",
        "format_version",
    }
)
FORMAT_VERSION = 1


@dataclass(frozen=True, slots=True)
class IntrinsicsResult:
    """Resolved intrinsics and where they came from."""

    intrinsics: CameraIntrinsics  # frame resolution
    source: str  # metrics.json["focal_source"]
    # Provider-specific facts for metrics.json["camera"] (e.g. the cross-check).
    details: Mapping[str, Any] = field(default_factory=dict)


class IntrinsicsProvider(ABC):
    """Resolves the camera intrinsics of a frame source."""

    @abstractmethod
    def resolve(self, source: FrameSource, *, device: str) -> IntrinsicsResult: ...


class FixedIntrinsics(IntrinsicsProvider):
    """A known focal length, principal point at the frame centre."""

    def __init__(self, focal_px: float) -> None:
        if focal_px <= 0:
            raise ConfigError(f"camera.focal_px must be > 0, got {focal_px}")
        self.focal_px = focal_px

    def resolve(self, source: FrameSource, *, device: str) -> IntrinsicsResult:
        return IntrinsicsResult(
            CameraIntrinsics.centered(self.focal_px, *source.frame_size),
            "config (camera.focal_px)",
        )


class EstimatedIntrinsics(IntrinsicsProvider):
    """Focal estimated by an any-view DA3 model over frames sampled from the source.

    The estimation model is loaded, used and released inside `resolve`.
    """

    def __init__(self, cfg: EstimateConfig) -> None:
        self.cfg = cfg

    def resolve(self, source: FrameSource, *, device: str) -> IntrinsicsResult:
        frames = source.sample(self.cfg.frames)
        focal = estimate_focal_frame_px(
            frames,
            self.cfg.model,
            device=device,
            process_res=self.cfg.process_res,
            process_res_method=self.cfg.process_res_method,
            cache_dir=str(self.cfg.hf_cache) if self.cfg.hf_cache else None,
        )
        return IntrinsicsResult(
            CameraIntrinsics.centered(focal, *source.frame_size),
            f"estimated by {self.cfg.model} (median of {len(frames)} frames)",
        )


def focal_estimate_config(
    cfg: CameraConfig, rectify: RectifyConfig | None = None
) -> EstimateConfig | None:
    """`camera.estimate` when a run would use it (no `focal_px`, and either no
    calibration nor rectification, or `calibration.cross_check`), else None.
    Only matters when the depth model needs intrinsics."""
    if cfg.focal_px is not None or cfg.estimate is None:
        return None
    if cfg.calibration is not None:
        return cfg.estimate if cfg.calibration.cross_check else None
    if rectify is not None and rectify.enabled:  # intrinsics from the CameraInfo
        return None
    return cfg.estimate


def build_intrinsics_provider(cfg: CameraConfig) -> IntrinsicsProvider | None:
    """`camera.focal_px` takes precedence over `camera.estimate`; None if neither is set.

    Ignores `camera.calibration`, which needs the source: see `build_camera`.
    """
    if cfg.focal_px is not None:
        return FixedIntrinsics(cfg.focal_px)
    if cfg.estimate is not None:
        return EstimatedIntrinsics(cfg.estimate)
    return None


@dataclass(frozen=True, slots=True)
class Calibration:
    """Camera calibration of one recording mode, with K in continuous pixel
    coordinates. `dist` follows OpenCV's order (k1, k2, p1, p2[, k3, …]);
    empty means no distortion."""

    image_size: tuple[int, int]  # (W, H) of the decoded, upright frame
    intrinsics: CameraIntrinsics  # continuous convention
    dist: tuple[float, ...] = ()
    rms_px: float | None = None  # RMS reprojection error of the calibration
    std: Mapping[str, float] = field(default_factory=dict)  # fx, fy, cx, cy, k1...
    camera: Mapping[str, Any] = field(default_factory=dict)  # recording metadata
    meta: Mapping[str, Any] = field(default_factory=dict)  # target, n_views, tool...

    @classmethod
    def from_opencv(
        cls,
        camera_matrix: np.ndarray,
        dist: np.ndarray,
        image_size: tuple[int, int],
        **fields: Any,
    ) -> "Calibration":
        """From an OpenCV camera matrix (pixel i centred at i): cx/cy gain 0.5."""
        k = np.asarray(camera_matrix, dtype=np.float64)
        intrinsics = CameraIntrinsics(
            fx=float(k[0, 0]),
            fy=float(k[1, 1]),
            cx=float(k[0, 2]) + 0.5,
            cy=float(k[1, 2]) + 0.5,
        )
        coeffs = tuple(float(v) for v in np.ravel(dist))
        return cls(
            (int(image_size[0]), int(image_size[1])), intrinsics, coeffs, **fields
        )

    def camera_matrix(self) -> np.ndarray:
        """(3, 3) float64 camera matrix in OpenCV's convention (cx/cy - 0.5)."""
        k = self.intrinsics
        return np.array(
            [[k.fx, 0.0, k.cx - 0.5], [0.0, k.fy, k.cy - 0.5], [0.0, 0.0, 1.0]]
        )

    def dist_array(self) -> np.ndarray:
        """(1, N) float64 distortion coefficients; zeros(1, 5) when there are none."""
        if not self.dist:
            return np.zeros((1, 5))
        return np.array(self.dist, dtype=np.float64).reshape(1, -1)

    @property
    def has_distortion(self) -> bool:
        return any(self.dist)

    def scaled_to(self, width: int, height: int) -> "Calibration":
        """Rescale to another frame size of the same aspect ratio.

        Distortion coefficients are in normalised coordinates and do not change.
        Another aspect ratio means another sensor crop, so it is refused.

        Raises:
            ConfigError: aspect ratio differs (or portrait vs landscape).
        """
        w0, h0 = self.image_size
        if (width, height) == (w0, h0):
            return self
        s = width / w0
        if abs(height / h0 - s) > 1e-3 * s:
            orientation = (
                " (portrait vs landscape)" if (width > height) != (w0 > h0) else ""
            )
            raise ConfigError(
                f"calibration is for {w0}x{h0} frames, the video is {width}x{height}: "
                f"different aspect ratio{orientation}, i.e. another sensor crop; "
                "calibrate in the video's recording mode"
            )
        k = self.intrinsics
        return replace(
            self,
            image_size=(width, height),
            intrinsics=CameraIntrinsics(k.fx * s, k.fy * s, k.cx * s, k.cy * s),
        )

    def describe(self) -> dict[str, Any]:
        """JSON-ready summary (K in the continuous convention, as everywhere in metrics.json)."""
        return {
            "image_size": list(self.image_size),
            "intrinsics": asdict(self.intrinsics),
            "dist": list(self.dist),
            "rms_px": self.rms_px,
            "std": dict(self.std),
            "camera": dict(self.camera),
        }

    @classmethod
    def from_description(cls, data: Mapping[str, Any]) -> "Calibration":
        """Inverse of `describe` (e.g. metrics.json["camera"]["calibration"]):
        the exact K and distortion a run used, independent of the file.

        Raises:
            ConfigError: a key is missing or malformed.
        """
        try:
            width, height = data["image_size"]
            return cls(
                (int(width), int(height)),
                CameraIntrinsics(**data["intrinsics"]),
                tuple(float(v) for v in data["dist"]),
                rms_px=data.get("rms_px"),
                std=dict(data.get("std") or {}),
                camera=dict(data.get("camera") or {}),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ConfigError(f"invalid calibration description ({exc!r})") from None


_CAMERA_INFO_MODELS = frozenset({"plumb_bob", "rational_polynomial"})


def has_distortion(msg: CameraInfoMsg) -> bool:
    """True for a raw CameraInfo (D non-empty and not all zero)."""
    return any(msg.d)


def calibration_from_camera_info(
    msg: CameraInfoMsg, image_size: tuple[int, int]
) -> Calibration:
    """Calibration from a ROS `CameraInfo`.

    Raw (distorted) messages use K and D; rectified ones (D empty or all zero)
    use the left 3x3 of P with no distortion. K/P are in OpenCV's convention,
    so `Calibration.from_opencv` adds the 0.5 of the continuous convention.

    Raises:
        ConfigError: unsupported model, ROI, binning, bad D length, or a
            CameraInfo size different from `image_size` (W, H).
    """
    w, h = int(image_size[0]), int(image_size[1])
    if (msg.width, msg.height) != (w, h):
        raise ConfigError(
            f"CameraInfo is {msg.width}x{msg.height}, the frames are {w}x{h}"
        )
    if msg.model and msg.model not in _CAMERA_INFO_MODELS:
        raise ConfigError(
            f"CameraInfo distortion_model {msg.model!r} is not supported "
            f"(use one of {sorted(_CAMERA_INFO_MODELS)})"
        )
    if any(msg.roi):
        raise ConfigError(
            f"CameraInfo roi {tuple(msg.roi)} is not supported (non-zero)"
        )
    if msg.binning > 1:
        raise ConfigError(f"CameraInfo binning {msg.binning} is not supported (> 1)")
    if has_distortion(msg):
        if len(msg.d) not in DIST_COUNTS:
            raise ConfigError(
                f"CameraInfo D has {len(msg.d)} coefficients, expected one of "
                f"{sorted(DIST_COUNTS)}"
            )
        matrix = np.array(msg.k, dtype=np.float64).reshape(3, 3)
        return Calibration.from_opencv(matrix, np.array(msg.d), (w, h))
    matrix = np.array(msg.p, dtype=np.float64).reshape(3, 4)[:, :3]
    return Calibration.from_opencv(matrix, np.zeros(0), (w, h))


def _node_to_python(node: cv2.FileNode) -> Any:
    """A FileStorage node as plain Python (matrices become nested lists)."""
    if node.isMap():
        keys = node.keys()
        if {"rows", "cols", "dt", "data"} <= set(keys):
            return node.mat().tolist()
        return {k: _node_to_python(node.getNode(k)) for k in keys}
    if node.isSeq():
        return [_node_to_python(node.at(i)) for i in range(node.size())]
    if node.isInt():
        return int(node.real())
    if node.isReal():
        return node.real()
    if node.isString():
        return node.string()
    return None


def _matrix(fs: cv2.FileStorage, key: str, path: Path) -> np.ndarray:
    node = fs.getNode(key)
    if node.empty() or node.isNone():
        raise ConfigError(f"{path}: missing required key {key!r}")
    keys = node.keys()  # FileNode.keys(), not a dict
    if not node.isMap() or "data" not in keys:
        raise ConfigError(f"{path}: {key} must be an opencv-matrix")
    return np.asarray(node.mat(), dtype=np.float64)


def _validate_matrix(k: np.ndarray, dist: np.ndarray, path: Path) -> None:
    if k.shape != (3, 3):
        raise ConfigError(f"{path}: camera_matrix must be 3x3, got {k.shape}")
    if not np.allclose(k[2], (0.0, 0.0, 1.0)) or k[1, 0] != 0.0:
        raise ConfigError(f"{path}: camera_matrix last row must be (0, 0, 1)")
    if k[0, 1] != 0.0:
        raise ConfigError(f"{path}: camera_matrix has skew {k[0, 1]}; not supported")
    if k[0, 0] <= 0 or k[1, 1] <= 0:
        raise ConfigError(f"{path}: camera_matrix fx and fy must be > 0")
    if dist.size not in DIST_COUNTS or 1 not in dist.shape[:2]:
        raise ConfigError(
            f"{path}: distortion_coefficients must be 1xN with N in "
            f"{sorted(DIST_COUNTS)}, got {dist.shape}"
        )


def load_calibration(path: Path) -> Calibration:
    """Read a calibration file (OpenCV FileStorage YAML, JSON or XML).

    Required keys: image_width, image_height, camera_matrix (3x3, OpenCV
    convention), distortion_coefficients (1xN). Optional: rms_px,
    std_intrinsics, camera; every other key is kept in `meta`.

    Raises:
        ConfigError: missing or unreadable file, missing key or invalid values
            (the message starts with the file path).
    """
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"calibration file not found: {path}")
    try:
        fs = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
    except (cv2.error, SystemError):  # the binding wraps cv2.error in SystemError
        raise ConfigError(
            f"{path}: not an OpenCV FileStorage file (YAML needs the %YAML:1.0 header)"
        ) from None
    try:
        version = _node_to_python(fs.getNode("format_version"))
        if version is not None and version != FORMAT_VERSION:
            raise ConfigError(f"{path}: unsupported format_version {version}")
        size = []
        for key in ("image_width", "image_height"):
            node = fs.getNode(key)
            if node.empty() or not node.isInt() or node.real() <= 0:
                raise ConfigError(f"{path}: {key} must be a positive integer")
            size.append(int(node.real()))
        k = _matrix(fs, "camera_matrix", path)
        dist = _matrix(fs, "distortion_coefficients", path)
        _validate_matrix(k, dist, path)
        names = fs.root().keys()
        values = {key: _node_to_python(fs.getNode(key)) for key in names}
    finally:
        fs.release()
    rms, std, camera = (
        values.get(key) for key in ("rms_px", "std_intrinsics", "camera")
    )
    if rms is not None and not isinstance(rms, int | float):
        raise ConfigError(f"{path}: rms_px must be a number")
    if std is not None and not (
        isinstance(std, dict) and all(isinstance(v, int | float) for v in std.values())
    ):
        raise ConfigError(f"{path}: std_intrinsics must map names to numbers")
    if camera is not None and not isinstance(camera, dict):
        raise ConfigError(f"{path}: camera must be a map")
    return Calibration.from_opencv(
        k,
        dist,
        (size[0], size[1]),
        rms_px=float(rms) if rms is not None else None,
        std={n: float(v) for n, v in (std or {}).items()},
        camera=camera or {},
        meta={key: value for key, value in values.items() if key not in _FIELD_KEYS},
    )


def _write(fs: cv2.FileStorage, name: str, value: Any) -> None:
    """Write a plain Python value (nested maps and lists included); None is skipped."""
    if value is None:
        return
    if isinstance(value, Mapping):
        fs.startWriteStruct(name, cv2.FileNode_MAP)
        for key, item in value.items():
            _write(fs, str(key), item)
        fs.endWriteStruct()
    elif isinstance(value, list | tuple):
        fs.startWriteStruct(name, cv2.FileNode_SEQ)
        for item in value:
            _write(fs, "", item)
        fs.endWriteStruct()
    elif isinstance(value, bool):  # FileStorage has no boolean type
        fs.write(name, str(value).lower())
    elif isinstance(value, int | float | str | np.ndarray):
        fs.write(name, value)
    else:
        raise TypeError(f"cannot write {name}={value!r} to a calibration file")


def save_calibration(calibration: Calibration, path: Path) -> None:
    """Write a calibration file; the format follows the extension (.yaml/.yml,
    .json or .xml). The camera matrix is written in OpenCV's convention, so
    any OpenCV tool reads the file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fs = cv2.FileStorage(str(path), cv2.FILE_STORAGE_WRITE)
    try:
        fs.write("format_version", FORMAT_VERSION)
        fs.write("image_width", calibration.image_size[0])
        fs.write("image_height", calibration.image_size[1])
        fs.write("camera_matrix", calibration.camera_matrix())
        fs.write("distortion_coefficients", calibration.dist_array())
        _write(fs, "rms_px", calibration.rms_px)
        _write(fs, "std_intrinsics", dict(calibration.std) or None)
        _write(fs, "camera", dict(calibration.camera) or None)
        for key, value in calibration.meta.items():
            _write(fs, key, value)
    finally:
        fs.release()


class Undistorter(FrameTransform):
    """Frame transform removing lens distortion with precomputed remap tables.

    The output camera is pinhole (no distortion) with the matrix from
    getOptimalNewCameraMatrix(alpha): alpha 0 keeps only valid pixels (slightly
    longer focal), 1 keeps the whole field of view (black borders). Frame size
    is unchanged, so the depth-map geometry is unchanged. Everything downstream
    (detections, depth, CSV boxes, video) is in undistorted coordinates.
    """

    def __init__(self, calibration: Calibration, *, alpha: float = 0.0) -> None:
        self.calibration = calibration
        self.alpha = alpha
        k, dist, size = (
            calibration.camera_matrix(),
            calibration.dist_array(),
            calibration.image_size,
        )
        new_k, _ = cv2.getOptimalNewCameraMatrix(k, dist, size, alpha, size)
        self._new_k = new_k
        # Fixed-point maps (CV_16SC2): OpenCV's fastest representation for remap.
        self._map1, self._map2 = cv2.initUndistortRectifyMap(
            k, dist, None, new_k, size, cv2.CV_16SC2
        )
        self._output = replace(
            Calibration.from_opencv(new_k, np.zeros(0), size),
            rms_px=calibration.rms_px,
            std=calibration.std,
            camera=calibration.camera,
            meta=calibration.meta,
        )

    @property
    def output(self) -> Calibration:
        """The undistorted camera: new K, no distortion, same image size."""
        return self._output

    def __call__(self, rgb: np.ndarray) -> np.ndarray:
        return cv2.remap(rgb, self._map1, self._map2, cv2.INTER_LINEAR)

    def undistort_points(self, xy: np.ndarray) -> np.ndarray:
        """(N, 2) continuous pixel coordinates of the original video -> the same
        points in the undistorted video (continuous coordinates)."""
        pts = np.asarray(xy, dtype=np.float64).reshape(-1, 1, 2) - 0.5
        out = cv2.undistortPointsIter(
            pts,
            self.calibration.camera_matrix(),
            self.calibration.dist_array(),
            None,
            self._new_k,
            # Iterate to convergence: the default 5 iterations leave ~2e-3 px at
            # 1080p frame corners (k1 0.1-0.25); exact points cost nothing here.
            (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 100, 1e-10),
        )
        return out.reshape(-1, 2) + 0.5

    def describe(self) -> dict[str, Any]:
        return {"alpha": self.alpha, "intrinsics": asdict(self._output.intrinsics)}


def check_camera_metadata(
    expected: Mapping[str, Any], actual: Mapping[str, Any], policy: LensCheck
) -> dict[str, Any]:
    """Compare the calibration's recording camera with the video's.

    Only CAMERA_CHECK_KEYS present on both sides are compared. Status: "off"
    (policy off), "unavailable" (nothing to compare, e.g. a non-Apple video or
    a calibration from images), "ok" or "mismatch" (a warning on stderr with
    policy "warn").

    Raises:
        ConfigError: a mismatch with policy "error".
    """
    if policy == "off":
        return {"status": "off", "compared": {}, "differences": {}}
    compared = {
        key: {"calibration": expected[key], "video": actual[key]}
        for key in CAMERA_CHECK_KEYS
        if key in expected and key in actual
    }
    differences = {k: v for k, v in compared.items() if v["calibration"] != v["video"]}
    if not compared:
        status = "unavailable"
    else:
        status = "mismatch" if differences else "ok"
    if differences:
        text = "; ".join(
            f"{k}: calibration {v['calibration']!r}, video {v['video']!r}"
            for k, v in differences.items()
        )
        if policy == "error":
            raise ConfigError(
                f"camera.calibration: recorded with another camera ({text}); "
                "set camera.calibration.check_lens to warn or off to use it anyway"
            )
        print(f"[warn] camera.calibration: camera mismatch ({text})", file=sys.stderr)
    return {"status": status, "compared": compared, "differences": differences}


class CalibratedIntrinsics(IntrinsicsProvider):
    """Intrinsics from a calibration file, already rescaled (and undistorted when
    frames are). With `cross_check` the focal is also estimated from the video
    and both are reported, to catch a calibration made in another mode
    (stabilisation crop, zoom)."""

    def __init__(
        self,
        calibration: Calibration,
        *,
        label: str,
        cross_check: EstimatedIntrinsics | None = None,
    ) -> None:
        self.calibration = calibration
        self.label = label
        self.cross_check = cross_check

    def resolve(self, source: FrameSource, *, device: str) -> IntrinsicsResult:
        """Raises ConfigError when the source frames are not the calibrated size."""
        if tuple(source.frame_size) != self.calibration.image_size:
            w, h = self.calibration.image_size
            raise ConfigError(
                f"camera.calibration: calibrated for {w}x{h} frames, the source "
                f"yields {source.frame_size[0]}x{source.frame_size[1]}"
            )
        k = self.calibration.intrinsics
        details: dict[str, Any] = {}
        if self.cross_check is not None:
            estimated = self.cross_check.resolve(source, device=device)
            f_est = estimated.intrinsics.fx
            details = {
                "estimated_focal_frame_px": f_est,
                "estimated_source": estimated.source,
                # The DA3METRIC distance is proportional to the focal: this ratio
                # is the distance ratio between the two runs.
                "estimated_over_calibrated": f_est / ((k.fx + k.fy) / 2),
            }
        return IntrinsicsResult(k, self.label, details)


@dataclass(frozen=True, slots=True)
class CameraSetup:
    """What the camera configuration contributes to a run."""

    source: FrameSource  # with the undistortion transform, if any
    intrinsics: IntrinsicsProvider | None
    details: Mapping[str, Any]  # metrics.json["camera"]


NO_CAMERA_INFO = "no CameraInfo for this topic"


def _camera_info(source: FrameSource, option: str) -> CameraInfoMsg:
    """The source's CameraInfo, or ConfigError naming `option`."""
    info = source.camera_info()
    if info is None:
        raise ConfigError(f"{option}: {NO_CAMERA_INFO} ({source.name})")
    return info


def build_camera(
    cfg: CameraConfig, source: FrameSource, rectify: RectifyConfig | None = None
) -> CameraSetup:
    """Compose the camera of a run (composition root of `camera.*` and `rectify`).

    Without `camera.calibration` and with rectification off the source is
    unchanged and the provider is `build_intrinsics_provider(cfg)`. Otherwise
    the calibration comes from, by precedence:

    1. `rectify.enabled`: the source's CameraInfo, which must have distortion;
       the frames are undistorted with it (`rectify.alpha`) and the provider
       gives the rectified K. A `camera.calibration.path` is then ignored with
       a warning (K and undistortion come from the CameraInfo, so the file
       cannot undistort the frames twice); its `cross_check` still applies.
    2. `camera.calibration.from_source`: the source's CameraInfo, frames unchanged.
    3. `camera.calibration.path`: the file, rescaled to the frames, checked
       against the recording camera, optionally undistorting the frames.

    `details` (metrics.json["camera"]) has the same keys in every case:
    `calibration` holds the K and distortion before any undistortion and
    `undistort` the rectified K, so the evaluation can transform references.
    Runs before any model is loaded.

    Raises:
        ConfigError: unreadable calibration, another aspect ratio, a camera
            mismatch with check_lens "error", no CameraInfo for rectify or
            from_source, or an already-rectified CameraInfo with rectify.
    """
    rectifying = rectify is not None and rectify.enabled
    c: CalibrationConfig | None = cfg.calibration
    if c is None and not rectifying:
        return CameraSetup(source, build_intrinsics_provider(cfg), {})
    ignored: str | None = None
    info: CameraInfoMsg | None = None
    if rectifying:
        info = _camera_info(source, "rectify.enabled")
        if not has_distortion(info):
            raise ConfigError(
                f"rectify.enabled: the CameraInfo of this topic is already "
                f"rectified (no distortion) ({source.name}); disable rectify"
            )
        if c is not None and c.path is not None:
            ignored = str(c.path)
            print(
                f"[warn] camera.calibration.path ignored ({ignored}): rectify uses "
                "the CameraInfo of the source",
                file=sys.stderr,
            )
    elif c is not None and c.from_source:
        info = _camera_info(source, "camera.calibration.from_source")
    if info is not None:
        loaded = calibration_from_camera_info(info, source.frame_size)
        origin: dict[str, Any] = {"path": None, "source": f"CameraInfo ({source.name})"}
        label = origin["source"]
    else:
        assert c is not None and c.path is not None
        loaded = load_calibration(c.path)
        origin = {"path": str(c.path)}
        label = f"calibration {c.path}"
    calibration = loaded.scaled_to(*source.frame_size)
    details: dict[str, Any] = {
        "calibration": {**origin, **calibration.describe()},
        "scaled_from": (
            list(loaded.image_size)
            if loaded.image_size != calibration.image_size
            else None
        ),
        # A source without recording metadata (an MCAP) gives "unavailable".
        "lens_check": check_camera_metadata(
            calibration.camera,
            source.camera_metadata(),
            c.check_lens if c is not None else "off",
        ),
        "undistort": None,
    }
    if ignored is not None:
        details["ignored_calibration"] = ignored
    alpha = None
    if rectifying:
        assert rectify is not None
        alpha, suffix = rectify.alpha, " (rectified)"
    elif c is not None and c.undistort:
        alpha, suffix = c.alpha, " (undistorted)"
    if alpha is not None:
        undistorter = Undistorter(calibration, alpha=alpha)
        source = source.with_transform(undistorter)
        calibration = undistorter.output
        details["undistort"] = undistorter.describe()
        label += suffix
    cross = (
        EstimatedIntrinsics(cfg.estimate) if c is not None and c.cross_check else None
    )
    return CameraSetup(
        source,
        CalibratedIntrinsics(calibration, label=label, cross_check=cross),
        details,
    )
