"""Depth Anything 3 wrapper that always yields metric depth in metres.

Model kinds are decided from the checkpoint's own `model_name`
(config.json), not from `Prediction.is_metric`, which is an empty addict
Dict for DA3METRIC. DA3METRIC outputs `metres * 300 / focal`, with focal =
(fx + fy) / 2 at the processed resolution (see
depth_anything_3.utils.alignment.apply_metric_scaling); NESTED models are
already metric; relative models are rejected.

Frames are inferred one per call: passing several images to an any-view
model switches it to multi-view mode. torch / depth_anything_3 are imported
lazily so the pure helpers (geometry, focal, unit conversion) run without
the GPU stack. DA3_LOG_LEVEL defaults to WARN to silence per-call INFO logs.
"""

import json
import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from det_depth_distance.distance import CameraIntrinsics, MapGeometry

# Read by depth_anything_3's logger at import time, so it must be set before the lazy import.
os.environ.setdefault("DA3_LOG_LEVEL", "WARN")

DepthKind = Literal["metric_focal", "metric", "relative"]

# Short alias -> Hugging Face repo. Any other value is passed through as a repo id / local dir.
DEPTH_MODELS: dict[str, str] = {
    "da3-small": "depth-anything/DA3-SMALL",
    "da3-base": "depth-anything/DA3-BASE",
    "da3-large": "depth-anything/DA3-LARGE",
    "da3-large-1.1": "depth-anything/DA3-LARGE-1.1",
    "da3-giant": "depth-anything/DA3-GIANT",
    "da3-giant-1.1": "depth-anything/DA3-GIANT-1.1",
    "da3mono-large": "depth-anything/DA3MONO-LARGE",
    "da3metric-large": "depth-anything/DA3METRIC-LARGE",
    "da3nested-giant-large": "depth-anything/DA3NESTED-GIANT-LARGE",
    "da3nested-giant-large-1.1": "depth-anything/DA3NESTED-GIANT-LARGE-1.1",
}
PROCESS_RES_METHODS = (
    "upper_bound_resize",
    "upper_bound_crop",
    "lower_bound_resize",
    "lower_bound_crop",
)
DA3_METRIC_SCALE = 300.0  # apply_metric_scaling(scale_factor=300.0)
_PATCH = 14  # DA3 InputProcessor.PATCH_SIZE


def resolve_depth_model(ref: str) -> str:
    """Alias -> HF repo id; anything else (repo id, local dir) is returned unchanged."""
    return DEPTH_MODELS.get(ref.lower(), ref)


def peek_model_name(ref: str) -> str:
    """`model_name` from a local checkpoint dir's config.json, else `ref` unchanged.

    Lets callers classify a model before loading it (e.g. to decide whether the
    focal must be estimated) even when the directory name does not reveal the kind.
    """
    config = Path(ref) / "config.json"
    if config.is_file():
        name = json.loads(config.read_text()).get("model_name")
        if isinstance(name, str):
            return name
    return ref


def classify_depth_model(name: str) -> DepthKind:
    """Kind of a DA3 model from its `model_name` (e.g. "da3metric-large") or repo / dir name."""
    base = name.rstrip("/").split("/")[-1].lower()
    if base.startswith("da3metric"):
        return "metric_focal"
    if base.startswith("da3nested"):
        return "metric"
    return "relative"


def _nearest_multiple(x: int, p: int) -> int:
    down = (x // p) * p
    up = down + p
    return up if abs(up - x) <= abs(x - down) else down


def da3_map_geometry(
    frame_w: int, frame_h: int, process_res: int, method: str
) -> MapGeometry:
    """Replicate DA3 InputProcessor sizing for one frame size.

    Mirrors `_resize_longest_side` / `_resize_shortest_side` (Python round)
    followed by `_make_divisible_by_resize` (nearest multiple of 14, ties up) or
    `_make_divisible_by_crop` (floor + centre crop). 1920x1080 @ 504,
    upper_bound_resize -> 504x280.
    """
    if method not in PROCESS_RES_METHODS:
        raise ValueError(f"unsupported process_res_method {method!r}")
    ref = max(frame_w, frame_h) if method.startswith("upper") else min(frame_w, frame_h)
    if ref == process_res:
        w, h = frame_w, frame_h
    else:
        scale = process_res / float(ref)
        w, h = max(1, round(frame_w * scale)), max(1, round(frame_h * scale))
    if method.endswith("resize"):
        map_w = max(1, _nearest_multiple(w, _PATCH))
        map_h = max(1, _nearest_multiple(h, _PATCH))
        return MapGeometry(
            frame_w, frame_h, map_w, map_h, map_w / frame_w, map_h / frame_h
        )
    map_w, map_h = (w // _PATCH) * _PATCH, (h // _PATCH) * _PATCH
    left, top = (w - map_w) // 2, (h - map_h) // 2
    return MapGeometry(
        frame_w, frame_h, map_w, map_h, w / frame_w, h / frame_h, -left, -top
    )


def intrinsics_from_focal(focal_frame_px: float, geom: MapGeometry) -> CameraIntrinsics:
    """Depth-map intrinsics for a focal length given in frame pixels.

    The principal point is assumed at the frame centre.
    """
    return CameraIntrinsics(
        fx=focal_frame_px * geom.sx,
        fy=focal_frame_px * geom.sy,
        cx=geom.frame_w / 2 * geom.sx + geom.ox,
        cy=geom.frame_h / 2 * geom.sy + geom.oy,
    )


def to_metres(
    raw: np.ndarray, kind: DepthKind, intrinsics: CameraIntrinsics | None
) -> np.ndarray:
    """Convert raw DA3 output to metres (float32).

    Raises:
        ValueError: relative model, or a metric_focal model without intrinsics.
    """
    if kind == "metric":
        return raw.astype(np.float32, copy=False)
    if kind == "metric_focal":
        if intrinsics is None:
            raise ValueError("DA3METRIC output needs the focal length to become metres")
        focal = (intrinsics.fx + intrinsics.fy) / 2
        return (raw * (focal / DA3_METRIC_SCALE)).astype(np.float32, copy=False)
    raise ValueError("relative-depth models cannot produce metric distances")


@dataclass(frozen=True, slots=True)
class DepthFrame:
    """Metric depth of one frame at depth-map resolution."""

    depth_m: np.ndarray  # (map_h, map_w) float32, metres along the optical axis (z)
    intrinsics: CameraIntrinsics  # at depth-map resolution
    # bool (map_h, map_w), or None when the model has no sky head
    sky: np.ndarray | None


def _load_da3(repo_id: str, device: str, cache_dir: str | None):
    from depth_anything_3.api import DepthAnything3

    return (
        DepthAnything3.from_pretrained(repo_id, cache_dir=cache_dir)
        .to(device=device)
        .eval()
    )


def estimate_focal_frame_px(
    frames: Sequence[np.ndarray],
    model: str,
    *,
    device: str,
    process_res: int,
    process_res_method: str,
    cache_dir: str | None = None,
) -> float:
    """Estimate the focal length in *frame* pixels with an any-view DA3 model.

    Runs one multi-view inference over `frames`, converts each predicted K back
    to frame pixels ((fx/sx + fy/sy) / 2) and returns the median. The model is
    released (and the CUDA cache emptied) before returning.

    Raises:
        ValueError: no frames were given, or the model has no camera head
            (prediction.intrinsics is None).
    """
    if not frames:
        raise ValueError(
            "no frames to estimate the focal from (check --start/--duration)"
        )
    import torch

    h, w = frames[0].shape[:2]
    geom = da3_map_geometry(w, h, process_res, process_res_method)
    net = _load_da3(resolve_depth_model(model), device, cache_dir)
    try:
        pred = net.inference(
            list(frames), process_res=process_res, process_res_method=process_res_method
        )
    finally:
        del net
        torch.cuda.empty_cache()
    if pred.intrinsics is None:
        raise ValueError(
            f"{model} has no camera head; pick an any-view model for --focal-model"
        )
    k = np.asarray(pred.intrinsics, dtype=np.float64)
    return float(np.median((k[:, 0, 0] / geom.sx + k[:, 1, 1] / geom.sy) / 2))


class DepthEstimator:
    """Single-frame DA3 inference returning `DepthFrame` in metres.

    Raises at construction if the model is relative, or if a DA3METRIC model
    is given no focal length. Checks on the first frame that the depth map
    shape matches `geometry` (guards the InputProcessor replica).
    """

    def __init__(
        self,
        model: str,
        *,
        frame_size: tuple[int, int],
        device: str,
        process_res: int = 504,
        process_res_method: str = "upper_bound_resize",
        focal_frame_px: float | None = None,
        cache_dir: str | None = None,
    ) -> None:
        self.repo_id = resolve_depth_model(model)
        self.geometry = da3_map_geometry(*frame_size, process_res, process_res_method)
        self._process_res = process_res
        self._process_res_method = process_res_method
        self._net = _load_da3(self.repo_id, device, cache_dir)
        self.model_name: str = self._net.model_name
        self.kind = classify_depth_model(self.model_name)
        if self.kind == "relative":
            raise ValueError(
                f"{self.repo_id} ({self.model_name}) predicts relative depth only"
            )
        if self.kind == "metric_focal" and not focal_frame_px:
            raise ValueError(
                f"{self.model_name} needs a focal length (--focal-px or estimation)"
            )
        # Constant map-resolution K from the given focal; None means K is predicted per frame.
        self.fixed_intrinsics = (
            intrinsics_from_focal(focal_frame_px, self.geometry)
            if self.kind == "metric_focal"
            else None
        )
        self._shape_checked = False

    def __call__(self, rgb: np.ndarray) -> DepthFrame:
        pred = self._net.inference(
            [rgb],
            process_res=self._process_res,
            process_res_method=self._process_res_method,
        )
        raw = np.asarray(pred.depth[0], dtype=np.float32)
        if not self._shape_checked:
            expected = (self.geometry.map_h, self.geometry.map_w)
            if raw.shape != expected:
                raise RuntimeError(
                    f"DA3 depth shape {raw.shape} != replicated geometry {expected}"
                )
            self._shape_checked = True
        if self.kind == "metric_focal":
            intrinsics = self.fixed_intrinsics
        else:  # NESTED scales with its own per-frame focal, so report the predicted K
            k = pred.intrinsics[0]
            intrinsics = CameraIntrinsics(
                float(k[0, 0]), float(k[1, 1]), float(k[0, 2]), float(k[1, 2])
            )
        sky = None if pred.sky is None else np.asarray(pred.sky[0], dtype=bool)
        return DepthFrame(to_metres(raw, self.kind, intrinsics), intrinsics, sky)
