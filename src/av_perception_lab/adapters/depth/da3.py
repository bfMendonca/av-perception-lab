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

The model kind (and so `needs_intrinsics`) is decided at construction from
the checkpoint's model_name, before loading, so relative models are refused
and focal estimation is scheduled without loading the main model.

Nothing is downloaded at run time: checkpoints are read with
local_files_only=True from `hf_cache` (default `<weights root>/hf`), and a
missing one raises a ConfigError naming `--prepare` before depth_anything_3 is
imported. `da3_requirements` declares the same files (also used for the focal
model of `camera.estimate`).
"""

import json
import os
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, get_args

import numpy as np

from av_perception_lab.adapters._hf_hub import (
    default_cache,
    download_command,
    hf_asset,
    is_repo_id,
    resolve_cache,
)
from av_perception_lab.adapters._weights import INFER_EXTRA
from av_perception_lab.core.components import DepthEstimator, LoadContext
from av_perception_lab.core.options import ConfigError
from av_perception_lab.core.registry import register
from av_perception_lab.core.requirements import (
    HuggingFaceAsset,
    Requirements,
    UnsafePathError,
)
from av_perception_lab.core.types import CameraIntrinsics, DepthResult, MapGeometry
from av_perception_lab.runtime import package_version

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
ProcessResMethod = Literal[
    "upper_bound_resize",
    "upper_bound_crop",
    "lower_bound_resize",
    "lower_bound_crop",
]
PROCESS_RES_METHODS: tuple[str, ...] = get_args(ProcessResMethod)
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
    return geom.scale_intrinsics(
        CameraIntrinsics.centered(focal_frame_px, geom.frame_w, geom.frame_h)
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


_DA3_WEIGHTS = "model.safetensors"


def da3_requirements(model: str, hf_cache: Path | None) -> Requirements:
    """`config.json` + `model.safetensors` of a DA3 alias / repo id in the HF
    cache layout (nothing for a local checkpoint directory)."""
    asset = hf_asset("weights", resolve_depth_model(model), hf_cache, _DA3_WEIGHTS)
    return Requirements(
        assets=() if asset is None else (asset,),
        packages=("torch", "depth_anything_3"),
        extra=INFER_EXTRA,
    )


def _check_cached(repo_id: str, cache: Path) -> None:
    """ConfigError naming `--prepare` when `repo_id` is not in `cache`
    (read from disk, nothing imported)."""
    if not is_repo_id(repo_id):
        return  # a local checkpoint directory
    probe = HuggingFaceAsset(
        key="weights",
        target=cache.name,
        repo=repo_id,
        files=(_DA3_WEIGHTS,),
        patterns=("*.json",),
    )
    try:
        state = probe.present(cache.parent).state
    except (UnsafePathError, OSError):
        return  # e.g. a symlinked cache: local_files_only still never downloads
    if state != "ready":
        raise ConfigError(
            f"model: {repo_id} is not in {cache} (nothing is downloaded at run "
            "time); run `av-perception-lab --config <yaml> --prepare` or download "
            f"it with:\n  {download_command(repo_id, cache, _DA3_WEIGHTS)}"
        )


def _load_da3(repo_id: str, device: str, cache_dir: str | None):
    """The DA3 network from the local cache only (`cache_dir`, default
    `<weights root>/hf`); never downloads."""
    cache = default_cache() if cache_dir is None else resolve_cache(Path(cache_dir))
    _check_cached(repo_id, cache)
    from depth_anything_3.api import DepthAnything3

    try:
        net = DepthAnything3.from_pretrained(
            repo_id, cache_dir=str(cache), local_files_only=True
        )
    except FileNotFoundError:  # LocalEntryNotFoundError: a file the probe missed
        raise ConfigError(
            f"model: {repo_id} is incomplete in {cache}; run "
            "`av-perception-lab --config <yaml> --prepare`"
        ) from None
    return net.to(device=device).eval()


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
        ConfigError: no frames were given, or the model has no camera head
            (prediction.intrinsics is None).
    """
    if not frames:
        raise ConfigError(
            "no frames to estimate the focal from (check source.start/duration)"
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
        raise ConfigError(
            f"{model} has no camera head; pick an any-view model for camera.estimate.model"
        )
    k = np.asarray(pred.intrinsics, dtype=np.float64)
    return float(np.median((k[:, 0, 0] / geom.sx + k[:, 1, 1] / geom.sy) / 2))


@register("depth", "da3")
class DA3Depth(DepthEstimator):
    """Single-frame DA3 inference returning `DepthResult` in metres.

    Relative models are refused at construction. DA3METRIC needs frame
    intrinsics in the LoadContext (scaled once to the map); NESTED models
    predict K per frame and need none.
    """

    @dataclass(frozen=True, slots=True)
    class Options:
        model: str = "da3metric-large"  # alias (DEPTH_MODELS), HF repo id or local dir
        process_res: int = 504
        process_res_method: ProcessResMethod = "upper_bound_resize"
        hf_cache: Path | None = (
            None  # Hugging Face cache dir (default <weights root>/hf)
        )

        def __post_init__(self) -> None:
            if self.process_res < 14:
                raise ValueError("process_res must be >= 14 (DA3 patch size)")

    def __init__(self, options: Options, *, name: str | None = None) -> None:
        self.repo_id = resolve_depth_model(options.model)
        # Kind from the checkpoint's own model_name when it is a local dir (dir
        # names may say nothing), so it is known without loading the model.
        self.kind: DepthKind = classify_depth_model(peek_model_name(self.repo_id))
        if self.kind == "relative":
            raise ConfigError(
                f"model: {self.repo_id} predicts relative depth only; use a metric model"
            )
        super().__init__(options, name=name)
        self.model_name: str | None = None  # checkpoint's model_name, after load
        # Constant map-resolution K from the given focal; None means K is predicted per frame.
        self.fixed_intrinsics: CameraIntrinsics | None = None
        self._net = None

    @classmethod
    def requirements(cls, options: Options) -> Requirements:
        return da3_requirements(options.model, options.hf_cache)

    def default_name(self) -> str:
        return self.repo_id.rstrip("/").split("/")[-1].lower()

    @property
    def needs_intrinsics(self) -> bool:
        return self.kind == "metric_focal"

    def _load(self, ctx: LoadContext) -> None:
        if self.needs_intrinsics and ctx.intrinsics is None:
            raise ConfigError(
                f"{self.repo_id} needs a focal length: set camera.focal_px or camera.estimate"
            )
        opts = self.options
        geometry = da3_map_geometry(
            *ctx.frame_size, opts.process_res, opts.process_res_method
        )
        net = _load_da3(
            self.repo_id, ctx.device, str(opts.hf_cache) if opts.hf_cache else None
        )
        kind = classify_depth_model(net.model_name)
        if kind != self.kind:
            raise ConfigError(
                f"model: checkpoint {net.model_name} is {kind}, but {self.repo_id} "
                f"looked {self.kind} before loading"
            )
        self._net, self.model_name, self._geometry = net, net.model_name, geometry
        if self.needs_intrinsics:
            self.fixed_intrinsics = geometry.scale_intrinsics(ctx.intrinsics)

    def _close(self) -> None:
        import torch

        self._net = None
        self.fixed_intrinsics = None
        torch.cuda.empty_cache()

    def _predict(self, rgb: np.ndarray) -> DepthResult:
        pred = self._net.inference(
            [rgb],
            process_res=self.options.process_res,
            process_res_method=self.options.process_res_method,
        )
        raw = np.asarray(pred.depth[0], dtype=np.float32)
        if self.kind == "metric_focal":
            intrinsics = self.fixed_intrinsics
        else:  # NESTED scales with its own per-frame focal, so report the predicted K
            k = pred.intrinsics[0]
            intrinsics = CameraIntrinsics(
                float(k[0, 0]), float(k[1, 1]), float(k[0, 2]), float(k[1, 2])
            )
        sky = None if pred.sky is None else np.asarray(pred.sky[0], dtype=bool)
        return DepthResult(
            to_metres(raw, self.kind, intrinsics), intrinsics, sky, self._geometry
        )

    def _describe(self) -> dict[str, Any]:
        return {
            "repo_id": self.repo_id,
            "model_name": self.model_name,
            "kind": self.kind,
            "process_res": self.options.process_res,
            "process_res_method": self.options.process_res_method,
            "geometry": asdict(self._geometry) if self._geometry else None,
            "fixed_intrinsics": asdict(self.fixed_intrinsics)
            if self.fixed_intrinsics
            else None,
            "version": package_version("depth-anything-3"),
        }
