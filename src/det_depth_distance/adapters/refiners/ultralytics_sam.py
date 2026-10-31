"""Ultralytics SAM refiner (MobileSAM, SAM, SAM 2.x .pt files).

A second backend for BoxPromptRefiner, mainly for MobileSAM. Uses the
predictor API directly (set_image + inference_features) because
SAM.predict drops masks under `conf` and breaks the one-mask-per-box
contract. Frames are converted to BGR here, as Ultralytics expects.
Ultralytics letterboxes SAM 2.x inputs instead of the square resize of
the original SAM 2: prefer the `sam2` refiner for SAM 2.1. The weights
file must exist and keep its canonical name (e.g. mobile_sam.pt), from which
Ultralytics picks the architecture; it is checked before importing
ultralytics, which would otherwise download it. A relative `weights` (a bare
file name) resolves under `<weights root>/ultralytics/` and release names are
declared as URL assets for `--prepare` (adapters._ultralytics). No object score. The
Ultralytics environment policy (offline, no auto-install) is applied when
this module is imported. Ultralytics is AGPL-3.0.
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from det_depth_distance.adapters._ultralytics import (
    ASSETS_URL,
    check_weights,
    resolve_weights,
    weights_asset,
)
from det_depth_distance.adapters._ultralytics_env import apply_ultralytics_env
from det_depth_distance.adapters._weights import INFER_EXTRA, not_prepared, under_root
from det_depth_distance.adapters.refiners.box_prompt import (
    BoxPromptOptions,
    BoxPromptRefiner,
    DecodedMasks,
)
from det_depth_distance.core.components import LoadContext
from det_depth_distance.core.options import ConfigError
from det_depth_distance.core.registry import register
from det_depth_distance.core.requirements import Requirements
from det_depth_distance.core.types import ContractError
from det_depth_distance.runtime import package_version

apply_ultralytics_env()

# Checkpoint names known to ultralytics.models.sam.build.sam_model_map (8.4.175),
# matched as a suffix of the file name; SAM 3 has its own interactive path.
ULTRALYTICS_SAM_WEIGHTS: tuple[str, ...] = (
    "sam_h.pt",
    "sam_l.pt",
    "sam_b.pt",
    "mobile_sam.pt",
    "sam2_t.pt",
    "sam2_s.pt",
    "sam2_b.pt",
    "sam2_l.pt",
    "sam2.1_t.pt",
    "sam2.1_s.pt",
    "sam2.1_b.pt",
    "sam2.1_l.pt",
)


@dataclass(frozen=True, slots=True)
class _UltralyticsFrame:
    """Per-frame state of `_prepare`."""

    features: Any  # image-encoder output (tensor for SAM, dict for SAM 2)
    size: tuple[int, int]  # (H, W) of the frame
    xyxy: np.ndarray  # (K, 4) float32 target boxes, frame pixels


@register("refiner", "ultralytics_sam")
class UltralyticsSamRefiner(BoxPromptRefiner):
    """An Ultralytics SAM / MobileSAM / SAM 2.x checkpoint (has_object_score is
    False: the predictor discards it)."""

    @dataclass(frozen=True, slots=True, kw_only=True)
    class Options(BoxPromptOptions):
        # Existence is checked in _load (before the import), the name here.
        weights: Path
        imgsz: int = 1024  # square input size, multiple of 32

        def __post_init__(self) -> None:
            # Explicit base call: zero-arg super() fails in slots dataclasses.
            BoxPromptOptions.__post_init__(self)
            check_weights(self.weights)
            if self.imgsz < 32 or self.imgsz % 32:
                raise ValueError(
                    f"imgsz must be a positive multiple of 32, got {self.imgsz}"
                )

    def __init__(self, options: Options, *, name: str | None = None) -> None:
        """Raises ConfigError for a weights name Ultralytics cannot build (no import)."""
        super().__init__(options, name=name)
        if not options.weights.name.endswith(ULTRALYTICS_SAM_WEIGHTS):
            raise ConfigError(
                f"weights: {options.weights.name!r} is not an Ultralytics SAM "
                f"checkpoint name ({', '.join(ULTRALYTICS_SAM_WEIGHTS)})"
            )
        self._predictor = None

    @classmethod
    def requirements(cls, options: Options) -> Requirements:
        asset = weights_asset(
            options.weights, "provide the Ultralytics SAM checkpoint (canonical name)"
        )
        return Requirements(
            assets=() if asset is None else (asset,),
            packages=("torch", "ultralytics"),
            extra=INFER_EXTRA,
        )

    def default_name(self) -> str:
        """Stem of the weights file, e.g. "mobile_sam"."""
        return self.options.weights.stem

    def _load(self, ctx: LoadContext) -> None:
        options = self.options
        weights = resolve_weights(options.weights)
        # Checked first: ultralytics would otherwise try to download a missing file.
        if not weights.is_file():
            if under_root(weights) is not None:
                raise not_prepared("weights", "file", weights)
            asset = next(n for n in ULTRALYTICS_SAM_WEIGHTS if weights.name.endswith(n))
            raise ConfigError(
                f"weights: file not found: {weights}; download it with:\n"
                f"  curl -fL -o {weights} {ASSETS_URL}/{asset}"
            )
        # ultralytics only creates its settings dir when the parent already exists.
        Path(os.environ["YOLO_CONFIG_DIR"]).mkdir(parents=True, exist_ok=True)
        from ultralytics import SAM

        sam = SAM(str(weights))
        # conf 0: nothing is filtered (inference_features does not filter anyway).
        overrides = {
            "imgsz": options.imgsz,
            "conf": 0.0,
            "device": ctx.device,
            "verbose": False,
            "save": False,
        }
        predictor = sam.task_map["segment"]["predictor"](overrides=overrides)
        predictor.setup_model(model=sam.model, verbose=False)
        predictor.model.mask_threshold = options.mask_threshold
        self._predictor = predictor

    def _prepare(self, rgb: np.ndarray, xyxy: np.ndarray) -> _UltralyticsFrame:
        # RGB -> BGR here, and only here: the predictor assumes OpenCV channel order.
        # set_image and inference_features run under ultralytics' own inference mode.
        self._predictor.set_image(np.ascontiguousarray(rgb[..., ::-1]))
        return _UltralyticsFrame(self._predictor.features, rgb.shape[:2], xyxy)

    def _decode(self, state: _UltralyticsFrame, batch: slice) -> DecodedMasks:
        masks, boxes = self._predictor.inference_features(
            state.features,
            state.size,
            # A copy: Ultralytics scales the prompt boxes in place (`bboxes *= r`)
            # after torch.as_tensor, which shares memory with a float32 array on CPU.
            bboxes=state.xyxy[batch].copy(),
            multimask_output=self.options.multimask,
        )
        masks, scores = split_flat(
            masks.cpu().numpy(),
            boxes[:, 4].float().cpu().numpy(),
            batch.stop - batch.start,
        )
        return DecodedMasks(masks, scores)

    def _close(self) -> None:
        import torch

        self._predictor = None
        torch.cuda.empty_cache()

    def _describe(self) -> dict[str, Any]:
        return {"version": package_version("ultralytics")}


def split_flat(
    masks: np.ndarray, scores: np.ndarray, k: int
) -> tuple[np.ndarray, np.ndarray]:
    """(k*C, H, W) masks and (k*C,) scores, box-major as Ultralytics flattens
    them, -> (k, C, H, W) and (k, C).

    Raises:
        ContractError: the counts are not a multiple of k.
    """
    if k < 1 or len(scores) % k or len(masks) != len(scores):
        raise ContractError(f"{len(masks)} masks / {len(scores)} scores for {k} boxes")
    c = len(scores) // k
    return masks.reshape(k, c, *masks.shape[1:]), scores.reshape(k, c)
