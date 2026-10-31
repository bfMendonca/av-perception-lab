"""RF-DETR detector adapter.

Takes RGB uint8 frames directly; boxes come back in original frame pixels,
already clamped to the image. Class names come from predict()'s
data["class_name"] (correct for COCO and fine-tuned checkpoints).
`optimize` calls RFDETR.inference(dtype=torch.float16) (TorchScript + FP16).

Seg variants (RFDETRSeg*, COCO) return Detections.mask: mask logits at
resolution/4, bilinearly resized to the frame and thresholded at 0; masks are
not cropped to the box (the distance stage does that).

Checkpoints: the COCO checkpoint of each variant is declared as a `UrlAsset`
(`RFDETR_CHECKPOINTS`: the library's own URL table, sizes read with HTTP HEAD
in task_0021) at `<weights root>/rfdetr/<library file name>` and loaded from
that explicit path (`pretrain_weights=`); the library's `~/.roboflow/models`
cache is neither read nor written by default. The file is checked before the
import, because the constructor would download a missing default checkpoint.
A fine-tuned `weights` file goes through `rfdetr.from_checkpoint` as before.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

from det_depth_distance.adapters._weights import INFER_EXTRA, not_prepared, under_root
from det_depth_distance.core.components import Detector, DetectorOptions, LoadContext
from det_depth_distance.core.options import ConfigError
from det_depth_distance.core.registry import register
from det_depth_distance.core.requirements import (
    Requirements,
    UrlAsset,
    UserProvided,
    weights_root,
)
from det_depth_distance.core.types import Instances
from det_depth_distance.runtime import package_version

# Class names as strings, resolved with getattr(rfdetr, ...) only when a model is built.
# RFDETRBase and RFDETRSegPreview are deprecated upstream and intentionally absent.
DET_MODELS: dict[str, str] = {
    "nano": "RFDETRNano",
    "small": "RFDETRSmall",
    "medium": "RFDETRMedium",
    "large": "RFDETRLarge",
    "seg-nano": "RFDETRSegNano",
    "seg-small": "RFDETRSegSmall",
    "seg-medium": "RFDETRSegMedium",
    "seg-large": "RFDETRSegLarge",
    "seg-xlarge": "RFDETRSegXLarge",
    "seg-2xlarge": "RFDETRSeg2XLarge",
}

_URL = "https://repo.roboflow.com/rfdetr"
RFDETR_DIR = "rfdetr"  # <weights root>/rfdetr/<file>
RFDETR_LICENSE = "Apache-2.0"  # every variant above (rfdetr README tables)

# variant -> (local file name = rfdetr.assets.ModelWeights filename, URL, bytes)
RFDETR_CHECKPOINTS: dict[str, tuple[str, str, int]] = {
    "nano": (
        "rf-detr-nano.pth",
        f"{_URL}/nano_coco/checkpoint_best_regular.pth",
        366_287_238,
    ),
    "small": (
        "rf-detr-small.pth",
        f"{_URL}/small_coco/checkpoint_best_regular.pth",
        386_045_550,
    ),
    "medium": (
        "rf-detr-medium.pth",
        f"{_URL}/medium_coco/checkpoint_best_regular.pth",
        404_992_918,
    ),
    "large": ("rf-detr-large-2026.pth", f"{_URL}/rf-detr-large-2026.pth", 135_954_129),
    "seg-nano": ("rf-detr-seg-nano.pt", f"{_URL}/rf-detr-seg-n-ft.pth", 134_545_398),
    "seg-small": ("rf-detr-seg-small.pt", f"{_URL}/rf-detr-seg-s-ft.pth", 135_042_342),
    "seg-medium": (
        "rf-detr-seg-medium.pt",
        f"{_URL}/rf-detr-seg-m-ft.pth",
        143_024_058,
    ),
    "seg-large": ("rf-detr-seg-large.pt", f"{_URL}/rf-detr-seg-l-ft.pth", 145_055_866),
    "seg-xlarge": (
        "rf-detr-seg-xlarge.pt",
        f"{_URL}/rf-detr-seg-xl-ft.pth",
        152_713_150,
    ),
    "seg-2xlarge": (
        "rf-detr-seg-xxlarge.pt",
        f"{_URL}/rf-detr-seg-2xl-ft.pth",
        154_851_262,
    ),
}


def checkpoint_path(variant: str) -> Path:
    """Default location of a variant's COCO checkpoint under the weights root."""
    return weights_root() / RFDETR_DIR / RFDETR_CHECKPOINTS[variant][0]


@register("detector", "rfdetr")
class RFDETRDetector(Detector):
    """A COCO-pretrained RF-DETR variant (detection or seg-*), or a checkpoint via
    rfdetr.from_checkpoint."""

    @dataclass(frozen=True, slots=True, kw_only=True)
    class Options(DetectorOptions):
        variant: Literal[
            "nano",
            "small",
            "medium",
            "large",
            "seg-nano",
            "seg-small",
            "seg-medium",
            "seg-large",
            "seg-xlarge",
            "seg-2xlarge",
        ] = "large"
        weights: Path | None = None  # fine-tuned checkpoint (.pth); overrides variant
        optimize: bool = False  # TorchScript + FP16

    def __init__(self, options: Options, *, name: str | None = None) -> None:
        super().__init__(options, name=name)
        self._model = None

    @classmethod
    def requirements(cls, options: Options) -> Requirements:
        """The variant's COCO checkpoint (URL), or a fine-tuned `weights` file
        inside the weights root (user-provided; outside it: not declared)."""
        packages = ("torch", "rfdetr")
        if options.weights is not None:
            target = under_root(options.weights)
            assets = (
                ()
                if target is None
                else (
                    UserProvided(
                        key="weights",
                        target=target,
                        option="weights",
                        hint=f"provide the fine-tuned checkpoint at {options.weights}",
                    ),
                )
            )
            return Requirements(assets=assets, packages=packages, extra=INFER_EXTRA)
        name, url, size = RFDETR_CHECKPOINTS[options.variant]
        asset = UrlAsset(
            key="weights",
            target=f"{RFDETR_DIR}/{name}",
            url=url,
            size_bytes=size,
            license=RFDETR_LICENSE,
        )
        return Requirements(assets=(asset,), packages=packages, extra=INFER_EXTRA)

    def default_name(self) -> str:
        if self.options.weights is not None:
            return f"rfdetr-{self.options.weights.stem}"
        return f"rfdetr-{self.options.variant}"

    def _load(self, ctx: LoadContext) -> None:
        opts = self.options
        if opts.weights is not None:
            if not opts.weights.is_file():
                raise ConfigError(f"weights: file not found: {opts.weights}")
            import rfdetr

            model = rfdetr.from_checkpoint(str(opts.weights), device=ctx.device)
        else:
            # Checked first: the constructor would download a missing file.
            path = checkpoint_path(opts.variant)
            if not path.is_file():
                raise not_prepared(
                    "variant", f"RF-DETR {opts.variant} checkpoint", path
                )
            import rfdetr

            model = getattr(rfdetr, DET_MODELS[opts.variant])(
                pretrain_weights=str(path), device=ctx.device
            )
        if opts.optimize:
            import torch

            model.inference(dtype=torch.float16)
        self._model = model

    def _close(self) -> None:
        import torch

        self._model = None
        torch.cuda.empty_cache()

    @property
    def class_names(self) -> list[str]:
        return self._model.class_names

    def _predict(self, rgb: np.ndarray) -> Instances:
        det = self._model.predict(
            rgb, threshold=self.options.threshold, include_source_image=False
        )
        return self._instances(
            det.xyxy,
            det.confidence,
            det.class_id,
            [str(name) for name in det.data["class_name"]],
            # Seg models only; already (N, H, W) at frame resolution.
            None if det.mask is None else np.asarray(det.mask, dtype=bool),
        )

    def _describe(self) -> dict[str, Any]:
        config = self._model.model_config if self._model else None
        segmentation = config.segmentation_head if config else None
        return {
            "resolution": config.resolution if config else None,
            "n_classes": len(self.class_names) if self._model else None,
            "segmentation": segmentation,
            # native side of the square mask logits, before resizing to the frame
            "mask_resolution": config.resolution // config.mask_downsample_ratio
            if segmentation
            else None,
            "version": package_version("rfdetr"),
        }
