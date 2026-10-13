"""RF-DETR detector adapter.

Takes RGB uint8 frames directly; boxes come back in original frame pixels,
already clamped to the image. Class names come from predict()'s
data["class_name"] (correct for COCO and fine-tuned checkpoints).
`optimize` calls RFDETR.inference(dtype=torch.float16) (TorchScript + FP16).
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

from det_depth_distance.core.components import Detector, DetectorOptions, LoadContext
from det_depth_distance.core.registry import register
from det_depth_distance.core.types import Instances
from det_depth_distance.runtime import package_version

# Class names as strings, resolved with getattr(rfdetr, ...) only when a model is built.
# RFDETRBase is deprecated upstream and intentionally absent.
DET_MODELS: dict[str, str] = {
    "nano": "RFDETRNano",
    "small": "RFDETRSmall",
    "medium": "RFDETRMedium",
    "large": "RFDETRLarge",
}


@register("detector", "rfdetr")
class RFDETRDetector(Detector):
    """A COCO-pretrained RF-DETR variant, or a checkpoint via rfdetr.from_checkpoint."""

    @dataclass(frozen=True, slots=True, kw_only=True)
    class Options(DetectorOptions):
        variant: Literal["nano", "small", "medium", "large"] = "large"
        weights: Path | None = None  # fine-tuned checkpoint (.pth); overrides variant
        optimize: bool = False  # TorchScript + FP16

    def __init__(self, options: Options, *, name: str | None = None) -> None:
        super().__init__(options, name=name)
        self._model = None

    def default_name(self) -> str:
        if self.options.weights is not None:
            return f"rfdetr-{self.options.weights.stem}"
        return f"rfdetr-{self.options.variant}"

    def _load(self, ctx: LoadContext) -> None:
        import rfdetr

        opts = self.options
        if opts.weights is not None:
            model = rfdetr.from_checkpoint(str(opts.weights), device=ctx.device)
        else:
            model = getattr(rfdetr, DET_MODELS[opts.variant])(device=ctx.device)
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
        )

    def _describe(self) -> dict[str, Any]:
        return {
            "resolution": self._model.model_config.resolution if self._model else None,
            "n_classes": len(self.class_names) if self._model else None,
            "version": package_version("rfdetr"),
        }
