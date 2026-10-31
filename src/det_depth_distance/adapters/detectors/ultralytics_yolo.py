"""Ultralytics YOLO detector adapter (YOLO26 COCO and Objects365 weights,
detection and segmentation).

predict() treats numpy input as BGR (OpenCV order), so RGB frames are
flipped here. Boxes are letterbox-corrected and clipped to the frame by
ultralytics. Class names are the checkpoint's own `model.names`.

Segmentation weights (*-seg.pt, task "segment") also fill `masks`:
predict() is called with retina_masks=True so masks come back at frame
resolution with the letterbox removed (without it, masks stay at the
letterboxed input size). Ultralytics crops masks to their boxes and drops
predictions with an empty mask.

`weights` follows adapters._ultralytics: a bare file name resolves under
`<weights root>/ultralytics/` (a path with a directory part keeps its usual
meaning); release names are declared as
URL assets for `--prepare`, other files as user-provided.

YOLO_OFFLINE, YOLO_AUTOINSTALL and YOLO_CONFIG_DIR are set (setdefault, see
adapters._ultralytics_env) when this module is imported, before the lazy
import, because ultralytics reads them at import time.
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from det_depth_distance.adapters._ultralytics import (
    check_weights,
    resolve_weights,
    weights_asset,
)
from det_depth_distance.adapters._ultralytics_env import apply_ultralytics_env
from det_depth_distance.adapters._weights import INFER_EXTRA, not_prepared, under_root
from det_depth_distance.core.components import Detector, DetectorOptions, LoadContext
from det_depth_distance.core.options import ConfigError
from det_depth_distance.core.registry import register
from det_depth_distance.core.requirements import Requirements
from det_depth_distance.core.types import Instances
from det_depth_distance.runtime import package_version

apply_ultralytics_env()

_STRIDE = 32  # max stride of the YOLO26 detection heads


@register("detector", "yolo")
class YoloDetector(Detector):
    """An Ultralytics YOLO detection or segmentation checkpoint (.pt) run with PyTorch."""

    @dataclass(frozen=True, slots=True, kw_only=True)
    class Options(DetectorOptions):
        # Existence is checked in _load (before the import), not here, so example
        # configs can be validated on machines without the weights. A bare file
        # name = under <weights root>/ultralytics/ (adapters._ultralytics).
        weights: Path
        imgsz: int = 640  # inference size, multiple of the stride
        half: bool = False  # FP16 inference

        def __post_init__(self) -> None:
            # Explicit base call: zero-arg super() fails in slots dataclasses.
            DetectorOptions.__post_init__(self)
            check_weights(self.weights)
            if self.imgsz <= 0 or self.imgsz % _STRIDE:
                raise ValueError(f"imgsz must be a positive multiple of {_STRIDE}")

    def __init__(self, options: Options, *, name: str | None = None) -> None:
        super().__init__(options, name=name)
        self._model = None
        self._class_names: list[str] = []
        self._device = ""
        self._task: str | None = None  # "detect" or "segment", from the checkpoint

    @classmethod
    def requirements(cls, options: Options) -> Requirements:
        asset = weights_asset(options.weights, "provide the YOLO checkpoint (.pt)")
        return Requirements(
            assets=() if asset is None else (asset,),
            packages=("torch", "ultralytics"),
            extra=INFER_EXTRA,
        )

    def default_name(self) -> str:
        return self.options.weights.stem

    def _load(self, ctx: LoadContext) -> None:
        weights = resolve_weights(self.options.weights)
        # Checked first: ultralytics would otherwise try to download a missing file.
        if not weights.is_file():
            if under_root(weights) is not None:
                raise not_prepared("weights", "file", weights)
            raise ConfigError(f"weights: file not found: {weights}")
        # ultralytics only creates its settings dir when the parent already exists.
        Path(os.environ["YOLO_CONFIG_DIR"]).mkdir(parents=True, exist_ok=True)
        from ultralytics import YOLO

        # A .pt checkpoint carries its own task (ultralytics ignores task= for it),
        # so no task is passed and the check below admits detect and segment.
        model = YOLO(str(weights))
        if model.task not in ("detect", "segment"):
            raise ConfigError(
                f"weights: {weights}: task {model.task!r} not supported (detect or segment)"
            )
        self._model = model
        self._task = model.task
        self._class_names = [model.names[i] for i in sorted(model.names)]
        self._device = ctx.device

    def _close(self) -> None:
        import torch

        self._model = None
        torch.cuda.empty_cache()

    @property
    def class_names(self) -> list[str]:
        return self._class_names

    def _predict(self, rgb: np.ndarray) -> Instances:
        # RGB -> BGR here, and only here: predict() assumes OpenCV channel order.
        bgr = np.ascontiguousarray(rgb[..., ::-1])
        result = self._model.predict(
            bgr,
            conf=self.options.threshold,
            imgsz=self.options.imgsz,
            device=self._device,
            # `quantize` replaces the deprecated `half` flag in ultralytics 8.4.
            quantize=16 if self.options.half else None,
            # Frame-resolution masks with the letterbox removed (segment only).
            retina_masks=self._task == "segment",
            verbose=False,
        )[0]
        boxes = result.boxes
        class_id = boxes.cls.cpu().numpy().astype(np.int64)
        masks = None
        if self._task == "segment":
            # ultralytics returns masks=None, not an empty stack, without detections.
            masks = (
                np.zeros((0, *rgb.shape[:2]), dtype=bool)
                if result.masks is None
                else result.masks.data.bool().cpu().numpy()
            )
        return self._instances(
            boxes.xyxy.cpu().numpy(),
            boxes.conf.cpu().numpy(),
            class_id,
            [self._class_names[i] for i in class_id],
            masks,
        )

    def _describe(self) -> dict[str, Any]:
        return {
            "task": self._task,
            "n_classes": len(self._class_names),
            "version": package_version("ultralytics"),
        }
