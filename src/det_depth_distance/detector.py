"""RF-DETR wrapper returning plain-numpy `Detections`.

Class names come from `predict()`'s `data["class_name"]`, which is correct
for both COCO-pretrained (sparse ids 1..90) and fine-tuned checkpoints; the
`--classes` filter is validated against the loaded model's `class_names`.
`rfdetr` is imported lazily so `resolve_classes` and `Detections` are usable
without the GPU stack.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Class names as strings, resolved with getattr(rfdetr, ...) only when a model is built.
# RFDETRBase is deprecated upstream and intentionally absent.
DET_MODELS: dict[str, str] = {
    "nano": "RFDETRNano",
    "small": "RFDETRSmall",
    "medium": "RFDETRMedium",
    "large": "RFDETRLarge",
}

# Classes relevant to a driving scene; COCO has no generic "traffic sign" class (only "stop sign").
DRIVING_CLASSES: tuple[str, ...] = (
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "bus",
    "truck",
    "traffic light",
    "stop sign",
)


@dataclass(frozen=True, slots=True)
class Detections:
    """Detections of one frame; boxes in original frame pixels."""

    xyxy: np.ndarray  # (N, 4) float32
    confidence: np.ndarray  # (N,) float32
    class_id: np.ndarray  # (N,) int64
    class_name: tuple[str, ...]

    def __len__(self) -> int:
        return len(self.class_name)

    def select(self, keep: np.ndarray) -> "Detections":
        """Subset by a boolean mask of length N."""
        return Detections(
            self.xyxy[keep],
            self.confidence[keep],
            self.class_id[keep],
            tuple(name for name, k in zip(self.class_name, keep, strict=True) if k),
        )


def resolve_classes(spec: str, known: Sequence[str]) -> frozenset[str] | None:
    """Class filter from --classes: 'all' (None = keep everything), 'driving', or a
    comma-separated list of names.

    Raises:
        ValueError: a requested name is not a class of the loaded model.
    """
    if spec == "all":
        return None
    if spec == "driving":
        wanted = set(DRIVING_CLASSES)
    else:
        wanted = {name.strip() for name in spec.split(",") if name.strip()}
    unknown = wanted - set(known)
    if unknown:
        raise ValueError(
            f"classes not known to the model: {sorted(unknown)} (use --classes all)"
        )
    return frozenset(wanted)


class Detector:
    """Load an RF-DETR variant (or a checkpoint via rfdetr.from_checkpoint) and
    detect objects in RGB uint8 frames.

    `optimize=True` calls `RFDETR.inference(dtype=torch.float16)` (TorchScript
    trace + FP16). Boxes are in original frame pixels.
    """

    def __init__(
        self,
        variant: str = "large",
        *,
        weights: str | None = None,
        device: str = "cuda:0",
        threshold: float = 0.5,
        classes: str = "driving",
        optimize: bool = False,
    ) -> None:
        import rfdetr

        if weights:
            self._model = rfdetr.from_checkpoint(weights, device=device)
            self.name = f"rfdetr-{Path(weights).stem}"
        else:
            if variant not in DET_MODELS:
                raise ValueError(
                    f"unknown RF-DETR variant {variant!r}; use one of {list(DET_MODELS)}"
                )
            self._model = getattr(rfdetr, DET_MODELS[variant])(device=device)
            self.name = f"rfdetr-{variant}"
        if optimize:
            import torch

            self._model.inference(dtype=torch.float16)
        self.threshold = threshold
        self.resolution: int = self._model.model_config.resolution
        self.class_names: list[str] = self._model.class_names
        self._keep = resolve_classes(classes, self.class_names)

    def __call__(self, rgb: np.ndarray) -> Detections:
        det = self._model.predict(
            rgb, threshold=self.threshold, include_source_image=False
        )
        names = tuple(str(name) for name in det.data["class_name"])
        found = Detections(
            np.asarray(det.xyxy, dtype=np.float32).reshape(-1, 4),
            np.asarray(det.confidence, dtype=np.float32),
            np.asarray(det.class_id, dtype=np.int64),
            names,
        )
        if self._keep is None:
            return found
        return found.select(
            np.array([name in self._keep for name in names], dtype=bool)
        )
