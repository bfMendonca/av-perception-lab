"""Framework-agnostic base for box-prompted mask refiners (SAM family).

BoxPromptRefiner implements MaskRefiner._refine once for every model that
turns a box into a mask: the frame is prepared once (one image embedding,
reused by every batch), the target boxes are decoded in batches of
`max_boxes`, and for each box the mask with the highest predicted IoU is
kept (C=1 unless `multimask`). `mask_score` is that IoU, or the object
probability (sigmoid of the object-score logit) with score="object";
`min_object_score` empties masks of boxes the model says hold no object,
so MaskRefiner keeps the previous mask or falls back to the box.

Subclasses implement only `_prepare` (frame -> opaque state) and `_decode`
(state, slice of boxes -> DecodedMasks at frame resolution, thresholded)
and set `has_object_score`. This module imports numpy only and registers
nothing.

Adding a box-prompted model:
  1. subclass BoxPromptRefiner (or HFSamRefiner for a transformers model
     with the SAM API) and set the class attributes that differ;
  2. give it a nested `Options` dataclass deriving from BoxPromptOptions
     (or HFSamOptions) with its defaults; a `__post_init__` of its own calls
     `BoxPromptOptions.__post_init__(self)` first, never a bare super();
  3. decorate it with @register("refiner", "<type>") and list its module
     in core.registry.ADAPTER_MODULES if it is a new module;
  4. add its case to REFINER_CASES (tests/gpu/conftest.py) so it runs the
     shared refiner contract test, plus pure tests for any helper;
  5. add its type to the expected set of
     tests/test_registry.py::test_refiners_and_segmenters_are_registered
     (and, for an HFSamRefiner family, its row to FAMILIES in
     tests/test_hf_sam.py): both pin the registered refiners on purpose.
"""

import math
from abc import abstractmethod
from dataclasses import dataclass
from typing import Any, ClassVar, Literal, final

import numpy as np

from av_perception_lab.core.components import MaskRefiner, RefinerOptions
from av_perception_lab.core.options import ConfigError
from av_perception_lab.core.types import (
    ContractError,
    Instances,
    RefinedMasks,
    check_array,
)


@dataclass(frozen=True, slots=True, kw_only=True)
class BoxPromptOptions(RefinerOptions):
    """Options read by BoxPromptRefiner, shared by every box-prompted refiner."""

    max_boxes: int = 32  # boxes per decoder call (bounds the peak memory), >= 1
    multimask: bool = False  # True: best of C=3 candidate masks by predicted IoU
    mask_threshold: float = 0.0  # logit threshold of the binarisation
    score: Literal["iou", "object"] = "iou"  # what goes to Instances.mask_score
    # Object probability in [0, 1] below which the mask is emptied (0.5 is the
    # logit-0 rule of SAM 2's video path); None = no filter.
    min_object_score: float | None = None

    def __post_init__(self) -> None:
        if self.max_boxes < 1:
            raise ValueError(f"max_boxes must be >= 1, got {self.max_boxes}")
        if not math.isfinite(self.mask_threshold):
            raise ValueError(
                f"mask_threshold must be finite, got {self.mask_threshold}"
            )
        if self.min_object_score is not None and not 0.0 <= self.min_object_score <= 1:
            raise ValueError(
                f"min_object_score must be in [0, 1], got {self.min_object_score}"
            )


@dataclass(frozen=True, slots=True)
class DecodedMasks:
    """Output of `_decode` for a batch of k boxes, C candidate masks each."""

    masks: np.ndarray  # (k, C, H, W) bool at frame resolution, thresholded
    iou: np.ndarray  # (k, C) float32 IoU predicted by the model
    object_logit: np.ndarray | None = None  # (k,) float32; None without the signal

    def validate(self, k: int, frame_w: int, frame_h: int) -> None:
        """Raises ContractError on wrong shapes or dtypes."""
        masks = self.masks
        if not isinstance(masks, np.ndarray) or masks.dtype != np.bool_:
            got = getattr(masks, "dtype", type(masks).__name__)
            raise ContractError(f"masks: expected bool array, got {got}")
        if masks.ndim != 4 or masks.shape[0] != k or masks.shape[1] < 1:
            raise ContractError(f"masks: shape {masks.shape} is not ({k}, C, H, W)")
        if masks.shape[2:] != (frame_h, frame_w):
            raise ContractError(
                f"masks: size {masks.shape[2:]} != frame {(frame_h, frame_w)}"
            )
        check_array(np.float32, (masks.shape[1],))("iou", self.iou, k, frame_w, frame_h)
        if self.object_logit is not None:
            check_array(np.float32)(
                "object_logit", self.object_logit, k, frame_w, frame_h
            )


class BoxPromptRefiner(MaskRefiner):
    """Template method of the second level: owns batching, the single frame
    embedding and the mask choice; a subclass only translates its framework.
    Every subclass returns the same RefinedMasks and passes the same contract
    test."""

    # Whether `_decode` returns object-score logits (SAM 2.x, EdgeTAM, SAM 3).
    has_object_score: ClassVar[bool] = False
    options: BoxPromptOptions

    def __init__(self, options: BoxPromptOptions, *, name: str | None = None) -> None:
        """Raises ConfigError for object-score options on a model without that signal."""
        super().__init__(options, name=name)
        if not self.has_object_score:
            if options.score == "object":
                raise ConfigError(f"score: {self.type_name} has no object score")
            if options.min_object_score is not None:
                raise ConfigError(
                    f"min_object_score: {self.type_name} has no object score"
                )

    @final
    def _refine(self, rgb: np.ndarray, instances: Instances) -> RefinedMasks:
        """Refine the target boxes: one `_prepare`, then `_decode` per batch.

        Batches never change the result (each box is decoded independently from
        the same frame state). Raises ContractError if a hook breaks DecodedMasks.
        """
        h, w = rgb.shape[:2]
        state = self._prepare(rgb, instances.xyxy)
        masks, scores = [], []
        for batch in box_batches(len(instances), self.options.max_boxes):
            decoded = self._decode(state, batch)
            decoded.validate(batch.stop - batch.start, w, h)
            if (decoded.object_logit is not None) != self.has_object_score:
                raise ContractError(
                    f"{self.name}: object_logit must be given iff has_object_score"
                )
            m, s = select_masks(
                decoded,
                score=self.options.score,
                min_object_score=self.options.min_object_score,
            )
            masks.append(m)
            scores.append(s)
        return RefinedMasks(np.concatenate(masks), np.concatenate(scores))

    @abstractmethod
    def _prepare(self, rgb: np.ndarray, xyxy: np.ndarray) -> Any:
        """Opaque per-frame state (image embedding + boxes in the model's format)
        for an (H, W, 3) uint8 RGB frame and its (K, 4) float32 target boxes in
        frame pixels. Called once per frame with targets; nothing is kept
        across frames."""

    @abstractmethod
    def _decode(self, state: Any, batch: slice) -> DecodedMasks:
        """Masks of the boxes `batch` of `state`, at frame resolution and
        binarised with `options.mask_threshold`."""


def box_batches(n: int, max_boxes: int) -> list[slice]:
    """Contiguous slices of at most `max_boxes` covering range(n) ([] for n = 0)."""
    return [slice(i, min(i + max_boxes, n)) for i in range(0, n, max_boxes)]


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return (1.0 / (1.0 + np.exp(-x.astype(np.float64)))).astype(np.float32)


def select_masks(
    decoded: DecodedMasks,
    *,
    score: Literal["iou", "object"],
    min_object_score: float | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Pick one mask per box: argmax of predicted IoU over the C candidates
    (first index on ties). Returns (k, H, W) bool masks and (k,) float32
    scores (the chosen IoU, or the object probability with score="object");
    masks whose object probability is below `min_object_score` are
    returned empty."""
    best = decoded.iou.argmax(axis=1)
    rows = np.arange(len(best))
    masks = decoded.masks[rows, best]
    # The refiner's __init__ guarantees object logits when these options need them.
    logit = decoded.object_logit
    probability = None if logit is None else _sigmoid(logit)
    scores = probability if score == "object" else decoded.iou[rows, best]
    if min_object_score is not None:
        masks[probability < min_object_score] = False
    return masks, scores.astype(np.float32)
