"""transformers SAM-family refiners: SAM, SAM-HQ, SAM 2.1, EdgeTAM, SAM 3.

HFSamRefiner runs any transformers model with the SAM prompt API
(get_image_embeddings + forward(input_boxes, image_embeddings,
multimask_output) + processor.post_process_masks). The processor is
called once per frame with the image and all target boxes (the SAM 1 /
SAM-HQ processors cannot normalise boxes without the image); batches
slice the normalised boxes (each box is scaled on its own, so slicing
after normalising equals normalising per batch). Family differences are
class attributes: SAM 1 and SAM-HQ resize the longest side to 1024 and
pad, so their post-processing also needs `reshaped_input_sizes`; SAM 2.x,
EdgeTAM and SAM 3 resize to a square (1024, or 1008 for SAM 3) and also
predict an object score; SAM-HQ returns its embedding as (image,
intermediate) embeddings. The facebook/sam2.1-* and EdgeTAM checkpoints
are video configs that load into the image models. The SAM 1 / SAM-HQ IoU
head has no sigmoid, so their scores may leave [0, 1].

Weights are read with local_files_only=True from `hf_cache`: nothing is
downloaded at run time. Models run in float32 (quality first). EdgeTAM
needs `timm` (its backbone is a TimmWrapper). facebook/sam3 is gated
("SAM License"): request access and set HF_TOKEN before downloading it.
torch and transformers are imported inside the hooks only.
"""

from abc import abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

import numpy as np

from det_depth_distance.adapters._hf_hub import from_pretrained_offline
from det_depth_distance.adapters.refiners.box_prompt import (
    BoxPromptOptions,
    BoxPromptRefiner,
    DecodedMasks,
)
from det_depth_distance.core.components import LoadContext
from det_depth_distance.core.registry import register
from det_depth_distance.runtime import package_version


@dataclass(frozen=True, slots=True, kw_only=True)
class HFSamOptions(BoxPromptOptions):
    """Options of every transformers SAM-family refiner; each family's Options
    gives `model` its default (no __post_init__ of its own: the base one runs)."""

    model: str  # Hugging Face repo in transformers format
    hf_cache: Path | None = None  # e.g. weights/hf; None = default HF cache


@dataclass(frozen=True, slots=True)
class _HFFrame:
    """Per-frame state of `_prepare`."""

    # Tensor (SAM), (image, intermediate) tuple (SAM-HQ) or FPN list (SAM 2.x).
    embedding: Any
    boxes: Any  # torch (1, K, 4), normalised by the processor
    # (original_sizes,) or (original_sizes, reshaped_input_sizes) (SAM 1 convention).
    sizes: tuple[Any, ...]


class HFSamRefiner(BoxPromptRefiner):
    """A transformers model with the SAM prompt API; abstract, not registered.
    Families set `model_class` and `processor_class` as plain class attributes
    (which makes them concrete), the flags below and, if needed, the two hooks."""

    @property
    @abstractmethod
    def model_class(self) -> str:
        """transformers model class name, e.g. "Sam2Model"."""

    @property
    @abstractmethod
    def processor_class(self) -> str:
        """transformers processor class name, e.g. "Sam2Processor"."""

    # SAM 1 convention: post_process_masks also takes reshaped_input_sizes.
    uses_reshaped_input_sizes: ClassVar[bool] = False
    options: HFSamOptions

    def __init__(self, options: HFSamOptions, *, name: str | None = None) -> None:
        super().__init__(options, name=name)
        self._model = None
        self._processor = None
        self._device = ""

    def default_name(self) -> str:
        """Lower-case basename of the repo, e.g. "sam2.1-hiera-large"."""
        return self.options.model.rstrip("/").split("/")[-1].lower()

    def _load(self, ctx: LoadContext) -> None:
        import torch
        import transformers

        options = self.options
        load = (options.model, options.hf_cache, "model.safetensors")
        self._processor = from_pretrained_offline(
            getattr(transformers, self.processor_class), *load
        )
        model = from_pretrained_offline(
            getattr(transformers, self.model_class), *load, dtype=torch.float32
        )
        self._model = model.to(ctx.device).eval()
        self._device = ctx.device

    def _prepare(self, rgb: np.ndarray, xyxy: np.ndarray) -> _HFFrame:
        import torch

        with torch.inference_mode():
            inputs = self._processor(
                images=rgb, input_boxes=[xyxy.tolist()], return_tensors="pt"
            )
            embedding = self._model.get_image_embeddings(
                inputs["pixel_values"].to(self._device)
            )
        sizes = (inputs["original_sizes"],)
        if self.uses_reshaped_input_sizes:
            sizes += (inputs["reshaped_input_sizes"],)
        return _HFFrame(embedding, inputs["input_boxes"], sizes)

    def _decode(self, state: _HFFrame, batch: slice) -> DecodedMasks:
        import torch

        options = self.options
        with torch.inference_mode():
            out = self._model(
                **self._decoder_inputs(state.embedding),
                input_boxes=state.boxes[:, batch].to(self._device, torch.float32),
                multimask_output=options.multimask,
                **self._forward_kwargs(),
            )
            # Upsampled on the device; only the boolean masks come back.
            masks = self._processor.post_process_masks(
                out.pred_masks,
                *state.sizes,
                mask_threshold=options.mask_threshold,
                binarize=True,
            )[0]
        logit = None
        if self.has_object_score:
            logit = out.object_score_logits[0, :, 0].float().cpu().numpy()
        return DecodedMasks(
            masks.cpu().numpy(), out.iou_scores[0].float().cpu().numpy(), logit
        )

    def _decoder_inputs(self, embedding: Any) -> dict[str, Any]:
        """forward() keyword arguments carrying the frame embedding."""
        return {"image_embeddings": embedding}

    def _forward_kwargs(self) -> dict[str, Any]:
        """Family-specific forward() options."""
        return {}

    def _close(self) -> None:
        import torch

        self._model = self._processor = None
        torch.cuda.empty_cache()

    def _describe(self) -> dict[str, Any]:
        size = None if self._processor is None else self._processor.image_processor.size
        return {
            "model_class": self.model_class,
            "input_size": None if size is None else dict(size),
            "version": package_version("transformers"),
        }


@register("refiner", "sam2")
class Sam2Refiner(HFSamRefiner):
    """SAM 2.1 (Apache-2.0); the refiner of the example configs."""

    model_class = "Sam2Model"
    processor_class = "Sam2Processor"
    has_object_score = True

    @dataclass(frozen=True, slots=True, kw_only=True)
    class Options(HFSamOptions):
        model: str = "facebook/sam2.1-hiera-large"


@register("refiner", "edgetam")
class EdgeTamRefiner(HFSamRefiner):
    """EdgeTAM (Apache-2.0), a light SAM 2 variant; needs timm."""

    model_class = "EdgeTamModel"
    processor_class = "Sam2Processor"
    has_object_score = True

    @dataclass(frozen=True, slots=True, kw_only=True)
    class Options(HFSamOptions):
        model: str = "yonigozlan/EdgeTAM-hf"


@register("refiner", "sam3_tracker")
class Sam3TrackerRefiner(HFSamRefiner):
    """The tracker (box -> mask) part of SAM 3; gated, "SAM License"."""

    model_class = "Sam3TrackerModel"
    processor_class = "Sam3TrackerProcessor"
    has_object_score = True

    @dataclass(frozen=True, slots=True, kw_only=True)
    class Options(HFSamOptions):
        model: str = "facebook/sam3"


@register("refiner", "sam")
class SamRefiner(HFSamRefiner):
    """Segment Anything (SAM 1, Apache-2.0)."""

    model_class = "SamModel"
    processor_class = "SamProcessor"
    uses_reshaped_input_sizes = True

    @dataclass(frozen=True, slots=True, kw_only=True)
    class Options(HFSamOptions):
        model: str = "facebook/sam-vit-huge"


@register("refiner", "sam_hq")
class SamHQRefiner(HFSamRefiner):
    """SAM-HQ (Apache-2.0): SAM 1 plus a high-quality output token."""

    model_class = "SamHQModel"
    processor_class = "SamHQProcessor"
    uses_reshaped_input_sizes = True

    @dataclass(frozen=True, slots=True, kw_only=True)
    class Options(HFSamOptions):
        model: str = "syscv-community/sam-hq-vit-huge"
        # False (the forward default): the SAM mask plus the HQ mask; True: HQ only.
        hq_token_only: bool = False

    def _decoder_inputs(self, embedding: Any) -> dict[str, Any]:
        # get_image_embeddings returns a tuple and forward() needs both parts.
        image, intermediate = embedding
        return {"image_embeddings": image, "intermediate_embeddings": intermediate}

    def _forward_kwargs(self) -> dict[str, Any]:
        return {"hq_token_only": self.options.hq_token_only}
