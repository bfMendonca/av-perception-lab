"""SegFormer semantic segmentation adapter (transformers).

Default checkpoint: Cityscapes, 19 classes (road, sidewalk, ..., vegetation,
terrain, sky, ...), a street-scene domain. The frame is resized to `size`
(height, width; a square default stretches 16:9 frames) and the logits are
bilinearly resized back to the frame before argmax
(post_process_semantic_segmentation). Weights are read with
local_files_only=True from `hf_cache` (default `<weights root>/hf`; the NVIDIA
checkpoints ship pytorch_model.bin), declared by `requirements` for `--prepare`;
torch and transformers are imported inside the hooks only.
"""

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from av_perception_lab.adapters._hf_hub import (
    from_pretrained_offline,
    hf_asset,
    is_repo_id,
    resolve_cache,
)
from av_perception_lab.adapters._weights import INFER_EXTRA
from av_perception_lab.core.components import (
    ClassCatalog,
    LoadContext,
    SemanticSegmenter,
)
from av_perception_lab.core.registry import register
from av_perception_lab.core.requirements import Requirements
from av_perception_lab.core.types import ContractError, SemanticMap
from av_perception_lab.runtime import package_version


@register("semantic", "segformer")
class SegformerSegmenter(SemanticSegmenter):
    """SegformerForSemanticSegmentation on the full frame."""

    @dataclass(frozen=True, slots=True)
    class Options:
        model: str = "nvidia/segformer-b5-finetuned-cityscapes-1024-1024"
        hf_cache: Path | None = None  # None = <weights root>/hf
        size: tuple[int, int] = (1024, 1024)  # (height, width) input, multiples of 32

        def __post_init__(self) -> None:
            if any(s < 32 or s % 32 for s in self.size):
                raise ValueError(f"size must be multiples of 32, got {self.size}")

    def __init__(self, options: Options, *, name: str | None = None) -> None:
        super().__init__(options, name=name)
        self._model = None
        self._processor = None
        self._device = ""
        self._class_names: tuple[str, ...] = ()

    @classmethod
    def requirements(cls, options: Options) -> Requirements:
        """`pytorch_model.bin` + `*.json` of `model` in the HF cache layout."""
        asset = hf_asset(
            "weights", options.model, options.hf_cache, "pytorch_model.bin"
        )
        return Requirements(
            assets=() if asset is None else (asset,),
            packages=("torch", "transformers"),
            extra=INFER_EXTRA,
        )

    @classmethod
    def declared_classes(cls, options: Options) -> ClassCatalog | None:
        """id2label of the checkpoint's `config.json` (plain JSON, no
        transformers); None if the file is missing or not a valid id2label."""
        try:
            config = _config_path(options)
            if config is None:
                return None
            id2label = json.loads(config.read_text(encoding="utf-8"))["id2label"]
            return ClassCatalog(_id2label_names(id2label), "metadata")
        except (
            OSError,
            ValueError,
            KeyError,
            TypeError,
            AttributeError,
            ContractError,
        ):
            return None

    def default_name(self) -> str:
        """Lower-case basename of the repo (programmatic use: configs require a name)."""
        return self.options.model.rstrip("/").split("/")[-1].lower()

    def _load(self, ctx: LoadContext) -> None:
        import torch
        from transformers import (
            SegformerForSemanticSegmentation,
            SegformerImageProcessor,
        )

        load = (self.options.model, self.options.hf_cache, "pytorch_model.bin")
        self._processor = from_pretrained_offline(SegformerImageProcessor, *load)
        model = from_pretrained_offline(
            SegformerForSemanticSegmentation, *load, dtype=torch.float32
        )
        self._model = model.to(ctx.device).eval()
        self._device = ctx.device
        self._class_names = _id2label_names(model.config.id2label)

    @property
    def class_names(self) -> tuple[str, ...]:
        return self._class_names

    def _segment(self, rgb: np.ndarray) -> SemanticMap:
        import torch

        h, w = rgb.shape[:2]
        height, width = self.options.size
        with torch.inference_mode():
            inputs = self._processor(
                images=rgb, size={"height": height, "width": width}, return_tensors="pt"
            )
            out = self._model(pixel_values=inputs["pixel_values"].to(self._device))
            labels = self._processor.post_process_semantic_segmentation(
                out, target_sizes=[(h, w)]
            )[0]
        return SemanticMap(labels.cpu().numpy().astype(np.int32), self._class_names)

    def _close(self) -> None:
        import torch

        self._model = self._processor = None
        torch.cuda.empty_cache()

    def _describe(self) -> dict[str, Any]:
        return {
            "n_classes": len(self._class_names),
            "version": package_version("transformers"),
        }


def _id2label_names(id2label: Mapping[int | str, str]) -> tuple[str, ...]:
    """Class names in id order from a model config's id2label (int or numeric
    string keys, as in config.json). Raises ContractError unless the ids are
    exactly 0..C-1, because SemanticMap labels index this tuple."""
    by_id = {int(k): v for k, v in id2label.items()}
    if sorted(by_id) != list(range(len(by_id))):
        raise ContractError(f"id2label ids are not 0..C-1: {sorted(by_id)}")
    return tuple(by_id[i] for i in range(len(by_id)))


def _config_path(options: Any) -> Path | None:
    """`config.json` of the repo's snapshot in the HF cache layout, or of a
    local checkpoint directory."""
    if not is_repo_id(options.model):
        return Path(options.model).expanduser() / "config.json"
    repo_dir = resolve_cache(options.hf_cache) / (
        "models--" + options.model.replace("/", "--")
    )
    snapshots = repo_dir / "snapshots"
    if not snapshots.is_dir():
        return None
    for snap in sorted(p for p in snapshots.iterdir() if p.is_dir()):
        if (snap / "config.json").is_file():
            return snap / "config.json"
    return None
