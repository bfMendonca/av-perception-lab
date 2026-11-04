"""transformers SAM-family refiners without transformers: the family table,
defaults, names, hooks and the offline-loading error (no torch)."""

import sys
from pathlib import Path

import pytest

from av_perception_lab.adapters._hf_hub import (
    download_command,
    from_pretrained_offline,
)
from av_perception_lab.adapters.refiners.hf_sam import (
    EdgeTamRefiner,
    HFSamOptions,
    HFSamRefiner,
    Sam2Refiner,
    Sam3TrackerRefiner,
    SamHQRefiner,
    SamRefiner,
)
from av_perception_lab.core import registry
from av_perception_lab.core.options import ConfigError

# type -> (class, model_class, processor_class, reshaped sizes, object score, model)
FAMILIES = {
    "sam2": (
        Sam2Refiner,
        "Sam2Model",
        "Sam2Processor",
        False,
        True,
        "facebook/sam2.1-hiera-large",
    ),
    "edgetam": (
        EdgeTamRefiner,
        "EdgeTamModel",
        "Sam2Processor",
        False,
        True,
        "yonigozlan/EdgeTAM-hf",
    ),
    "sam3_tracker": (
        Sam3TrackerRefiner,
        "Sam3TrackerModel",
        "Sam3TrackerProcessor",
        False,
        True,
        "facebook/sam3",
    ),
    "sam": (
        SamRefiner,
        "SamModel",
        "SamProcessor",
        True,
        False,
        "facebook/sam-vit-huge",
    ),
    "sam_hq": (
        SamHQRefiner,
        "SamHQModel",
        "SamHQProcessor",
        True,
        False,
        "syscv-community/sam-hq-vit-huge",
    ),
}


@pytest.mark.parametrize("type_name", list(FAMILIES))
def test_family_table(type_name):
    cls, model_class, processor_class, reshaped, objectness, model = FAMILIES[type_name]
    assert registry.available("refiner")["refiner"][type_name] is cls
    assert (cls.model_class, cls.processor_class) == (model_class, processor_class)
    assert cls.uses_reshaped_input_sizes is reshaped
    assert cls.has_object_score is objectness
    refiner = registry.build("refiner", {"type": type_name}, path="refiners[0]")
    assert refiner.options.model == model and refiner.options.hf_cache is None
    assert isinstance(refiner.options, HFSamOptions) and not refiner.is_loaded
    assert refiner.options.targets == "missing" and refiner.options.max_boxes == 32


@pytest.mark.parametrize(
    ("spec", "name"),
    [
        ({"type": "sam2"}, "sam2.1-hiera-large"),
        ({"type": "edgetam"}, "edgetam-hf"),
        ({"type": "sam_hq"}, "sam-hq-vit-huge"),
        ({"type": "sam", "model": "facebook/sam-vit-base"}, "sam-vit-base"),
        ({"type": "sam2", "name": "fine"}, "fine"),
    ],
)
def test_default_name_is_the_repo_basename(spec, name):
    assert registry.build("refiner", spec, path="refiners[0]").name == name


def test_sam_hq_hooks():
    refiner = registry.build(
        "refiner", {"type": "sam_hq", "hq_token_only": True}, path="refiners[0]"
    )
    image, intermediate = object(), object()
    assert refiner._decoder_inputs((image, intermediate)) == {
        "image_embeddings": image,
        "intermediate_embeddings": intermediate,
    }
    assert refiner._forward_kwargs() == {"hq_token_only": True}
    sam2 = registry.build("refiner", {"type": "sam2"}, path="refiners[0]")
    assert sam2._decoder_inputs(image) == {"image_embeddings": image}
    assert sam2._forward_kwargs() == {}


def test_hf_sam_refiner_is_abstract_and_unregistered():
    with pytest.raises(TypeError, match="abstract"):
        HFSamRefiner(HFSamOptions(model="x"))
    assert HFSamRefiner not in registry.available("refiner")["refiner"].values()


def test_download_command():
    with_cache = download_command("facebook/sam2.1-hiera-large", Path("weights/hf"))
    assert with_cache == (
        'uv run --extra infer python -c "from huggingface_hub import snapshot_download; '
        "snapshot_download('facebook/sam2.1-hiera-large', cache_dir='weights/hf', "
        "allow_patterns=['*.json', 'model.safetensors'])\""
    )
    default = download_command("nvidia/segformer-b0", None, "pytorch_model.bin")
    assert "cache_dir" not in default
    assert "allow_patterns=['*.json', 'pytorch_model.bin']" in default


def test_missing_checkpoint_is_a_config_error():
    class Missing:
        @classmethod
        def from_pretrained(cls, repo, **kwargs):
            assert kwargs["local_files_only"] is True  # never downloads
            assert kwargs["cache_dir"] == "weights/hf"
            raise OSError("not cached")

    with pytest.raises(
        ConfigError, match=r"^model: facebook/sam3 is not in weights/hf"
    ) as exc:
        from_pretrained_offline(
            Missing, "facebook/sam3", Path("weights/hf"), "model.safetensors"
        )
    assert download_command("facebook/sam3", Path("weights/hf")) in str(exc.value)


def test_no_framework_imported():
    assert not {"torch", "transformers", "timm"} & set(sys.modules)
