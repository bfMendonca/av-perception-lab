"""Offline loading and requirement declaration of Hugging Face checkpoints.

Weights are never downloaded at run time: `from_pretrained` runs with
local_files_only=True against `hf_cache`, whose default is the project cache
`<weights root>/hf` (`default_cache`); an explicit `hf_cache` in the YAML
(relative to the working directory) overrides it. A missing checkpoint becomes
a ConfigError naming `--prepare` and the equivalent download command, filtered
to the one weights file the adapter loads (an unfiltered snapshot_download
also fetches duplicate .pt/.bin files).

`hf_asset` declares the same files (weights file + `*.json`) as a
`HuggingFaceAsset` in that cache layout, so an existing `weights/hf` cache
counts as ready unchanged. Sizes and licences of the repos the project ships
(read from the Hugging Face API, task_0021) are in `HF_FACTS`; any other repo
is declared without them and with `known=False` (the web server does not
download it; the CLI does); so is a known repo in a cache other than the
canonical `<weights root>/hf`. Local checkpoint directories are not declared.
A reference is a repo id only when it has the `org/name` shape
(`REPO_ID_RE`), checked before any file-system probe.
"""

import re
from pathlib import Path
from typing import Any

from av_perception_lab.adapters._weights import under_root
from av_perception_lab.core.options import ConfigError
from av_perception_lab.core.requirements import HuggingFaceAsset, weights_root

HF_DIR = "hf"  # <weights root>/hf: the Hugging Face cache layout
# `org/name`: letters, digits, `_`, `.`, `-`; each part starts with a letter or
# digit (so never `.`/`..`), at most 96 characters.
REPO_ID_RE = re.compile(r"[A-Za-z0-9][\w.-]{0,95}/[A-Za-z0-9][\w.-]{0,95}", re.ASCII)

# repo -> (weights file bytes, licence tag, gated); Hugging Face API, 2026-10-31.
HF_FACTS: dict[str, tuple[int, str, bool]] = {
    "depth-anything/DA3-SMALL": (137_248_940, "apache-2.0", False),
    "depth-anything/DA3-BASE": (541_518_028, "apache-2.0", False),
    "depth-anything/DA3-LARGE": (1_643_843_860, "cc-by-nc-4.0", False),
    "depth-anything/DA3-LARGE-1.1": (1_643_843_860, "apache-2.0", False),
    "depth-anything/DA3-GIANT": (5_422_814_644, "cc-by-nc-4.0", False),
    "depth-anything/DA3-GIANT-1.1": (5_422_814_644, "cc-by-nc-4.0", False),
    "depth-anything/DA3MONO-LARGE": (1_336_734_448, "apache-2.0", False),
    "depth-anything/DA3METRIC-LARGE": (1_336_734_448, "apache-2.0", False),
    "depth-anything/DA3NESTED-GIANT-LARGE": (6_759_558_100, "cc-by-nc-4.0", False),
    "depth-anything/DA3NESTED-GIANT-LARGE-1.1": (
        6_759_558_100,
        "cc-by-nc-4.0",
        False,
    ),
    "facebook/sam2.1-hiera-large": (897_897_416, "apache-2.0", False),
    "yonigozlan/EdgeTAM-hf": (55_912_824, "apache-2.0", False),
    "facebook/sam3": (3_439_938_512, "other (SAM License)", True),
    "facebook/sam-vit-huge": (2_564_432_288, "apache-2.0", False),
    "syscv-community/sam-hq-vit-huge": (2_570_816_384, "apache-2.0", False),
    "nvidia/segformer-b5-finetuned-cityscapes-1024-1024": (
        338_896_453,
        "other (NVIDIA)",
        False,
    ),
    "nvidia/segformer-b0-finetuned-cityscapes-640-1280": (
        14_957_601,
        "other (NVIDIA)",
        False,
    ),
}


def default_cache() -> Path:
    """`<weights root>/hf`: where `hf_cache: null` reads and Prepare writes."""
    return weights_root() / HF_DIR


def resolve_cache(hf_cache: Path | None) -> Path:
    """The cache an adapter reads: the explicit `hf_cache` or the default."""
    return default_cache() if hf_cache is None else Path(hf_cache).expanduser()


def is_repo_id(ref: str) -> bool:
    """True for an `org/name` Hugging Face id, False for a local directory.
    The shape is checked first: a string that is not `org/name` is never
    probed on disk (it is a local path); one that is counts as a local
    directory only when it exists."""
    if not REPO_ID_RE.fullmatch(ref) or ".." in ref:
        return False
    return not Path(ref).exists()


def hf_asset(
    key: str, repo: str, hf_cache: Path | None, weights_file: str
) -> HuggingFaceAsset | None:
    """`repo`'s `weights_file` + `*.json` in the cache layout under the weights
    root, or None (local directory, or an explicit cache outside the root)."""
    if not is_repo_id(repo):
        return None
    target = under_root(resolve_cache(hf_cache))
    if target is None:
        return None
    size, licence, gated = HF_FACTS.get(repo, (None, None, False))
    # Only the repos of HF_FACTS are vouched for; any other id came from a
    # user option (prepared by the CLI, refused by the web server).
    return HuggingFaceAsset(
        key=key,
        target=target,
        repo=repo,
        files=(weights_file,),
        patterns=("*.json",),
        size_bytes=size,
        license=licence,
        gated=gated,
        known=repo in HF_FACTS and target == HF_DIR,
    )


def download_command(
    repo: str, hf_cache: Path | None, weights_file: str = "model.safetensors"
) -> str:
    """Shell command (run on the inference host, in the repo) that downloads
    `repo`'s JSON files and `weights_file` into `hf_cache`."""
    cache = "" if hf_cache is None else f"cache_dir={str(hf_cache)!r}, "
    return (
        'uv run --extra infer python -c "from huggingface_hub import snapshot_download; '
        f"snapshot_download({repo!r}, {cache}"
        f"allow_patterns=['*.json', {weights_file!r}])\""
    )


def from_pretrained_offline(
    cls: Any, repo: str, hf_cache: Path | None, weights_file: str, **kwargs: Any
) -> Any:
    """`cls.from_pretrained(repo, ...)` from the local cache only (`hf_cache`,
    default `<weights root>/hf`).

    Raises:
        ConfigError: the checkpoint is not in the cache (message prefixed with
            "model:", naming `--prepare` and ending with the download command).
    """
    cache = resolve_cache(hf_cache)
    try:
        return cls.from_pretrained(
            repo, cache_dir=str(cache), local_files_only=True, **kwargs
        )
    except OSError:
        raise ConfigError(
            f"model: {repo} is not in {cache} (nothing is downloaded at run time); "
            "run `av-perception-lab --config <yaml> --prepare` or download it "
            f"with:\n  {download_command(repo, cache, weights_file)}"
        ) from None
