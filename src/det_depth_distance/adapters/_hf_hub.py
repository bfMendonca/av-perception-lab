"""Offline loading of Hugging Face checkpoints (transformers adapters).

Weights are never downloaded at run time: `from_pretrained` runs with
local_files_only=True against `hf_cache` (or the default HF cache), and a
missing checkpoint becomes a ConfigError carrying the exact download command,
filtered to the one weights file the adapter loads (an unfiltered
snapshot_download also fetches duplicate .pt/.bin files).
"""

from pathlib import Path
from typing import Any

from det_depth_distance.core.options import ConfigError


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
    """`cls.from_pretrained(repo, ...)` from the local cache only.

    Raises:
        ConfigError: the checkpoint is not in the cache (message prefixed with
            "model:" and ending with the download command).
    """
    try:
        return cls.from_pretrained(
            repo,
            cache_dir=None if hf_cache is None else str(hf_cache),
            local_files_only=True,
            **kwargs,
        )
    except OSError:
        where = hf_cache or "the default Hugging Face cache"
        raise ConfigError(
            f"model: {repo} is not in {where} (nothing is downloaded at run time); "
            f"download it with:\n  {download_command(repo, hf_cache, weights_file)}"
        ) from None
