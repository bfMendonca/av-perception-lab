"""Runtime facts for metrics.json: GPU memory, versions, repository commit.

The only module outside the adapters that imports torch (lazily, inside
DeviceMonitor), so the core stays importable without the GPU stack.
"""

import subprocess
from importlib import metadata
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]


def package_version(dist: str) -> str | None:
    """Installed version of distribution `dist`, None if absent."""
    try:
        return metadata.version(dist)
    except metadata.PackageNotFoundError:
        return None


def git_commit() -> str:
    """Short commit of the repository checkout ("unknown" outside git)."""
    try:
        return subprocess.check_output(
            ["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


class DeviceMonitor:
    """Peak torch allocation and device-wide memory delta over a run.

    Uses torch.cuda.mem_get_info instead of nvidia-smi, which can fail on the
    inference host (driver/library mismatch).
    """

    def __init__(self, device: str) -> None:
        self.device = device
        self._free0 = 0

    def start(self) -> None:
        """Initialise CUDA on the device and take the baseline (before model loads)."""
        import torch

        torch.cuda.set_device(torch.device(self.device))
        torch.cuda.reset_peak_memory_stats(self.device)
        self._free0, _ = torch.cuda.mem_get_info(self.device)

    def summary(self) -> dict[str, Any]:
        """Memory figures and torch/CUDA/GPU identification."""
        import torch

        free1, _ = torch.cuda.mem_get_info(self.device)
        return {
            "peak_vram_torch_allocated_gb": torch.cuda.max_memory_allocated(self.device)
            / 1e9,
            "device_mem_used_delta_gb_approx": (self._free0 - free1) / 1e9,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(self.device),
        }
