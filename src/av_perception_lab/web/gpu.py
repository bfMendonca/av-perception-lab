"""Free GPU memory probe.

Runs in a short subprocess so the server process never imports torch (and
never holds a CUDA context). `nvidia-smi` is unreliable on the inference
host (driver mismatch), hence `torch.cuda.mem_get_info()`.
"""

import subprocess
import sys

_PROBE = (
    "import torch\nfree, _total = torch.cuda.mem_get_info()\nprint(free // 2**20)\n"
)
TIMEOUT_S = 60.0


def free_memory_mib() -> int | None:
    """Free device memory in MiB; None when torch/CUDA is unavailable."""
    try:
        done = subprocess.run(
            [sys.executable, "-c", _PROBE],
            capture_output=True,
            text=True,
            timeout=TIMEOUT_S,
            check=False,
        )
        if done.returncode != 0:
            return None
        return int(done.stdout.strip().splitlines()[-1])
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return None
