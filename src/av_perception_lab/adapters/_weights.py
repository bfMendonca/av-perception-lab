"""Default weight locations under the weights root and YAML overrides.

Adapters compute their default locations from `core.requirements.weights_root()`
(`--weights-root`, `AVPL_WEIGHTS_ROOT`, else the repo `weights/`); an explicit
path in the YAML overrides the default. An explicit path that resolves
*inside* the weights root is declared like a default (status and `--prepare`
see it); one *outside* is the user's responsibility: it is not declared (asset
targets never leave the root) and `load()` fails with a `ConfigError` if it is
missing. Pure stdlib: nothing here imports a framework.
"""

from pathlib import Path

from av_perception_lab.core.options import ConfigError
from av_perception_lab.core.requirements import weights_root

INFER_EXTRA = "infer"  # the pyproject extra every current adapter needs


def under_root(path: Path) -> str | None:
    """`path` (resolved, relative paths from the working directory) as a
    POSIX target relative to the weights root, or None when it is outside
    the root (or is the root itself)."""
    root = weights_root().resolve()
    resolved = Path(path).expanduser().resolve()
    if resolved == root or not resolved.is_relative_to(root):
        return None
    return resolved.relative_to(root).as_posix()


def not_prepared(option: str, what: str, path: Path) -> ConfigError:
    """The error of a missing default (or in-root) file: names `--prepare`."""
    return ConfigError(
        f"{option}: {what} not found at {path}; nothing is downloaded at run "
        "time: run `av-perception-lab --config <yaml> --prepare` "
        "(or set the option to an existing file)"
    )
