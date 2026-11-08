"""Classes an adapter spec declares, for the editor (task_0035).

Never loads a model: the spec is validated like `schema.validate_config` does
(`registry.build` only parses the options), its paths pass the server-side
checks of a run, and only then `declared_classes(options)` is asked.
"""

from pathlib import Path
from typing import Any

from av_perception_lab.adapters._hf_hub import is_repo_id
from av_perception_lab.core import registry
from av_perception_lab.core.options import ConfigError
from av_perception_lab.core.requirements import weights_root
from av_perception_lab.web import schema, security
from av_perception_lab.web.paths import WebPaths

LOCAL_MODEL_OUTSIDE_ROOTS = (
    "a local model directory must be inside the weights root or an asset "
    "folder of the server (--asset-root)"
)


def declared(
    kind: str, spec: dict[str, Any], paths: WebPaths
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    """`(errors, {"names", "source"})`; names are None when not knowable."""
    try:
        component = registry.build(kind, spec, path="spec")
    except ConfigError as exc:
        return [schema.split_error(exc)], {}
    found: list[tuple[str, Path]] = []
    schema._walk_paths(component.options, "spec", found)
    errors = security.found_path_problems(found, paths)
    model = getattr(component.options, "model", None)
    if isinstance(model, str) and not is_repo_id(model):
        # A local `model` string is a directory read by the adapter (its
        # config.json): confined, so the endpoint cannot probe the filesystem.
        roots = [weights_root().resolve(), *(r.resolve() for r in paths.asset_roots)]
        real = security.resolve(model)
        if security.within(real, security.client_writable_roots(paths)) or (
            not security.within(real, roots)
        ):
            errors.append(
                {
                    "path": "spec.model",
                    "message": f"{model}: {LOCAL_MODEL_OUTSIDE_ROOTS}",
                }
            )
    if errors:
        return errors, {}
    catalog = type(component).declared_classes(component.options)
    if catalog is None:
        return [], {"names": None, "source": None}
    return [], {"names": list(catalog.names), "source": catalog.source}
