"""Classes an adapter spec declares, for the editor (task_0035).

Never loads a model and never constructs the component (its constructor reads
files such as the label map): the spec's options are only parsed into the
options dataclass, their paths pass the server-side checks of a run, and only
then `declared_classes(options)` is asked.
"""

from typing import Any

from av_perception_lab.adapters._hf_hub import is_repo_id
from av_perception_lab.core import registry
from av_perception_lab.core.options import ConfigError, from_mapping
from av_perception_lab.core.requirements import weights_root
from av_perception_lab.web import schema, security
from av_perception_lab.web.paths import WebPaths

LOCAL_MODEL_OUTSIDE_ROOTS = (
    "a local model directory must be inside the weights root or an asset "
    "folder of the server (--asset-root)"
)


def _parse(kind: str, spec: dict[str, Any]) -> tuple[type, Any]:
    """Class and parsed options of a spec; nothing is constructed."""
    family = registry.available(kind).get(kind, {})
    fields = {k: v for k, v in spec.items() if k not in ("type", "name")}
    type_name = spec.get("type")
    cls = family.get(type_name) if isinstance(type_name, str) else None
    if cls is None:
        raise ConfigError(
            f"spec.type: unknown {kind} {spec.get('type')!r}; "
            f"available: {', '.join(family)}"
        )
    name = spec.get("name")
    if name is not None and not isinstance(name, str):
        raise ConfigError(f"spec.name: expected a string, got {name!r}")
    return cls, from_mapping(cls.Options, fields, "spec")


def declared(
    kind: str, spec: dict[str, Any], paths: WebPaths
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    """`(errors, {"names", "source"})`; names are None when not knowable."""
    try:
        cls, options = _parse(kind, spec)
    except ConfigError as exc:
        return [schema.split_error(exc)], {}
    errors = security.found_path_problems(schema.options_paths(options, "spec"), paths)
    model = getattr(options, "model", None)
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
    catalog = cls.declared_classes(options)
    if catalog is None:
        return [], {"names": None, "source": None}
    return [], {"names": list(catalog.names), "source": catalog.source}
