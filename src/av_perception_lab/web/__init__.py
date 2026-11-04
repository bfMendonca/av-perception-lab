"""Web interface backend (FastAPI), importable without torch.

The server derives everything it shows from the pipeline's own registries
(`core.registry` adapters and stages, dataclass field metadata): nothing in
this package names a model or a stage. Needs the `web` extra.
"""
