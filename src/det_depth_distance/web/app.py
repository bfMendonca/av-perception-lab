"""FastAPI application: thin routes over `schema` and `paths` (no torch)."""

import dataclasses
from typing import Any

import yaml
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from det_depth_distance import config
from det_depth_distance.web import schema
from det_depth_distance.web.paths import WebPaths


class ValidateBody(BaseModel):
    config: dict[str, Any]


class ParseBody(BaseModel):
    yaml: str


class DumpBody(BaseModel):
    config: dict[str, Any]
    extras: dict[str, Any] = {}


def _known_keys() -> set[str]:
    """Top-level YAML keys a configuration may hold."""
    fields = {f.name for f in dataclasses.fields(config.RunConfig)}
    return fields - set(config.PIPELINE_DISTANCE_KEYS)


def create_app(paths: WebPaths) -> FastAPI:
    """Build the application serving the catalog and the config endpoints."""
    app = FastAPI(title="det-depth-distance")
    app.state.paths = paths

    @app.get("/api/catalog")
    def catalog() -> dict[str, Any]:
        return schema.module_catalog()

    @app.post("/api/config/validate")
    def validate(body: ValidateBody) -> dict[str, Any]:
        return {"errors": schema.validate_config(body.config)}

    @app.post("/api/config/parse")
    def parse(body: ParseBody) -> dict[str, Any]:
        try:
            data = yaml.safe_load(body.yaml) or {}
        except yaml.YAMLError as exc:
            raise HTTPException(422, f"invalid YAML: {exc}") from None
        if not isinstance(data, dict):
            raise HTTPException(422, "the top level must be a mapping")
        known = _known_keys()
        return {
            "config": {k: v for k, v in data.items() if k in known},
            "extras": {k: v for k, v in data.items() if k not in known},
        }

    @app.post("/api/config/dump")
    def dump(body: DumpBody) -> dict[str, str]:
        merged = {**body.config, **body.extras}
        return {"yaml": yaml.safe_dump(merged, sort_keys=False, allow_unicode=True)}

    return app
