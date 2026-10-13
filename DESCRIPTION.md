# det-depth-distance — Project Description

Comprehensive project description maintained for external LLMs. Keep this
file up to date in the same commit as any structural change (see
`AGENTS.md` § "Manter DESCRIPTION.md Atualizado").

## 1. What the Project Is

A research prototype that combines one or more object detectors (Roboflow
RF-DETR, Ultralytics YOLO26) and a monocular metric depth estimator (ByteDance Depth Anything 3, "DA3") on
the same video frames. For every detected bounding box, the depth pixels
inside the box are aggregated with robust statistics (mean, median,
variance, percentiles, outlier rejection) to estimate the object's distance
to the camera. The intended output is an annotated video (boxes plus
distance) and a per-detection CSV. Accuracy matters more than speed: all
models run in PyTorch, frame by frame. The user is a computer-vision
researcher working on agricultural robotics; first test data is iPhone
driving footage (1920×1080 HEVC).

Current state: task_0001 built the first RF-DETR + DA3 pipeline; task_0002
reorganised it into model-agnostic components with abstract contracts,
adapters per framework (RF-DETR, YOLO26, DA3), several detectors per frame
(concatenated, tagged by source) and a YAML run configuration. The CLI
writes an annotated video, a per-box CSV and a metrics JSON.

## 2. Technology Stack

| Layer | Choice |
|---|---|
| Language | Python 3.11 (`requires-python >=3.11,<3.12`) |
| Env / deps | Astral uv, `pyproject.toml` + versioned `uv.lock` |
| Base deps | numpy<2 (required by depth-anything-3), opencv-python, pillow, pyyaml>=6 (run configuration, read in the light env) |
| `infer` extra (GPU host only) | torch 2.8.0 + torchvision 0.23.0 + xformers 0.0.32.post2 from the PyTorch cu128 index; `rfdetr` @ git 1f2af2ea; `depth-anything-3` @ git 3d835ec; `addict` (imported by depth-anything-3 but not declared by it); `ultralytics==8.4.175` (YOLO26; **AGPL-3.0** — acceptable for this internal research prototype, optional extra) |
| Dev group | pytest, ruff |
| Video I/O | system ffmpeg/ffprobe |
| Inference host | `gpu-host`: NVIDIA RTX PRO 6000 Blackwell 96 GB (sm_120), no sudo |
| CI / Docker | none |

## 3. Project Structure

```
det-depth-distance/
├── pyproject.toml           # uv manifest; "infer" extra holds the GPU stack; pytest "gpu" marker
├── uv.lock
├── configs/                 # example run YAMLs: rfdetr_da3, yolo_o365_da3, multi_detector
├── src/det_depth_distance/
│   ├── core/                # contracts + orchestration, no ML framework imports
│   │   ├── types.py         # Frame, Instances, DepthResult, FrameResult, MapGeometry, CameraIntrinsics, ContractError
│   │   ├── options.py       # ConfigError, from_mapping (typed YAML -> dataclass, dotted error paths)
│   │   ├── components.py    # LoadContext, Component, DetectorOptions, Detector, DepthEstimator (ABCs)
│   │   ├── registry.py      # @register, build, available, load_adapters (ADAPTER_MODULES)
│   │   ├── labels.py        # Vocabulary, COCO / OBJECTS365 presets, identify_vocabulary, resolve_classes
│   │   ├── aggregate.py     # Aggregator (bind_sources hook), Concat
│   │   └── pipeline.py      # Pipeline: detectors -> aggregation -> depth -> distance, timings
│   ├── adapters/detectors/  # rfdetr.py (RFDETRDetector), ultralytics_yolo.py (YoloDetector)
│   ├── adapters/depth/      # da3.py (DA3Depth + geometry/focal/unit helpers)
│   ├── io/                  # video.py (ffmpeg pipes), source.py (VideoSource), sinks.py (CSV/video/npz/metrics)
│   ├── camera.py            # IntrinsicsProvider: FixedIntrinsics, EstimatedIntrinsics
│   ├── config.py            # RunConfig + sections, load_config, apply_overrides (--set)
│   ├── distance.py          # pure numpy: box -> map pixels -> robust stats -> distance
│   ├── render.py            # PIL boxes/labels/HUD (colour by class or source), depth panel + legend
│   ├── runtime.py           # DeviceMonitor (torch, lazy), package versions, git commit
│   ├── cli.py               # --config/--set/--list-models; composition root
│   └── __main__.py          # python -m det_depth_distance
├── tests/                   # pytest without GPU; tests/gpu/ (auto-marked "gpu") runs on gpu-host
├── utils/sync_gpu_host.sh    # rsync push code / pull outputs with the GPU host
├── docs/                    # task files (docs/task_DDDD.md) + template
└── _testbed/                # git-ignored: test videos, outputs, regression reference, ultralytics settings
```

## 4. Data Model

No database. Everything crossing a component boundary is a frozen numpy
dataclass (`core/types.py`), never a framework object:

- `Frame`: index, t, rgb (H, W, 3) uint8 upright RGB.
- `Instances`: xyxy (N,4) float32 frame pixels (continuous edges), score, class_id, `label` (canonical), `raw_label` (model vocabulary), `source` (component name), optional `masks` (N,H,W) bool. Each per-instance field declares its validator and concat fill in its metadata (`per_instance`), so `select`/`concat`/`validate` handle future fields unchanged; an all-False mask means "no mask".
- `DepthResult`: depth_m (map_h, map_w) float32 metres, `CameraIntrinsics` at map resolution, optional sky mask, and the frame -> map `MapGeometry`.
- `CameraIntrinsics` uses continuous pixel coordinates (principal point of a centred camera at (W/2, H/2)); `MapGeometry.scale_intrinsics` maps frame K to map K.
- `FrameResult`: frame, instances, per-instance `BoxDepthStats`, depth, timings.
- `DistanceConfig` / `BoxDepthStats` (distance.py): parameters and per-box statistics.
- `RunConfig` (config.py): the YAML run (source, detectors, depth, camera, aggregation, distance, render, output, device).

Per run: `video.mp4`, `detections.csv` (one row per box; task_0001 columns then `source`, `raw_label`), `metrics.json` (`components`, `pipeline`, focal, timings incl. `det.<name>` and `aggregate`, `detections_by_source`, resolved `config`), optional `depth.npz`.

## 5. Architecture Patterns

- Two environments from one manifest: a light local env for developing and
  unit-testing, and the heavy `infer` extra on the GPU host. Framework
  imports (`torch`, `rfdetr`, `depth_anything_3`, `ultralytics`) happen only
  inside adapters' `_load` (and `runtime.DeviceMonitor`), so `--help`,
  `--list-models` and config validation run without torch.
- Components: `Component` (construct = validate options only; `load(ctx)`
  heavy and idempotent; `close()` frees GPU; `describe()` for metrics).
  `Detector.__call__` is a template method: adapter `_predict` -> class
  filter by raw label -> label mapping hook -> contract validation.
  `DepthEstimator.needs_intrinsics` is known before loading.
- Registry: `@register(kind, type)`; `registry.build(kind, spec, path=...)`
  turns a YAML spec `{type, name?, **options}` into a component via the
  adapter's `Options` dataclass. New adapter = new module + entry in
  `ADAPTER_MODULES`.
- Pipeline: resolves camera intrinsics first (only if the depth model needs
  them; the focal-estimation DA3 model is loaded, used and released), then
  loads detectors and depth; per frame runs every detector, the aggregator
  (`concat`), depth and `box_depth_stats`, timing each stage.
- Sinks (CSV, video, npz, metrics) consume `FrameResult`s; the CLI is only a
  composition root. Configuration errors raise `ConfigError` (exit code 2);
  adapter bugs raise `ContractError`.
- DA3METRIC-LARGE output is not in metres: `metres = raw * (fx + fy) / 2 / 300`,
  with fx/fy at the depth-map resolution. Model kind is decided from the
  checkpoint's `model_name`, not `Prediction.is_metric`.
- Ultralytics specifics: `predict()` treats numpy input as BGR (the adapter
  flips RGB); class names come from the checkpoint (the Objects365 YOLO26
  names differ from ultralytics' `Objects365.yaml`); `YOLO_OFFLINE=1`,
  `YOLO_AUTOINSTALL=False` and `YOLO_CONFIG_DIR=<repo>/_testbed/config` are
  set before the lazy import (no telemetry, no pip installs, no writes to
  `~/.config`); `half` is passed as `quantize=16` (8.4 deprecates `half`).
- `distance.py` is pure numpy; outlier filters and statistics are registries
  (`_OUTLIER_FILTERS`, `_STATS`).
- DA3 is called with one frame per `inference()` call (a list switches
  any-view models to multi-view mode).

## 6. Authentication & Authorization

None. Local CLI tool; no users, sessions or network services.

## 7. URL / API Design

No network API. The interface is the CLI `det-depth-distance` /
`python -m det_depth_distance --config run.yaml [VIDEO] [--set key=value]...`
and `--list-models`. The YAML schema (sections source, detectors, depth,
camera, aggregation, distance, render, output, device) is documented in the
README; the task_0001 flags were replaced without aliases.

## 8. Key Business Concepts

- **Metric depth**: per-pixel distance in metres along the optical axis (z).
- **Euclidean range**: straight-line camera-to-point distance; larger than z
  for off-axis points.
- **Focal length (px)**: camera intrinsic needed to convert DA3METRIC output
  to metres; either calibrated or estimated by an any-view DA3 model.
- **Box shrink / outlier rejection**: techniques to suppress background
  pixels inside a non-tight bounding box.

## 9. Security Model

No secrets are read by the code. Hugging Face model downloads use the
user's existing token cache on the GPU host, outside this repository.

## 10. Development & Deployment

- Local: `uv sync`, `uv run pytest` (GPU tests deselected), `uv run ruff check . && uv run ruff format --check .`
- GPU tests (adapter contracts + 60-frame regression against task_0001): `uv run --extra infer pytest -m gpu -v` on `gpu-host`.
- GPU host: `utils/sync_gpu_host.sh push`, then on `gpu-host`
  `~/.local/bin/uv sync --extra infer`; outputs fetched with
  `utils/sync_gpu_host.sh pull`.
- No deployment target beyond running the CLI on the GPU host.

## 11. Conventions Summary

- Code, docstrings, comments in English; task files, AGENTS.md, CLAUDE.md,
  README in pt-BR; CHANGELOG in pt-BR ASCII.
- Ruff defaults; snake_case functions/modules, PascalCase classes.
- Conventional Commits with pt-BR subject; scopes: core, adapters, io, distance, camera, render, cli, configs, tests, utils, deps, docs.
- Never commit to `main`; work on `feat/`/`fix/` branches.

## 12. Task Documentation & Lifecycle (docs/)

This project uses task-driven development: every implementation has a
`docs/task_DDDD.md` file (Descrição, Escopo, Planejamento, Análise de
Impacto, Verificação, Dependências) created before coding. On completion the
title is marked `(CONCLUÍDA)` and the file gains `## Detalhes de
Implementação` and `## Caderno de Testes` sections in the same commit as the
implementation. See `AGENTS.md` for the mandatory rules.

## 13. What This Project Does NOT Have

- No TensorRT/ONNX path, no batching, no real-time optimisation.
- No segmentation masks (the `Instances.masks` contract exists, but no adapter fills it and distance uses boxes only), no cross-model fusion (NMS/WBF), no object tracking or temporal smoothing.
- No ground-truth distance evaluation.
- No CI, no Docker, no git remote, no web UI or API.
