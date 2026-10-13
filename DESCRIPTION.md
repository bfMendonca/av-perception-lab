# det-depth-distance — Project Description

Comprehensive project description maintained for external LLMs. Keep this
file up to date in the same commit as any structural change (see
`AGENTS.md` § "Manter DESCRIPTION.md Atualizado").

## 1. What the Project Is

A research prototype that combines one or more object detectors (Roboflow
RF-DETR, Ultralytics YOLO26) and a monocular metric depth estimator (ByteDance Depth Anything 3, "DA3") on
the same video frames. For every detected object, the depth pixels inside
its instance mask (when the model segments) or its box are aggregated with robust statistics (mean, median,
variance, percentiles, outlier rejection) to estimate the object's distance
to the camera. The intended output is an annotated video (boxes plus
distance) and a per-detection CSV. Accuracy matters more than speed: all
models run in PyTorch, frame by frame. The user is a computer-vision
researcher working on agricultural robotics; first test data is iPhone
driving footage (1920×1080 HEVC).

Current state: task_0001 built the first RF-DETR + DA3 pipeline; task_0002
reorganised it into model-agnostic components with abstract contracts,
adapters per framework (RF-DETR, YOLO26, DA3), several detectors per frame
(concatenated, tagged by source) and a YAML run configuration. task_0003
added per-detector label maps (raw -> canonical labels), cross-detector
fusion (NMS, WBF) that keeps every merged detection, and cross-detector
agreement metrics. task_0004 added instance segmentation (RF-DETR Seg,
YOLO26-seg) and mask-based distance with a `region` policy (mask, box or
both for comparison). The CLI writes an annotated video, a per-box CSV and a
metrics JSON.

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
├── configs/                 # example run YAMLs: rfdetr_da3, yolo_o365_da3, multi_detector, multi_detector_fused
│   └── labels/              # versioned label maps: coco_driving, objects365_driving
├── src/det_depth_distance/
│   ├── core/                # contracts + orchestration, no ML framework imports
│   │   ├── types.py         # Frame, Instances, DepthResult, FrameResult, MapGeometry, CameraIntrinsics, ContractError
│   │   ├── options.py       # ConfigError, from_mapping (typed YAML -> dataclass, dotted error paths)
│   │   ├── components.py    # LoadContext, Component, DetectorOptions, Detector, DepthEstimator (ABCs)
│   │   ├── registry.py      # @register, build, available, load_adapters (ADAPTER_MODULES)
│   │   ├── labels.py        # Vocabulary, COCO / OBJECTS365 presets, identify_vocabulary, resolve_classes, LabelMap
│   │   ├── boxes.py         # numpy box geometry: iou_matrix, label_groups, greedy_match, nms_clusters, wbf_clusters
│   │   ├── aggregate.py     # Aggregator (bind_sources hook), Concat, FusionOptions, FusionAggregator, Nms, Wbf
│   │   ├── agreement.py     # frame_agreement, AgreementStats (cross-detector agreement)
│   │   └── pipeline.py      # Pipeline: detectors -> agreement -> aggregation -> depth -> distance, timings
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
- `Instances`: xyxy (N,4) float32 frame pixels (continuous edges), score, class_id, `label` (canonical), `raw_label` (model vocabulary), `source` (component name), optional `masks` (N,H,W) bool, optional `sources` (per instance, a tuple of `Provenance(source, raw_label, score)` members; set only by fusion aggregators, `provenance()` synthesises one member per unfused instance). Each per-instance field declares its validator and concat fill in its metadata (`per_instance`), so `select`/`concat`/`validate` handle future fields unchanged; an all-False mask means "no mask".
- `DepthResult`: depth_m (map_h, map_w) float32 metres, `CameraIntrinsics` at map resolution, optional sky mask, and the frame -> map `MapGeometry`.
- `CameraIntrinsics` uses continuous pixel coordinates (principal point of a centred camera at (W/2, H/2)); `MapGeometry.scale_intrinsics` maps frame K to map K.
- `FrameResult`: frame, instances, per-instance `BoxDepthStats`, depth, timings, optional `box_stats` (only with region "both": box statistics where `stats` came from the mask, None elsewhere), optional `FrameAgreement` (per detector `instances`/`duplicates` after in-detector NMS, and one `PairAgreement(a, b, matched, only_a, only_b, iou_sum)` per detector pair).
- `LabelMap` (core/labels.py): name, raw -> canonical `mapping`, `unmapped` policy (`keep`/`drop`), optional declared `vocabulary`; loaded from `configs/labels/*.yaml`, checked against the model's class names and identified vocabulary at load.
- `DistanceConfig` / `BoxDepthStats` (distance.py): parameters (incl. `mask_erode`, depth-map pixels) and per-region statistics; `mask_px` / `mask_erode` are set (and `used_mask` true) when the region was the instance mask. `Instances.masks` are filled by the Seg adapters: RF-DETR Seg (`rfdetr` `variant: seg-*`, COCO) and YOLO26-seg (`*-seg.pt`, COCO; `*-objv1-seg.pt`, expected Objects365 — not yet verified on the GPU host).
- `RunConfig` (config.py): the YAML run (source, detectors, depth, camera, aggregation, agreement, distance, render, output, device).

Per run: `video.mp4`, `detections.csv` (one row per box; task_0001 columns then `source`, `raw_label`, `n_sources`, `sources`, `source_scores`, `raw_labels` — the last three `|`-joined, best member per detector — then `used_mask`, `mask_px`, `mask_erode`, `box_distance_m`), `metrics.json` (`components`, `pipeline` incl. `agreement`, `region` and `mask_erode` policy, focal, timings incl. `det.<name>`, `agree` and `aggregate`, `detections_by_source`, `detections_by_n_sources`, `distance_regions`, `region_comparison` (mask minus box, with region "both"), `agreement` summary with >= 2 detectors, resolved `config`), optional `depth.npz`.

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
  loads detectors and depth; per frame runs every detector, the agreement
  metrics (>= 2 detectors), the aggregator, depth and
  `instance_depth_stats` (region policy `mask`/`box`/`both`), timing each
  stage. Distances are computed on the aggregated (possibly fused)
  instances. Mask distance: the frame-resolution mask is resampled by
  nearest neighbour at the depth-map pixel centres of the unshrunk box
  window (pixels outside the box ignored), eroded in map pixels with a
  back-off for thin objects, then the box statistics core (`_region_stats`)
  runs on it; box-only detectors keep the task_0002 result bit for bit.
- Label harmonisation: `Detector._map_labels` applies the detector's
  `LabelMap` after the class filter (by raw label) and before validation.
- Fusion: `FusionAggregator` is a template method (concat -> per-detector
  weights -> label groups -> `_fuse` -> instance per cluster with
  `sources`); `Nms` keeps the best weighted member's own box and score,
  `Wbf` averages boxes by score (own numpy port of
  ZFTurbo/Weighted-Boxes-Fusion, MIT, which keeps cluster membership; the
  `avg` score is capped at 1 so it stays a confidence). Both
  fuse duplicates within one detector too. Weights are fixed per detector
  (`aggregation.weights`) and only rescale scores; they are not a
  calibration. Box geometry lives in `core/boxes.py` (numpy only; no
  torchvision/scipy in the light env).
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
camera, aggregation, agreement, distance, render, output, device) is documented in the
README; the task_0001 flags were replaced without aliases.

## 8. Key Business Concepts

- **Metric depth**: per-pixel distance in metres along the optical axis (z).
- **Euclidean range**: straight-line camera-to-point distance; larger than z
  for off-axis points.
- **Focal length (px)**: camera intrinsic needed to convert DA3METRIC output
  to metres; either calibrated or estimated by an any-view DA3 model.
- **Box shrink / outlier rejection**: techniques to suppress background
  pixels inside a non-tight bounding box.
- **Instance mask / region policy**: per-instance pixel mask from a
  segmentation model; `distance.region` chooses mask (default when the
  instance has one), box, or both (records the box distance too). An
  all-False mask means "no mask" and falls back to the box.
- **Canonical label**: detector-independent class name (e.g. `vehicle`)
  that label maps assign to raw names of each vocabulary, so instances of
  different models can be compared and fused.
- **Agreement**: per detector pair, how many deduplicated instances match
  one-to-one by IoU (and how many only one detector sees).

## 9. Security Model

No secrets are read by the code. Hugging Face model downloads use the
user's existing token cache on the GPU host, outside this repository.

## 10. Development & Deployment

- Local: `uv sync`, `uv run pytest` (GPU tests deselected), `uv run ruff check . && uv run ruff format --check .`
- GPU tests (adapter contracts + 60-frame regression against task_0001): `uv run --extra infer pytest -m gpu -v` on `gpu-host`.
- Weights live outside the repo on the GPU host: RF-DETR (incl. Seg, ~135–155 MB each, downloaded and MD5-checked by `rfdetr` on first construction) in `~/.roboflow/models`; YOLO26 (`yolo26*.pt`, `*-objv1-150.pt`, `*-seg.pt`, `*-objv1-seg.pt`, from the `ultralytics/assets` v8.4.0 release, fetched manually — the adapter never downloads) in `~/dev/ultralytics/_testbed/weights`. GPU tests skip when a weight is missing.
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
- No mask refiners for box-only detectors (SAM family, task_0005), no semantic segmentation, no fusion of masks across models (a fused instance keeps one member's mask), no object tracking or temporal smoothing. Masks are never saved to disk.
- No score calibration across detectors (fusion weights are fixed per detector); no new dependency for fusion (no `ensemble-boxes`, `torchvision.ops` or `scipy` in the core).
- No ground-truth distance evaluation.
- No CI, no Docker, no git remote, no web UI or API.
