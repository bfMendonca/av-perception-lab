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
both for comparison). task_0005 added mask refiners that turn any
detector's boxes into instance masks (SAM family: SAM 2.1, EdgeTAM, SAM-HQ,
SAM, SAM 3 tracker via `transformers`, plus MobileSAM/SAM via Ultralytics),
and several named semantic segmenters per run (SegFormer Cityscapes,
TwinLiteNet+ drivable area/lanes) whose excluded classes make pixels
ineligible for distance. The CLI writes an annotated video, a per-box CSV and a
metrics JSON. task_0006 added an optional tracker (own numpy IoU tracker with
constant-velocity box prediction; stable `track_id` per object) and per-track
temporal smoothing of the distance (1D Kalman with relative speed, EMA or
causal median), with no new dependency. The GPU validation of task_0005 and
task_0006 (contract/regression tests, real-video runs) is still pending on
the inference host.

## 2. Technology Stack

| Layer | Choice |
|---|---|
| Language | Python 3.11 (`requires-python >=3.11,<3.12`) |
| Env / deps | Astral uv, `pyproject.toml` + versioned `uv.lock` |
| Base deps | numpy<2 (required by depth-anything-3), opencv-python, pillow, pyyaml>=6 (run configuration, read in the light env) |
| `infer` extra (GPU host only) | torch 2.8.0 + torchvision 0.23.0 + xformers 0.0.32.post2 from the PyTorch cu128 index; `rfdetr` @ git 1f2af2ea; `depth-anything-3` @ git 3d835ec; `addict` (imported by depth-anything-3 but not declared by it); `ultralytics==8.4.175` (YOLO26 and the optional `ultralytics_sam` refiner; **AGPL-3.0** — acceptable for this internal research prototype, optional extra); `transformers>=5.1,<6` (locked 5.19.0; previously only transitive via rfdetr; SAM-family refiners and SegFormer); `timm>=1.0.17` (locked 1.0.30; only for EdgeTAM's TimmWrapper backbone) |
| External code (GPU host) | TwinLiteNet+ clone `~/dev/TwinLiteNetPlus` @ 90f1b86 (MIT; not a package, imported with a temporary `sys.path` entry; weights in its `pretrained/`) |
| Model licences (task_0005) | Apache-2.0: SAM, SAM 2.1, SAM-HQ, EdgeTAM, MobileSAM, `timm`; "SAM License", gated: SAM 3 (`sam3_tracker`, optional, never a default); `other` (NVIDIA terms, read before non-academic use): SegFormer checkpoints; MIT: TwinLiteNet+; AGPL-3.0: ultralytics |
| Dev group | pytest, ruff |
| Video I/O | system ffmpeg/ffprobe |
| Inference host | `gpu-host`: NVIDIA RTX PRO 6000 Blackwell 96 GB (sm_120), no sudo |
| CI / Docker | none |

## 3. Project Structure

```
det-depth-distance/
├── pyproject.toml           # uv manifest; "infer" extra holds the GPU stack; pytest "gpu" marker
├── uv.lock
├── configs/                 # example run YAMLs: rfdetr_da3, yolo_o365_da3, multi_detector(_fused), *_seg_da3, rfdetr_sam2_da3, yolo_o365_sam2_da3, rfdetr_semantic_da3, rfdetr_lanes_scene_da3, rfdetr_da3_track
│   └── labels/              # versioned label maps: coco_driving, objects365_driving
├── src/det_depth_distance/
│   ├── core/                # contracts + orchestration, no ML framework imports
│   │   ├── types.py         # Frame, Instances, DepthResult, FrameResult, RefinedMasks, SemanticMap, TrackEstimate, MapGeometry, CameraIntrinsics, ContractError
│   │   ├── options.py       # ConfigError, from_mapping (typed YAML -> dataclass, dotted error paths)
│   │   ├── components.py    # LoadContext, Component, DetectorOptions, Detector, DepthEstimator, RefinerOptions, MaskRefiner, SemanticSegmenter, TrackerOptions, Tracker (ABCs)
│   │   ├── registry.py      # @register, build, available, load_adapters (ADAPTER_MODULES)
│   │   ├── labels.py        # Vocabulary, COCO / OBJECTS365 presets, identify_vocabulary, resolve_classes, LabelMap, SEMANTIC_NAME_PATTERN, group_semantic_labels
│   │   ├── boxes.py         # numpy box geometry: iou_matrix, label_groups, greedy_match, nms_clusters, wbf_clusters
│   │   ├── aggregate.py     # Aggregator (bind_sources hook), Concat, FusionOptions, FusionAggregator, Nms, Wbf
│   │   ├── agreement.py     # frame_agreement, AgreementStats (cross-detector agreement)
│   │   ├── tracking.py      # IouTracker (registered "iou"), association_keys, TrackSummary (metrics.json "tracks")
│   │   └── pipeline.py      # Pipeline: detectors -> agreement -> aggregation -> refiners -> tracker -> depth -> semantic -> eligibility -> distance -> smoothing, timings
│   ├── adapters/detectors/  # rfdetr.py (RFDETRDetector), ultralytics_yolo.py (YoloDetector)
│   ├── adapters/depth/      # da3.py (DA3Depth + geometry/focal/unit helpers)
│   ├── adapters/refiners/   # box_prompt.py (BoxPromptRefiner, numpy), hf_sam.py (HFSamRefiner + sam2/edgetam/sam_hq/sam/sam3_tracker), ultralytics_sam.py
│   ├── adapters/semantic/   # segformer.py (SegformerSegmenter), twinlitenet.py (TwinLiteNetSegmenter)
│   ├── adapters/_hf_hub.py  # offline from_pretrained + exact download command
│   ├── adapters/_ultralytics_env.py  # YOLO_OFFLINE / YOLO_AUTOINSTALL / YOLO_CONFIG_DIR policy (YOLO and SAM)
│   ├── io/                  # video.py (ffmpeg pipes), source.py (VideoSource), sinks.py (CSV/video/npz/metrics)
│   ├── camera.py            # IntrinsicsProvider: FixedIntrinsics, EstimatedIntrinsics
│   ├── config.py            # RunConfig + sections, load_config, apply_overrides (--set)
│   ├── distance.py          # pure numpy: box -> map pixels -> robust stats -> distance
│   ├── temporal.py          # pure numpy: SmoothingConfig, KalmanFilter1D / EmaFilter / MedianFilter (FILTERS), TrackSmoother
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
- `Instances`: xyxy (N,4) float32 frame pixels (continuous edges), score, class_id, `label` (canonical), `raw_label` (model vocabulary), `source` (component name), optional `masks` (N,H,W) bool, optional `sources` (per instance, a tuple of `Provenance(source, raw_label, score)` members; set only by fusion aggregators, `provenance()` synthesises one member per unfused instance), optional `mask_source` (producer of each mask: detector, refiner, or the fusion member that owns it; "" = none) and `mask_score` (float32 refiner quality, NaN for detector masks); `mask_sources()` synthesises the producer when unset; optional `track_id` (N,) int64 >= 1, set by the tracker, stable across frames. Each per-instance field declares its validator and concat fill in its metadata (`per_instance`), so `select`/`concat`/`validate` handle future fields unchanged; an all-False mask means "no mask".
- `DepthResult`: depth_m (map_h, map_w) float32 metres, `CameraIntrinsics` at map resolution, optional sky mask, and the frame -> map `MapGeometry`.
- `CameraIntrinsics` uses continuous pixel coordinates (principal point of a centred camera at (W/2, H/2)); `MapGeometry.scale_intrinsics` maps frame K to map K.
- `RefinedMasks`: masks (K,H,W) bool + score (K,) float32, the output of `MaskRefiner._refine` for its K targets.
- `SemanticMap`: labels (H,W) int32 indexing `class_names`; `fractions()` per class.
- `TrackEstimate`: smoothed state of one track at one frame: distance (m), distance_std, rel_speed (m/s, d(distance)/dt, negative = approaching), rel_speed_std (the three NaN for ema/median), n_updates.
- `FrameResult`: frame, instances, per-instance `BoxDepthStats`, depth, timings, optional `temporal` (with smoothing: one `TrackEstimate | None` per instance), `semantic` (segmenter name -> `SemanticMap`, config order; `{}` without segmenters; maps are never merged), optional `box_stats` (only with region "both": box statistics where `stats` came from the mask, None elsewhere), optional `FrameAgreement` (per detector `instances`/`duplicates` after in-detector NMS, and one `PairAgreement(a, b, matched, only_a, only_b, iou_sum)` per detector pair).
- `LabelMap` (core/labels.py): name, raw -> canonical `mapping`, `unmapped` policy (`keep`/`drop`), optional declared `vocabulary`; loaded from `configs/labels/*.yaml`, checked against the model's class names and identified vocabulary at load.
- `DistanceConfig` / `BoxDepthStats` (distance.py): parameters (incl. `mask_erode`, depth-map pixels) and per-region statistics; `mask_px` / `mask_erode` are set (and `used_mask` true) when the region was the instance mask. `Instances.masks` are filled by the Seg adapters: RF-DETR Seg (`rfdetr` `variant: seg-*`, COCO) and YOLO26-seg (`*-seg.pt`, COCO; `*-objv1-seg.pt`, expected Objects365 — not yet verified on the GPU host).
- `SmoothingConfig` (temporal.py): `filter` (kalman/ema/median), `accel_std`, `std_scale`, `rel_floor`, `init_speed_std`, `tau_s`, `window_s`, `reset_after_s` (times in seconds).
- `RunConfig` (config.py): the YAML run (source, detectors, depth, camera, aggregation, refiners, semantic, tracking (flat tracker spec or null), smoothing (`SmoothingConfig` or null), agreement, distance incl. `exclude_semantic` "<segmenter>.<class>" labels, render incl. `semantic_alpha`/`semantic_overlay`/`show_smoothed`/`show_speed`, output, device).

Per run: `video.mp4`, `detections.csv` (one row per box; task_0001 columns then `source`, `raw_label`, `n_sources`, `sources`, `source_scores`, `raw_labels` — the last three `|`-joined, best member per detector — then `used_mask`, `mask_px`, `mask_erode`, `box_distance_m`, then `mask_source`, `mask_score`, then `track_id`, `distance_smooth_m`, `distance_smooth_std_m`, `rel_speed_mps`, `rel_speed_std_mps` (empty without tracking/smoothing; `distance_m` stays raw); no semantic column), `metrics.json` (`components` incl. `refiners`/`semantic`/`tracker`, `pipeline` incl. `agreement`, `region`, `mask_erode`, `exclude_semantic` and `smoothing` policy, focal, timings incl. `det.<name>`, `agree`, `aggregate`, `refine(.<name>)`, `track`, `semantic(.<name>)`, `smooth`, `tracks` (with a tracker: per-track summary — label mode and changes, sources, span, coverage, gaps, distance range, raw/smoothed jitter, relative-speed percentiles; tracks shorter than 1 s only counted), `detections_by_source`, `detections_by_n_sources`, `distance_regions`, `mask_sources`, `refined` (targets/filled per refiner), `semantic` (with segmenters: per-map class fractions and excluded fraction, plus the excluded fraction of the union), `region_comparison` (mask minus box, with region "both"), `agreement` summary with >= 2 detectors, versions incl. `transformers`/`timm`, resolved `config`), optional `depth.npz` (plus `semantic_labels_<name>` on the depth-map grid and `semantic_classes_<name>` per segmenter).

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
  Stage order (task_0005): detectors -> agreement -> aggregation -> mask
  refiners (in order) -> depth -> semantic segmenters (in order) -> pixel
  eligibility (`distance.map_eligibility`: not sky and not in the union of
  the excluded classes, labels sampled on the depth-map grid) -> distance.
  Without refiners and segmenters the outputs are those of task_0004.
  task_0006 inserts the tracker after the refiners and per-track smoothing
  after the distance; without them the outputs are those of task_0005.
- Tracking: `Tracker(Component)` template method (`_assign` returns one id
  per instance; the base checks same N/order, int64 >= 1, unique per frame;
  `load()`/`reset()` restart ids at 1). `IouTracker` (core/tracking.py,
  numpy over `core/boxes.py`): greedy one-to-one IoU matching between each
  track's constant-velocity predicted box (px/s, dt from `Frame.t`, so
  stride-aware) and the frame's instances, within association groups
  (`source` for unfused instances, one group for fused ones, + label only
  with `class_aware`); tracks die after `max_age_s`. Not ultralytics'
  ByteTrack/BoT-SORT (needs `lap`, imports torch, counts calls, AGPL-3.0).
- Smoothing: `TrackSmoother` (temporal.py) keeps one `TemporalFilter` per
  track id (filters registry `FILTERS`, same idiom as `_STATS`); measurement
  std `max(std_scale * box std, rel_floor * distance)`; state dropped after
  `reset_after_s` without a measurement. Smoothing needs tracking
  (`ConfigError` otherwise).
- Mask refiners (D9 hierarchy): `MaskRefiner` (core, the only contract the
  pipeline knows: target selection `missing`/`all`, merge where an empty
  refined mask never replaces one, `mask_source`/`mask_score`) ->
  `BoxPromptRefiner` (adapters/refiners/box_prompt.py, numpy only: one frame
  preparation, boxes in batches of `max_boxes`, highest-predicted-IoU mask,
  optional object score) -> `HFSamRefiner` (transformers SAM API; family
  differences as class attributes and two hooks) -> thin registered
  families `sam2`, `edgetam`, `sam_hq`, `sam`, `sam3_tracker`; plus
  `UltralyticsSamRefiner` (`ultralytics_sam`, predictor API, BGR) as a second
  backend. Options are layered `RefinerOptions` -> `BoxPromptOptions` ->
  `HFSamOptions` -> family `Options`. Weights are read with
  `local_files_only=True` from `hf_cache` (example configs: `weights/hf`);
  nothing is downloaded at run time.
- Semantic segmenters: `SemanticSegmenter` (template `__call__` validating
  `SemanticMap` against the frame and `class_names`); `name` is required in
  the YAML and namespaces excluded labels (`scene.road`); a prefix is checked
  at config load, the class after the segmenter loads.
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
  set by `adapters._ultralytics_env.apply_ultralytics_env()` when the YOLO
  or SAM adapter module is imported, before the lazy import (no telemetry, no pip installs, no writes to
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
camera, aggregation, refiners, semantic, tracking, smoothing, agreement, distance, render, output, device) is documented in the
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
- **Mask refiner**: a box-prompted segmentation model (SAM family) that
  fills instance masks from detector boxes; `targets: missing` keeps native
  masks of segmentation detectors.
- **Semantic exclusion**: per-pixel class maps from one or more named
  segmenters; pixels of excluded `<segmenter>.<class>` labels (e.g. road,
  lane lines, vegetation) are not used for distance. Cityscapes/BDD100K are
  street domains: in agricultural scenes vegetation/terrain cover the crop.
- **Track / relative speed**: an object's identity across frames
  (`track_id`) and the rate of change of its smoothed distance (m/s;
  negative = approaching). The DA3 noise is correlated in time, so
  smoothing removes jitter much more than level error, and speed beyond
  ~30 m is unreliable.

## 9. Security Model

No secrets are read by the code. Hugging Face model downloads use the
user's existing token cache on the GPU host, outside this repository; the
gated `facebook/sam3` checkpoint additionally needs approved access and
`HF_TOKEN` there. Adapters never download at run time.

## 10. Development & Deployment

- Local: `uv sync`, `uv run pytest` (GPU tests deselected), `uv run ruff check . && uv run ruff format --check .`
- GPU tests (adapter contracts + 60-frame regression against task_0001): `uv run --extra infer pytest -m gpu -v` on `gpu-host`.
- Refiner and SegFormer weights: Hugging Face cache `weights/hf` in the repo (git-ignored and excluded from the rsync push), downloaded explicitly with `snapshot_download(..., allow_patterns=['*.json', 'model.safetensors'])` (SegFormer: `pytorch_model.bin`) — command in the README; MobileSAM `mobile_sam.pt` beside the YOLO weights; TwinLiteNet+ weights in its clone.
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
- No fusion of masks across models (a fused instance keeps one member's mask) and no fusion of semantic maps (each segmenter keeps its own map); no FastSAM (not one-mask-per-box), no native `sam3` package (the SAM 3 tracker is used through `transformers`); no semantic CSV columns or video colour legend; no appearance-based (ReID) or ultralytics tracker, no 3D association and no per-track colour in the video. Instance masks are never saved to disk (semantic labels are, on the depth-map grid, with `save_depth`).
- No score calibration across detectors (fusion weights are fixed per detector); no new dependency for fusion (no `ensemble-boxes`, `torchvision.ops` or `scipy` in the core).
- No ground-truth distance evaluation.
- No CI, no Docker, no git remote, no web UI or API.
