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
causal median), with no new dependency. task_0007 added camera calibration
(chessboard/ChArUco tool, `camera.calibration` with rescaling, a lens check
against the video's QuickTime metadata, optional frame undistortion and a
cross-check against the estimated focal) and the evaluation of run outputs
against field reference distances (error tables and plots). The GPU
validation of task_0005–0007 (contract/regression tests, real-video runs),
the real calibration of the test iPhone and the field reference data are
still pending.

## 2. Technology Stack

| Layer | Choice |
|---|---|
| Language | Python 3.11 (`requires-python >=3.11,<3.12`) |
| Env / deps | Astral uv, `pyproject.toml` + versioned `uv.lock` |
| Base deps | numpy<2 (required by depth-anything-3), opencv-python (also calibration: `calibrateCameraExtended`, `findChessboardCornersSB`, `cv2.aruco` ChArUco, `FileStorage`, undistortion; no `opencv-contrib`), pillow, pyyaml>=6 (run configuration and field references, read in the light env) |
| `web` extra (optional) | `fastapi`, `uvicorn`, `python-multipart`, `httpx`, `av` (locked; python-multipart 0.0.32 added by task_0013; `av` (PyAV 18.1) declared by task_0016 for the `.h264` remux, already transitive via `infer`): backend of the web interface (`det_depth_distance.web`), torch-free |
| Web front-end (`web/`, not Python) | React 18.3.1, TypeScript 5.9, Vite 8, `yaml` 2 (raw-YAML field editor); tests with vitest 5 + Testing Library + jsdom; Node 24 LTS installed without sudo (README); `web/node_modules` and `web/dist` are not versioned |
| `eval` extra (optional) | `matplotlib>=3.8` (locked 3.11.2): plots of `det-depth-distance-evaluate --plot` only; imported lazily |
| `infer` extra (GPU machine; the laptop's 4 GB GPU in an isolated env, `UV_PROJECT_ENVIRONMENT=_testbed/venv-infer`) | torch 2.8.0 + torchvision 0.23.0 + xformers 0.0.32.post2 from the PyTorch cu128 index; `rfdetr` @ git 1f2af2ea; `depth-anything-3` @ git 3d835ec; `addict` (imported by depth-anything-3 but not declared by it); `ultralytics==8.4.175` (YOLO26 and the optional `ultralytics_sam` refiner; **AGPL-3.0** — acceptable for this internal research prototype, optional extra); `transformers>=5.1,<6` (locked 5.19.0; previously only transitive via rfdetr; SAM-family refiners and SegFormer); `timm>=1.0.17` (locked 1.0.30; only for EdgeTAM's TimmWrapper backbone) |
| External code (GPU machine) | TwinLiteNet+ clone @ 90f1b86 (MIT; not a package, imported with a temporary `sys.path` entry), default `weights/_code/TwinLiteNetPlus` (cloned by `--prepare`; `repo:` overrides, e.g. `~/dev/TwinLiteNetPlus`); `pretrained/<variant>.pth` is user-provided (upstream Google Drive) |
| Model licences (task_0005) | Apache-2.0: SAM, SAM 2.1, SAM-HQ, EdgeTAM, MobileSAM, `timm`; "SAM License", gated: SAM 3 (`sam3_tracker`, optional, never a default); `other` (NVIDIA terms, read before non-academic use): SegFormer checkpoints; MIT: TwinLiteNet+; AGPL-3.0: ultralytics |
| Dev group | pytest, ruff |
| Video I/O | system ffmpeg/ffprobe |
| GPU used so far | NVIDIA RTX PRO 6000 Blackwell 96 GB (sm_120); laptop RTX A1000 4 GB for the light example |
| CI / Docker | none |

## 3. Project Structure

```
det-depth-distance/
├── pyproject.toml           # uv manifest; "infer" extra holds the GPU stack, "eval" extra matplotlib, "web" extra FastAPI; 4 console scripts; pytest "gpu" marker
├── uv.lock
├── web/                     # front-end (Vite React TS): src/api (types, client, video, models), src/editor (yamlState store, Lanes, ModuleCard, Inspector, fieldControls, DiffView, registry), src/models (useModelStatus hook + ReadinessProvider, ReadinessBanner, ModelRequirements, readiness.ts/cardReadiness.ts rules, statusStyles.css), src/video (VideoDeck, DropZone, Player, SegmentBar, JobPanel, segment.ts, useVideoSession, useJobs, useProcess), src/api/runs.ts, src/runs (RunsView, RunList, RunDetail, CompareView (KpiCards, ConfigDiff), DistanceChart, JsonTree, Artifacts, artifactViews registry, format.ts), src/ui (Shell, route.ts, theme.css); built to web/dist (not versioned)
├── configs/                 # example run YAMLs: rfdetr_da3, yolo_o365_da3, multi_detector(_fused), *_seg_da3, rfdetr_sam2_da3, yolo_o365_sam2_da3, rfdetr_semantic_da3, rfdetr_lanes_scene_da3, rfdetr_da3_track, rfdetr_da3_calibrated, yolo_light_da3, rfdetr_da3_motion, rfdetr_sam2_lanes_motion_da3
│   ├── labels/              # versioned label maps: coco_driving, objects365_driving
│   └── references/          # example_references.yaml (fictitious field-reference template)
├── src/det_depth_distance/
│   ├── core/                # contracts + orchestration, no ML framework imports
│   │   ├── types.py         # Frame, Instances, DepthResult, FrameResult, RefinedMasks, SemanticMap, TrackEstimate, Trajectory, MotionEstimate, MapGeometry, CameraIntrinsics, ContractError
│   │   ├── options.py       # ConfigError, from_mapping (typed YAML -> dataclass, dotted error paths)
│   │   ├── components.py    # LoadContext, Component (+ requirements(options) hook), DetectorOptions, Detector, DepthEstimator, RefinerOptions, MaskRefiner, SemanticSegmenter, TrackerOptions, Tracker (ABCs)
│   │   ├── registry.py      # @register, build, available, load_adapters (ADAPTER_MODULES)
│   │   ├── requirements.py  # Requirements (assets, CodeRepo (hex commit pin), packages), registered Asset kinds (HuggingFaceAsset, UrlAsset, UserProvided; present(root)), weights_root, ModelStatus, check() (torch-free)
│   │   ├── provision/       # provisioning engine (framework-free): engine.py (Source protocol, register_source, ProvisionError/ProvisionCancelled, prepare(req, root, progress, cancel) -> ModelStatus), sources.py (built-in sources: Hugging Face snapshot_download (lazy import), URL via urllib (.part, atomic rename, accepted only when size/sha256/Content-Length verifies it, Range + If-Range resume only with a declared size or sha256, http(s)-only redirects), git clone + pinned checkout (hex commit, timeout, process-group kill on cancel), UserProvided never downloads; engine.install_packages / with_package_plan), packages.py (packages layer: declared_extras, detect_extras via importlib.metadata + find_spec, sync_command never bare (--locked --inexact, every detected extra), dry_run -> DryRunResult, install (process group, streamed), plan_packages -> PackagePlan ready|base|install|conflict|invalid; the base extra `infer` is never installed))
│   │   ├── labels.py        # Vocabulary, COCO / OBJECTS365 presets, identify_vocabulary, resolve_classes, LabelMap, SEMANTIC_NAME_PATTERN, group_semantic_labels
│   │   ├── boxes.py         # numpy box geometry: iou_matrix, label_groups, greedy_match, nms_clusters, wbf_clusters
│   │   ├── aggregate.py     # Aggregator (bind_sources hook), Concat, FusionOptions, FusionAggregator, Nms, Wbf
│   │   ├── agreement.py     # frame_agreement, AgreementStats (cross-detector agreement)
│   │   ├── tracking.py      # IouTracker (registered "iou"), association_keys, TrackSummary (metrics.json "tracks")
│   │   └── pipeline.py      # Pipeline: detectors -> agreement -> aggregation -> refiners -> tracker -> depth -> semantic -> eligibility -> distance -> smoothing -> motion, timings
│   ├── adapters/detectors/  # rfdetr.py (RFDETRDetector), ultralytics_yolo.py (YoloDetector)
│   ├── adapters/depth/      # da3.py (DA3Depth + geometry/focal/unit helpers)
│   ├── adapters/refiners/   # box_prompt.py (BoxPromptRefiner, numpy), hf_sam.py (HFSamRefiner + sam2/edgetam/sam_hq/sam/sam3_tracker), ultralytics_sam.py
│   ├── adapters/semantic/   # segformer.py (SegformerSegmenter), twinlitenet.py (TwinLiteNetSegmenter)
│   ├── adapters/_hf_hub.py  # offline from_pretrained (default cache <weights root>/hf), hf_asset() declaration, HF_FACTS (sizes/licences), download command
│   ├── adapters/_weights.py # under_root() (explicit path inside the weights root?), not_prepared() error naming --prepare
│   ├── adapters/_ultralytics.py  # ultralytics/assets v8.4.0 table (YOLO26, SAM), resolve_weights (bare file name -> weights/ultralytics/, other paths unchanged), weights_asset
│   ├── adapters/_ultralytics_env.py  # YOLO_OFFLINE / YOLO_AUTOINSTALL / YOLO_CONFIG_DIR policy (YOLO and SAM)
│   ├── motion/              # pure numpy, per-track 2D motion: config.py (MotionConfig, MeasurementConfig, PredictionConfig, CameraMount), measurement.py (box -> (x, z) + covariance, project_points), models.py (MotionModel/MotionFilter, registered "cv" and "ctrv" EKF), ego.py (EgoMotionProvider, registered "csv", PlanarTransform), estimator.py (MotionEstimator, build_motion)
│   ├── io/                  # video.py (ffmpeg pipes), source.py (FrameSource, FrameTransform, VideoSource), sinks.py (CSV/video/npz/metrics), quicktime.py (stdlib MOV metadata reader: moov/meta + trak/udta metas)
│   ├── camera.py            # IntrinsicsProvider: FixedIntrinsics, EstimatedIntrinsics, CalibratedIntrinsics; Calibration, load/save_calibration, Undistorter, check_camera_metadata, CameraSetup, build_camera
│   ├── calibration.py       # TargetSpec, CalibrationTarget (ChessboardTarget, CharucoTarget), detect_views, select_views, coverage, calibrate_views, focal_px_from_35mm
│   ├── evaluation/          # references.py (ReferenceSet), run_data.py (RunData, load_run), matching.py, stats.py (ErrorStats, summarize), evaluate.py, report.py
│   ├── tools/               # calibrate.py (det-depth-distance-calibrate), evaluate.py (det-depth-distance-evaluate)
│   ├── web/                 # web backend (extra `web`, torch-free): paths.py (WebPaths, safe_join), stages.py (stage list from core.registry), schema.py (options_schema, module_catalog, validate_config), app.py (create_app), jobs.py (JobQueue), worker.py, history.py (scan_runs, load_run, job_data), compare.py (run_summary, distance_series, config_diff, compare_runs, list_artifacts, effective_config), gpu.py (free_memory_mib), library.py (Library, VideoRecord, probe, .h264 remux, background preview), uploads.py (streaming multipart reader, bounded headers/parts/fields), security.py (Host/Origin/body-size middlewares, path policy, weights-root overlap check, served-file headers), models.py (model status rows, PrepareQueue), server.py (det-depth-distance-web: --allowed-host, --dev-origin, --asset-root, --weights-root); app.py also serves web/dist at / with an SPA fallback
│   ├── config.py            # RunConfig + sections, load_config, config_from_mapping, apply_overrides (--set); registers the built-in pipeline stages
│   ├── distance.py          # pure numpy: box -> map pixels -> robust stats -> distance
│   ├── temporal.py          # pure numpy: SmoothingConfig, distance_sigma, white_accel_cov, KalmanFilter1D / EmaFilter / MedianFilter (FILTERS), TrackSmoother
│   ├── render.py            # PIL boxes/labels/HUD (colour by class or source), predicted trajectories, top-view (BEV) inset, depth panel + legend
│   ├── runtime.py           # DeviceMonitor (torch, lazy), package versions, git commit
│   ├── cli.py               # --config/--set/--list-models/--prepare; composition root
│   └── __main__.py          # python -m det_depth_distance
├── tests/                   # pytest without GPU; tests/smoke/ (auto-marked "smoke", end-to-end); tests/gpu/ (auto-marked "gpu") runs on a machine with a GPU
├── utils/gpu_pending.sh     # pending GPU steps of the tasks, in order, with logs
├── docs/                    # task files (docs/task_DDDD.md) + template; refs/ (diagrams, calibration and field-reference protocol); web_extending.md (adding models/stages/views, incl. model requirements); model_environments.md (model recipe, weights root and locations, readiness states, Prepare in CLI/web, packages layer, the five dependency-conflict resolution paths: only detection is implemented)
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
- `Trajectory`: predicted positions of one track after the frame time — `t` (K,) s, `xz` (K, 2) m and `cov` (K, 2, 2) in the current camera frame, plus the reference point's height `y`.
- `MotionEstimate`: 2D motion state of one track at one frame on the camera plane (x right, z forward): model name, position and stds, velocity `vx`/`vz` (d/dt of the camera-frame position), speed and std, heading (deg; 0 = moving away, +90 = right; NaN below the model's min speed; with ego-motion the heading and yaw rate are those of the motion over ground, in the current camera axes, not `atan2(vx, vz)`) and std, yaw rate (deg/s, + = turning right; NaN without a turn-rate state), `abs_speed` over ground (NaN without ego-motion), `nis` (NaN without a measurement), `n_updates`, `restarted`, `trajectory`.
- `FrameResult`: frame, instances, per-instance `BoxDepthStats`, depth, timings, optional `temporal` (with smoothing: one `TrackEstimate | None` per instance), optional `motion` (with the motion stage: one `MotionEstimate | None` per instance), `semantic` (segmenter name -> `SemanticMap`, config order; `{}` without segmenters; maps are never merged), optional `box_stats` (only with region "both": box statistics where `stats` came from the mask, None elsewhere), optional `FrameAgreement` (per detector `instances`/`duplicates` after in-detector NMS, and one `PairAgreement(a, b, matched, only_a, only_b, iou_sum)` per detector pair).
- `LabelMap` (core/labels.py): name, raw -> canonical `mapping`, `unmapped` policy (`keep`/`drop`), optional declared `vocabulary`; loaded from `configs/labels/*.yaml`, checked against the model's class names and identified vocabulary at load.
- `DistanceConfig` / `BoxDepthStats` (distance.py): parameters (incl. `mask_erode`, depth-map pixels) and per-region statistics; `mask_px` / `mask_erode` are set (and `used_mask` true) when the region was the instance mask. `Instances.masks` are filled by the Seg adapters: RF-DETR Seg (`rfdetr` `variant: seg-*`, COCO) and YOLO26-seg (`*-seg.pt`, COCO; `*-objv1-seg.pt`, expected Objects365 — not yet verified on the GPU host).
- `MotionConfig` (motion/config.py): `models` (name -> spec `{type, **options}`, registry family "motion"), `classes` (canonical label -> model name), `default` (model or null), `reset_after_s`, `nis_gate`, `measurement` (`reference_point`, `std_scale`, `rel_floor`, `pixel_std`), `prediction` (`horizon_s`, `step_s`, `n_sigma`), `ego_motion` (spec, registry family "ego_motion", or null = camera frame), `camera_mount` (`x_m`, `y_m`, `yaw_deg`).
- `SmoothingConfig` (temporal.py): `filter` (kalman/ema/median), `accel_std`, `std_scale`, `rel_floor`, `init_speed_std`, `tau_s`, `window_s`, `reset_after_s` (times in seconds).
- `Calibration` (camera.py): one recording mode's calibration: `image_size` (W, H), K as `CameraIntrinsics` in the continuous convention (OpenCV cx/cy + 0.5, converted only in `from_opencv`/`camera_matrix`), OpenCV-order `dist`, `rms_px`, `std`, recording `camera` metadata, `meta`; `scaled_to` (same aspect ratio only). Stored as an OpenCV `FileStorage` YAML/JSON (`image_width`, `image_height`, `camera_matrix`, `distortion_coefficients` + optional keys); `CalibrationConfig` (config.py) is the `camera.calibration` section (`path`, `undistort`, `alpha`, `check_lens`, `cross_check`).
- `Stage` (core/registry.py): UI-facing descriptor of a pipeline stage (`yaml_key`, `title`, `kind` or `config_cls`, `multiple`, `optional`, `order`, `general`, `group`: consecutive stages with the same group share one UI lane), registered with `register_stage` (built-ins at the bottom of `config.py`, lane order source, detectors, aggregation, refiners, depth, tracking + smoothing (group "Tracking & filter"), semantic, motion); options dataclass fields may carry `metadata={"ui": {label, unit, min, max, step, advanced, help}}`, read by `web.schema.options_schema`.
- `ReferenceSet` (evaluation/references.py): field references of one video (YAML via `from_mapping`): `video`, `instruments` (uncertainty_m, uncertainty_rel), `references` (`Reference`: id, frame or time range, `Selector` box/point/track_id + label, constant `distance_m` or `series`, instrument or uncertainty, `quantity` range/z, target, notes). `RunData` (run_data.py): a run's CSV columns as numpy arrays + metrics.json.
- `RunConfig` (config.py): the YAML run (source, detectors, depth, camera, aggregation, refiners, semantic, tracking (flat tracker spec or null), smoothing (`SmoothingConfig` or null), motion (`MotionConfig` or null), agreement, distance incl. `exclude_semantic` "<segmenter>.<class>" labels, render incl. `semantic_alpha`/`semantic_overlay`/`show_smoothed`/`show_speed` and, for motion, `show_trajectory`/`trajectory_alpha`/`show_motion_speed`/`bev_inset`/`bev_size`/`bev_range_m`/`bev_width_m`, output, device).

Per run: `video.mp4`, `detections.csv` (one row per box; task_0001 columns then `source`, `raw_label`, `n_sources`, `sources`, `source_scores`, `raw_labels` — the last three `|`-joined, best member per detector — then `used_mask`, `mask_px`, `mask_erode`, `box_distance_m`, then `mask_source`, `mask_score`, then `track_id`, `distance_smooth_m`, `distance_smooth_std_m`, `rel_speed_mps`, `rel_speed_std_mps` (empty without tracking/smoothing; `distance_m` stays raw), then the 16 `motion_*` columns (`MOTION_FIELDS`: model, x/z and stds, vx/vz, speed and std, heading and std, yaw rate and std, abs speed, NIS, n_updates; empty without motion); no semantic column; 61 columns), `trajectories.jsonl` (with motion: one JSON line per frame x instance with its estimate and predicted trajectory), `metrics.json` (`components` incl. `refiners`/`semantic`/`tracker`, `pipeline` incl. `agreement`, `region`, `mask_erode`, `exclude_semantic`, `smoothing` and `motion` policy (motion models, classes, measurement, prediction, ego-motion with `clamped_frames`), focal, timings incl. `det.<name>`, `agree`, `aggregate`, `refine(.<name>)`, `track`, `semantic(.<name>)`, `smooth`, `motion`, `tracks` (with a tracker: per-track summary — label mode and changes, sources, span, coverage, gaps, distance range, raw/smoothed jitter, relative-speed percentiles and, with motion, a per-track `motion` entry: model, speed percentiles, mean NIS, restarts; tracks shorter than 1 s only counted), `detections_by_source`, `detections_by_n_sources`, `distance_regions`, `frame_intrinsics` (frame K given to the depth model or null), `camera` (`{}` without calibration; calibration, rescaling, lens check, undistortion, focal cross-check), `video.transform`, `mask_sources`, `refined` (targets/filled per refiner), `semantic` (with segmenters: per-map class fractions and excluded fraction, plus the excluded fraction of the union), `region_comparison` (mask minus box, with region "both"), `agreement` summary with >= 2 detectors, versions incl. `transformers`/`timm`, resolved `config`), optional `depth.npz` (plus `semantic_labels_<name>` on the depth-map grid and `semantic_classes_<name>` per segmenter).

## 5. Architecture Patterns

- Two environments from one manifest: a light local env for developing and
  unit-testing, and the heavy `infer` extra for the GPU. Framework
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
- Motion (task_0009, `motion/`, pure numpy): optional stage after
  smoothing; needs tracking (`ConfigError` otherwise). `build_motion` (the
  CLI's composition root) builds every named model through the registry
  family "motion" (`MotionModel`, stateless, like `Aggregator`; `cv`
  constant velocity, `ctrv` EKF with a cv warm-up of `warmup_s` because at
  v ~ 0 the CTRV Jacobian cannot see lateral motion) and the optional
  ego-motion provider (family "ego_motion", `csv`: t, x, y, yaw on the video
  clock, linear interpolation, edge pose with zero velocity and one warning
  outside the data). `MotionEstimator` keeps one `MotionFilter` per track id
  (the `TrackSmoother` idiom): model fixed at the track's birth from its
  canonical label (`classes`, then `default`), the box reference point
  back-projected with the depth-map intrinsics into a position measurement
  (x, z) with covariance (`distance_sigma` + pixel std), Joseph-form update,
  restart when the NIS exceeds `nis_gate`, prediction-only frames without a
  distance, state dropped after `reset_after_s`. With ego-motion the filter
  runs in the world (REP-103) and the outputs go back to the current camera
  frame (relative velocity = d/dt of the camera-frame position, abs speed
  over ground). `rollout` predicts the trajectory on a copy of the filter.
  Without ego-motion everything is relative to the camera (own turns read as
  object motion), so the example uses cv for every class and a ctrv model
  without ego-motion warns. Sinks draw the trajectory (projected with the
  depth-map intrinsics and `MapGeometry`), an optional BEV inset, and write
  `trajectories.jsonl`; `MetricsSink` re-reads `Pipeline.describe()` at close
  for counters only final after the last frame.
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
  `local_files_only=True` from `hf_cache` (default `<weights root>/hf`);
  nothing is downloaded at run time.
- Requirements (task_0021): every adapter implements `requirements(options)`
  (torch-free, from validated options): DA3/HF refiners/SegFormer ->
  `HuggingFaceAsset` in `<weights root>/hf`; RF-DETR -> `UrlAsset`
  `weights/rfdetr/<file>` (loaded with `pretrain_weights=`); YOLO and
  Ultralytics SAM -> `UrlAsset` for `ultralytics/assets` v8.4.0 names, else
  `UserProvided`; TwinLiteNet+ -> `CodeRepo` + `UserProvided` weights; packages
  = import names, `extra="infer"`. Explicit paths outside the weights root are
  not declared (the adapter checks them at load). `cli.config_requirements()`
  adds the `camera.estimate` focal model.
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
- Camera (task_0007): `camera.build_camera(cfg.camera, VideoSource)` is the
  camera composition root called by the CLI before the Pipeline is built:
  without `camera.calibration` it returns the source unchanged and
  `build_intrinsics_provider(cfg)` (task_0006 behaviour); with it, it loads
  the file, rescales K to the video, compares model/lens/35 mm-eq. with the
  video's QuickTime metadata (`check_lens`), optionally wraps the source with
  an `Undistorter` (`FrameTransform` applied by `VideoSource` in `__iter__`
  and `sample`; K handed over is the undistorted one) and returns
  `CalibratedIntrinsics` (optionally cross-checked by `EstimatedIntrinsics`).
  `IntrinsicsResult.details` -> `SetupInfo.intrinsics_details` and
  `RunContext.camera` -> `metrics.json["camera"]`. Adapters and
  `distance.py` are unchanged: `MapGeometry.scale_intrinsics` already handles
  fx != fy and an off-centre principal point.
- Evaluation (task_0007, light env, no I/O in the core): `load_run` +
  `load_references` -> `match_references` (per processed frame, reference and
  source group: best IoU box, smallest box containing a point, or track id;
  range <-> z via `distance.ray_factors`; reference pixels undistorted for
  undistorted runs) -> `build_samples` -> `summarize` (`ErrorStats`, per
  reference by default because frames are correlated) -> `write_report`
  (eval.json, eval.md, optional matplotlib PNGs).
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

The pipeline has no network API; its interface is the CLI `det-depth-distance` /
`python -m det_depth_distance --config run.yaml [VIDEO] [--set key=value]...`
and `--list-models`; `--config run.yaml --prepare` builds the components
(nothing loaded, the video is not probed), downloads their declared weights and
pinned code under the weights root (`DDD_WEIGHTS_ROOT`, default `weights/`;
also the focal model of `camera.estimate`),
prints a status table and exits 0 only when every component is ready (1
otherwise, 2 for configuration errors or an unsafe weights root; a file only
the user can provide is listed as "provide this file: <hint>" with state
`missing`, not `failed`; an existing file of another size is never
replaced). A run without `--prepare` checks the
same requirements after every configuration error and before any model is
loaded (`cli.require_ready`): a model that is not ready is a ConfigError
(exit 2) listing each one with what is missing (root-relative) and naming
`--prepare` (or, when only Python packages are missing, the one-time base
stack command `uv sync --extra infer`, add `--extra web` for the server; for
a per-model extra, `--prepare --install-packages`); files at explicit paths
outside the weights root are not declared, so not checked. Packages layer
(task_0024, `core/provision/packages.py`): `--prepare` reports it (base stack
missing -> the command, never run; a per-model extra -> a `uv sync --dry-run`
naming every installed extra plus the new one, with `--locked --inexact`;
failure -> state `conflict` with the resolver's reason, project-relative, and
a pointer to `docs/model_environments.md`); `--prepare --install-packages`
installs a per-model extra after a fresh clean dry-run (same command without
`--dry-run`, output streamed). Installed extras are detected from
`pyproject.toml` `[project.optional-dependencies]` with `importlib.metadata`
plus the components' import names via `find_spec` (nothing imported); a
dry-run that would remove or replace a package is a conflict; `uv` missing,
a timeout or a cancel is plan `error` (state `failed`, never cached, never a
conflict; the message names `DDD_UV`); `DDD_UV` names the `uv` executable.
Dry-runs are single flight per cache key and run in their own process group
(killed on timeout/cancel); uv output is scrubbed of credentials
(`packages.scrub`). Plus two auxiliary console scripts:
`det-depth-distance-calibrate calibrate|board|inspect` and
`det-depth-distance-evaluate --refs REFS.yaml RUN [RUN ...]`. The YAML schema (sections source, detectors, depth,
camera, aggregation, refiners, semantic, tracking, smoothing, agreement, distance, render, output, device) is documented in the
README; the task_0001 flags were replaced without aliases.

Optional web backend (`det-depth-distance-web`, extra `web`, binds
127.0.0.1 by default, no auth; task_0013 is its first slice):
`GET /api/catalog` (stages from `core.registry.stages()`, the options schema
of every registered adapter by family and of every config section, with
`catalog_version` and the dataclass-field `ui` metadata as `hint`),
`POST /api/config/validate` (`{config}` -> 200 `{errors: [{path, message}]}`),
`POST /api/config/parse` (`{yaml}` -> `{config, extras}`; unknown top-level
keys are kept as extras) and `POST /api/config/dump` (`{config, extras?}` ->
`{yaml}`). Models and stages are never listed in `web/`: a new adapter
(`@register`) or stage (`core.registry.register_stage`) appears by itself
(`docs/web_extending.md`).

Runs (task_0014): `POST /api/runs` (`{config, video_id?}` -> `{job_id}`; 422
`{errors}` when invalid), `GET /api/jobs/{id}` (status queued|running|done|
failed|cancelled, position, error, run_id, frames, log_tail), `GET
/api/jobs/{id}/events` (SSE; events queued, running, started, frame, done,
error, end), `DELETE /api/jobs/{id}` (cancel), `GET /api/runs`, `GET
/api/runs/{id}` and `GET /api/runs/{id}/{video|preview|config|detections}`,
`GET /api/gpu` (`{free_mib}`). One `JobQueue` runner starts one worker
subprocess (`python -m det_depth_distance.web.worker CONFIG.yaml`, JSON
lines on stdout) at a time; the server never imports torch. The worker
writes `config.yaml`, `web_run.json` (`status`, `video_id`, `started_at`) and
`preview.jpg` into the run folder; history (`web/history.py`) scans
`output.out_root`, tolerating old folders. `cli.run(cfg, on_frame=None)` and
`VideoSink(on_frame=...)` expose the composed frames.

Video library and Process (task_0016): `GET /api/videos`, `GET
/api/videos/roots`, `POST /api/videos` (multipart `file` + optional `meta`,
streamed to disk by `web/uploads.py`, never held in memory), `POST
/api/videos/import` (`{path}`: relative paths start at the server's cwd,
the repo root; must resolve inside a `--data-root`, default `data`; 403
otherwise; referenced, not copied; size/mtime recorded, and a moved, deleted
or modified original makes the record `missing: true`, which a re-import or
re-upload of the same content replaces), `GET
/api/videos/{id}` and `GET /api/videos/{id}/{file|preview|thumb}`.
`web/library.py`: one folder per video under `--library-root` (default
`_testbed/library`), id = first 12 hex of the sha256 (dedupe), probed with
ffprobe (size after rotation, avg fps, duration) plus first/last-frame decode
checks (non-video or truncated -> `InvalidVideo`, 422, nothing left behind:
entries are assembled in `.incoming/` and renamed into place);
`.h264` + `.meta.json` remuxed to VFR `.mkv` with the `SensorTimestamp`s
(PyAV; ffmpeg CLI fallback at the mean rate); `preview.mp4` (H.264,
longest side <= 1280, source timestamps) encoded in one background thread,
`preview: null` until ready. `GET /api/videos/roots` -> `{roots, cwd}`.
`POST /api/runs` rejects (422) a `video_id` not in the library or `missing`,
and with a `video_id` always sets `source.video` to the record's original. Front-end: a video
deck above the editor (drop anywhere / browse / server path / library,
player looping the segment, two-handle segment bar with start/length/stride
and a live estimate at 2 fps, Process and Process entire video, a GPU
confirmation below 4000 MiB free, a job panel with queue position, progress,
live `preview.jpg`, cancel, failure log tail) and the runs view (below) that a finished job opens.

Hardening (task_0018, `web/security.py`): `TrustedHostMiddleware` answers
only loopback names, the bound `--host` and `--allowed-host` values (400
otherwise); unsafe methods (POST/PUT/PATCH/DELETE) with a foreign `Origin`
(or `Sec-Fetch-Site: cross-site`) get 403 unless the origin is a
`--dev-origin`; JSON bodies of unsafe methods are capped at 1 MiB (413).
`POST /api/runs` without `video_id` requires `source.video` to resolve inside
a `--data-root` or to be a library original (422, same message whether it
exists or not); every other `Path`-typed field of the config (sections and
adapter options, nested specs included, discovered by
`schema.config_paths` from the annotations) must not resolve inside the
library or `out_root`, and with `--asset-root` (repeatable) must resolve
inside one (a non-loopback `--host` without it prints a startup warning).
Uploads: 413 over 8 GiB (Content-Length checked up front) or 64 MiB `meta`,
400 for a part header > 8 KiB, > 16 headers or > 8 parts, 429 beyond 2
concurrent uploads, 507 without free disk; stored suffix allow-listed
(`.mp4 .m4v .mov .mkv .webm .avi .ts .h264 .264`, else `.bin`). Library
ffprobe/ffmpeg/PyAV use `-protocol_whitelist file,pipe` and `-format_whitelist
mov,matroska,avi,mpegts,h264,hevc`; the preview encode times out after
max(600 s, 10x duration). Served files carry an explicit type, `nosniff` and
`CSP: sandbox`, and are attachments outside video/png/jpeg/webp/gif/csv/
text/json/ndjson/yaml. `/api/config/parse` refuses YAML expanding to > 100k
nodes or with recursive aliases (422). Run ids starting with `.` do not
resolve. Shutdown: `uvicorn.run(..., timeout_graceful_shutdown=3)`;
`JobQueue.aclose()` ends unfinished jobs `cancelled` ("the server shut
down") and a new queue settles leftover `running`/`queued` records
(`failed`/`cancelled`, "interrupted"). `GET /api/jobs` (unfinished + 20
latest finished, newest first; job dicts also carry `created`, `video_id`,
`video`), SSE events carry `id:` and honour `Last-Event-ID`; the front-end
re-subscribes after a reload and settles a 404 job. History titles runs of
library videos with the library name. `JOBS_DIR`/`VIDEO_ID_ENV` live in
`web/paths.py`; `WebPaths.asset_roots`.

Model readiness (task_0022, `web/models.py`): `POST /api/models/status`
(`{config}` -> `{models: [{stage, index, type, name, state, layers, missing,
size_bytes, message}], total_size_bytes}`; 422 `{errors}` for an invalid
config) lists every model the config would load exactly like `--prepare`
(`cli.config_requirements`; `stage`/`index` from the config path, e.g.
`detectors`/1, `depth`/null, `camera.estimate`/null), checked on disk in the
thread pool without importing torch; the model a Prepare is working on reads
`preparing`; `total_size_bytes` sums the declared sizes of models not ready.
`POST /api/models/prepare` (`{config, models: "all" | [{stage, index}]}` ->
job view `{id, status queued|running|done|failed|cancelled, position, error,
created, cancel_requested, models: [{stage, index, type, name, state, item,
done, total, layers, missing, size_bytes, message}]}`; 422 for an invalid
config, a path problem, an unknown key or any extra body field: a client never
supplies a URL or a path), `GET /api/models/prepare/{id}`, `GET
/api/models/prepare/{id}/events` (SSE with `id:`/`Last-Event-ID`; events
queued, running, model (preparing, then the final state), progress `{stage,
index, item, done, total}` at most every 0.2 s per item, cancelling, end) and
`DELETE /api/models/prepare/{id}` (cancel; a running job ends `cancelled`
once the engine notices between chunks, its model stays not ready). One
`PrepareQueue` runner, independent from the GPU `JobQueue`, awaits
`core.provision.prepare` in a worker thread (progress marshalled with
`call_soon_threadsafe`); nothing is persisted (the 20 latest finished jobs are
kept in memory). Messages name the weights root `<weights root>`, never its
absolute path (`core.requirements.redact_root`: absolute and resolved forms
only, at a path boundary — backtick, `<`, `>`, `{`, `}` included — one pass;
422 replies and job exception texts are also scrubbed of credentials,
`web.models.client_text`). Known sources only (fix round 1, fail closed since
task_0026): `Asset.known` and `CodeRepo.known` default to False and the
adapter's code sets them only for the canonical target — `ultralytics/<release
name>`, the `hf` cache for repos in `HF_FACTS`, the RF-DETR table, the pinned
TwinLiteNet+ clone — so a Hugging Face repo typed in `depth.model`,
`refiners[i].model` or `semantic[i].model` that is not in the table, a release
file name at a client-chosen `weights` path, or a known repo in another
`hf_cache` folder is `known=False`. From the web such a model reads `missing`
with the message "unknown source: prepare it from the CLI (--prepare)"; naming
it in `/api/models/prepare` is a 422; `models: "all"` prepares the other
models and lists it in the job's `skipped` (`[{stage, index, type, name,
message}]`; 422 only when every model is skipped). The queue runs
`engine.prepare(..., allow_unknown=False)`, which re-checks `known` when the
job runs (the CLI passes `allow_unknown=True` and downloads any repo). A row
whose only missing files are user-provided (`Asset.downloadable` False, code
layer ready) has `reason: "user_provided"`, is not preparable and is skipped
by "all" the same way (its message is the manual-download hint); the engine
reports such a file as "provide this file: ..." without failing the model,
and a weights job that fetched everything else ends `done`. An existing file
whose size or sha256 differs from the declaration is `failed` ("exists with a
different size at <target>; move or delete it, or point the config at
another file") and is never downloaded over. URL downloads stop when the
body exceeds the expected size and check free disk space first. `POST /api/runs` and rerun answer 409 `{models_not_ready:
[status rows]}` when a model is not `ready` (after the 422 checks).
`--weights-root` (else `DDD_WEIGHTS_ROOT`, else `weights/`) is set with
`set_weights_root` and exported to every run worker as an absolute, resolved
`DDD_WEIGHTS_ROOT`; it may not be `/` or the home folder, contain the project,
the active Python environment, or be, contain or lie inside the library or
`--out-root` (start-up error; `create_app` raises ValueError;
`core.requirements.weights_root_problem`, also checked by the CLI with the
config's `output.out_root`). With
`--asset-root`, the weights root is an implicit asset root, and a `Path`
option with `core.options.PATH_RESOLVER` field metadata (YOLO / Ultralytics
SAM `weights` -> `resolve_weights`) is checked as its adapter resolves it.

Status rows also carry (task_0023 review): `source` (each asset kind's
`Asset.describe_source()`, default the kind name; HF "Hugging Face · <repo>",
URL host, "git · <host>/<path>" for code, the user-provided hint), `license`,
`target` (weights-root-relative), all "; "-joined and redacted, plus `reason`
(`unknown_source` | `user_provided` | `packages_only` | null) and
`preparable` (state `missing` and no reason). `GET /api/models/prepare` lists the queue's jobs (newest
first, `PrepareQueue.recent()`), so a reloaded page re-attaches.

Packages in the web (task_0024): rows carry `packages` (null when present,
else `{action: base|install|conflict|invalid, extra, command, message}` from
`packages.plan_packages`; the dry-run is cached per extra, detected extras
and pyproject/lock mtimes); a failed dry-run makes the row `conflict` (not
preparable; a run is refused with 409). `POST /api/models/prepare` with
`packages: true` and named models (not "all"; 422 for the base stack or a
model without a per-model extra; the route computes the server-side plans
with config-wide import names first, `PrepareQueue.install_plans`, and each
must be `install`; 409 while a GPU run is queued or running) queues a job of
`kind: "packages"` in the same PrepareQueue (`POST /api/runs` answers 409
`{detail, packages_installing: true}` while one is queued or running; the
job refuses a fresh plan that differs from the confirmed one): `engine.install_packages` runs a fresh dry-run, then `uv
sync` in its own process group (cancel kills the group), streaming `output`
events `{stage, index, line}` (redacted, at most 2000 per job); the model view
gains `installed`; afterwards the status body's `restart_required` is true.
The server never installs the base stack: a `base` row shows
`uv sync --extra infer --extra web` with a copy button. UI: **Install
packages** (only for a clean dry-run) opens a Dialog stating the command and
that nothing is removed; the inspector shows the `uv` output while it runs and
afterwards (section note and banner) that new runs use the packages (each run
starts a fresh worker; a server restart is only for an upgraded package the
server had already imported).

Readiness in the UI (task_0023, `web/src/models/`): `useModelStatus(config,
{api})` posts the edited pipeline to `/api/models/status` (debounced like the
validation; a sequence bumped when a request is scheduled drops stale
answers; 422 -> `invalid`, other errors -> `unavailable`, neither blocks
Process), follows the Prepare jobs it starts over SSE (`model` / `progress` /
`end` events -> per-model byte progress) and keeps the outcome of the last
attempt per model (a failure stays `failed` with its reason after the status
refresh, until retried or ready; a changed type/name drops it); on mount
(and while the server reports `preparing` for a job it does not follow) it
lists the queue and follows every unfinished job; progress never rewinds on a
replayed stream, a model that finished stays ready until a later status
answer, and every overlay carries the module's type + name (rows are also
checked against the edited config), so removing or reordering modules never
moves a state to another card. The server's `reason`/`preparable`/`target`
win; message-prefix heuristics remain for older rows. Context
`ReadinessProvider`/`useReadiness` (null outside it: cards render as before).
Cards (`ModuleCard` `readiness` prop, `cardReadiness`): `ready` solid;
`missing`, `conflict` and any unknown state -> `card--notready` (yellow dashed
outline + alert badge `role="img"` at the top-right); `preparing`/`queued` ->
`card--preparing` with a `role="progressbar"` (`aria-valuenow` in %); `failed`
-> red dashed + badge. Rows map to slots by `{stage, index}`; a single-slot
stage also shows rows of its sub-paths (`camera.estimate` in the general
`camera` tab). `ReadinessBanner` ("N models need preparing · X to download", N and X
counting only the models Prepare all would fetch; licences in the sub line; Prepare all opens
a confirmation Dialog listing only the downloads — name, source, size,
licence — with the total and, apart, the files the user must provide; only
Download sends `models: "all"`; model list; overall progress + Cancel; "Check again" when
unavailable). Inspector "Weights & requirements" (`ModelRequirements`):
layers, missing items, message, size, source/licence/target when a row
carries them (else locations parsed from `<weights root>/...` in the
message), **Prepare model** / **Retry** for preparable rows, the copyable
`det-depth-distance --config <file> --prepare` for an unknown source (message
prefix `unknown source`), a "provide it by hand" note for `user_provided`, the packages plan (task_0024: base command, Install
packages, conflict reason) or, from older servers, a note when only
`package:` items are missing, a conflict note, and the module's unset path options as the location override.
Process is disabled with "N models are not ready — Prepare them first." while
the last status has a row not `ready`; `submitRun` throws `ModelsNotReady` on
a 409 and `useProcess` shows a "Models not ready" dialog listing them (with
Prepare all); rerun's 409 becomes a readable notice. No model, stage or source
name is written in `web/src`.

Results and comparison (task_0017, `web/compare.py`): `GET
/api/runs/{id}/summary` (`{summary, metrics}`), `GET /api/compare?a=ID&b=ID`
(`{a, b, metrics, summary_a, summary_b, deltas: {key: {abs, rel}} (b - a, only
metrics both runs have), config_diff: [{path, a, b, change, server_set}] |
null, config_origin}`), `GET /api/runs/{id}/series?track=N` (`{track, found,
tracks: [{id, rows, t0, t1, class_name}], t, raw, smoothed|null, video_t,
points, downsampled}`; default track = the longest; min/max decimation to
<= 1500 points; `video_t` = (frame - round(start*fps))/fps, the time in
`video.mp4`), `GET /api/runs/{id}/artifacts` (every file of the run folder:
`{name, size, media_type, modified, role, url}`, hidden/temporary files left
out) and `GET /api/runs/{id}/files/{path}`, `GET /api/runs/{id}/metrics`,
`POST /api/runs/{id}/reopen` (`{config, extras, origin, file_name}` from
`config.yaml`, else the `config` recorded in metrics.json, else the job
record) and `POST /api/runs/{id}/rerun` (`{job_id}`: same config, same
`video_id`; without one, `source.video` only inside a `--data-root`, else
409). Summary (`SUMMARY_METRICS`, each with label, unit and `better`
lower|higher|null): median distance, jitter (median over tracks of the
median |Δ distance_m| between consecutive processed frames), smoothed
jitter (same on `distance_smooth_m`), throughput (`end_to_end_fps`), frames,
boxes, tracks; detections.csv is read by `evaluation.run_data.read_columns`
(columns frame, t, track_id, distance_m, distance_smooth_m, class_name;
cached per file version); missing files/columns give null plus `notes`.
`job-<id>` entries answer 404 "has no files" on every file route.
`RunRecord` gained `video_id` and `log_tail`; `has_config` also counts the
config in metrics.json. Front-end runs view (`#/runs/<id>/vs/<id>`, ids
URI-encoded): filterable history (selected run in teal, comparison in
violet, per-entry compare button), annotated video, KPI cards with signed
deltas coloured good/bad by `better`, configuration diff (server-set
`output.tag`/`out_root` hidden behind a toggle), SVG distance chart (this
run smoothed over raw in grey, the comparison run dashed; hover readout,
keyboard cursor, click/Enter seeks the video, playhead follows it), the
discovered artifacts with inline views from `registerArtifactView(match,
view)` (built-ins: JSON tree, image, MP4) and `metrics.json` as a generic
collapsible tree; Reopen in editor and Rerun (the job joins the job
panel).

Front-end (task_0015, `web/`, English, dark theme only): a pipeline editor
built entirely from `/api/catalog`. Lanes follow `stages` (non-general) in
`order`, consecutive stages sharing a `group` merged into one lane; adapter stages show one card per spec and an "+ Add" menu of the
registered types, section stages one card; a right inspector renders the
selected spec/section from its schema (`Field.type` -> slider with hint
min/max, select, switch, text/path, list, nested group; `hint.advanced` in a
collapsed "Advanced" group; `map`/`any`/unknown types as a raw YAML editor);
"General settings" holds the `general` sections, top-level keys no stage owns
and the parse `extras`. State lives in a small store (`useEditor()`:
`addModule`, `removeModule` (rejected on single-slot stages), `disableStage`,
`changeType`, `setOption(path, value)`, `load(yaml)`, `toConfig()` (config +
extras), `dirty`, `diff()` against the last opened/saved YAML); validation is
debounced 400 ms and errors attach to fields by dotted path (spec-level
domain errors whose message starts with a field name are attached to that
field). `det-depth-distance-web --static-dir DIR` overrides the served build
(default `web/dist` of the checkout); missing files under `assets/` or with
an extension are 404, other paths fall back to `index.html`. Save YAML downloads `/api/config/dump` output and warns once that
comments/key order are lost. Extension registry: `registerStageView(yamlKey,
view)`, `registerFieldControl(match, control)`; a `catalog_version` above the
supported one shows a non-blocking banner. Process is disabled until task_0016.

## 8. Key Business Concepts

- **Metric depth**: per-pixel distance in metres along the optical axis (z).
- **Euclidean range**: straight-line camera-to-point distance; larger than z
  for off-axis points.
- **Focal length (px)**: camera intrinsic needed to convert DA3METRIC output
  to metres (distance is proportional to it); from a calibration file, a
  configured value, or estimated by an any-view DA3 model (the estimate
  varied ~8 % between two videos of the same lens). The 35 mm-equivalent
  focal in the metadata is diagnostic only.
- **Calibration / undistortion**: K and lens distortion measured with a
  chessboard or ChArUco in the recording mode of the videos; undistortion
  remaps every frame so all components see a pinhole camera.
- **Field reference / implied focal**: a distance measured on site (laser,
  ground marks, GNSS RTK) for one object over a frame range, with its
  uncertainty; the implied focal is the focal that would cancel the median
  scale error of a DA3METRIC run.
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
- **Motion / trajectory**: an object's position on the camera plane
  (lateral x, forward z) from its box's ground-contact point, its velocity,
  heading and turn rate, and its predicted positions over the next seconds
  with their uncertainty, under a motion model chosen by class. Without
  ego-motion it is motion relative to the (possibly turning) camera.

## 9. Security Model

No secrets are read by the code. Hugging Face model downloads use the
user's existing token cache, outside this repository; the
gated `facebook/sam3` checkpoint additionally needs approved access and
`HF_TOKEN` there. Adapters never download at run time.

## 10. Development & Deployment

- Local: `uv sync`, `uv run pytest` (unit + smoke, GPU tests deselected), `uv run ruff check . && uv run ruff format --check .`
- Three test levels: unit (each module, fixed doubles); smoke (`-m smoke`, `tests/smoke/`, ~8 s): the real CLI entry point in process on a synthetic 320x180 grey-level video of known depth planes, with frame-reading test doubles (detector, box-prompt refiner, semantic segmenter) inserted into the registry by `monkeypatch` and the real `DA3Depth` with a fake network; a pairwise matrix of 9 configurations checks every output and every distance analytically, plus the all-off = task_0002 invariant and a calibrate -> run -> evaluate chain; `tests/run_outputs.py` checks any run folder's structure; gpu (`-m gpu`, on a machine with a GPU).
- GPU tests (adapter contracts + 60-frame regression against task_0001 + a 5-frame smoke of 10 example configs, one per real adapter family plus the light 4 GB example, which skips where its weights are missing): `uv run --extra infer pytest -m gpu -v`.
- `utils/gpu_pending.sh` (run on the GPU machine): every pending GPU step of tasks 0002-0008 in dependency order, a log per step and `summary.tsv` in `_testbed/outputs/gpu_pending_<stamp>/`; `list`, `run --dry-run|--from|--only|--skip-long`; downloads only with `--allow-downloads` plus a per-item confirmation (task_0004 downloads only printed).
- Weights (task_0021): everything under the weights root (`weights/`, git-ignored; `DDD_WEIGHTS_ROOT` / `--weights-root`), fetched by `--prepare`; nothing is downloaded at run time (DA3 included). Layout: `hf/` (Hugging Face cache: DA3, SAM family, SegFormer; an existing cache keeps working), `rfdetr/<library file name>` (Apache-2.0, ~135–405 MB), `ultralytics/<name>.pt` (YOLO26 COCO/Objects365 and SAM/MobileSAM from the `ultralytics/assets` v8.4.0 release; any other `.pt` is user-provided), `_code/TwinLiteNetPlus` (pinned clone; `pretrained/<variant>.pth` user-provided). Existing `~/.roboflow`, `~/dev/...` files are untouched and usable as explicit YAML paths. Sources, sizes and licences: `docs/task_0021.md`.
- Light example (task_0010, `configs/yolo_light_da3.yaml`, for the laptop's RTX A1000 4 GB): YOLO26s-seg, SegFormer-B0 `nvidia/segformer-b0-finetuned-cityscapes-640-1280`, the DA3 checkpoints and TwinLiteNet+ `small.pth`, all under the weights root. Estimated peak ~2.4 GB allocated (main phase) and ~3 GB in the focal phase, which is not recorded in `metrics.json`; VRAM on the 4 GB GPU still to be measured. GPU tests skip when a weight is missing.
- GPU machine: `uv sync --extra infer` (in the notebook, with `UV_PROJECT_ENVIRONMENT=_testbed/venv-infer`).
- No deployment target beyond running the CLI locally.

## 11. Conventions Summary

- Code, docstrings, comments in English; AGENTS.md, CLAUDE.md, README in pt-BR
  (existing); new docs and task files in English; CHANGELOG pt-BR ASCII for old entries.
- Model requirements: each adapter declares `requirements(options)` (assets, pinned code, packages + extra); defaults under the weights root, YAML paths override; a new source kind is a registered `Asset` subclass + `register_source` (no web edits). Reference and conflict paths: `docs/model_environments.md`.
- Ruff defaults; snake_case functions/modules, PascalCase classes.
- Conventional Commits with an English subject (pt-BR in older commits); scopes: core, adapters, io, distance, camera, render, cli, eval, configs, tests, utils, deps, docs.
- Never commit to `main`; work on `feat/`/`fix/` branches.

## 12. Task Documentation & Lifecycle (docs/)

This project uses task-driven development: every implementation has a
`docs/task_DDDD.md` file (Description, Scope, Planning, Impact Analysis,
Verification, Dependencies) created before coding. On completion the
title is marked `(COMPLETED)` and the file gains `## Implementation
details` and `## Test notebook` sections in the same commit as the
implementation. See `AGENTS.md` for the mandatory rules. New documentation and task files are written in English (the task template is in English; existing pt-BR documents are not migrated).

## 13. What This Project Does NOT Have

- No TensorRT/ONNX path, no batching, no real-time optimisation.
- No fusion of masks across models (a fused instance keeps one member's mask) and no fusion of semantic maps (each segmenter keeps its own map); no FastSAM (not one-mask-per-box), no native `sam3` package (the SAM 3 tracker is used through `transformers`); no semantic CSV columns or video colour legend; no appearance-based (ReID) or ultralytics tracker, no 3D association (the tracker still matches by image IoU) and no per-track colour in the video. Motion: no IMM (one model per track), no learned trajectory predictor, no map/lane constraints and no visual ego-motion (only a CSV provider). Instance masks are never saved to disk (semantic labels are, on the depth-map grid, with `save_depth`).
- No score calibration across detectors (fusion weights are fixed per detector); no new dependency for fusion (no `ensemble-boxes`, `torchvision.ops` or `scipy` in the core).
- No interactive reference annotation tool (reference YAML is edited by hand), no per-frame intrinsics for stabilisation, no focal from the 35 mm equivalent, no `opencv-contrib`; no `pandas` in evaluation.
- No CI, no Docker, no git remote; the web UI has no auth, no run deletion, no cross-run track matching (comparisons pair equal track ids, default = the longest shared id with overlapping [t0, t1], and a note says equal ids may be different objects), no light theme or translations, no video deletion from the library and no resumable uploads.
