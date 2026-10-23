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
| `eval` extra (optional) | `matplotlib>=3.8` (locked 3.11.2): plots of `det-depth-distance-evaluate --plot` only; imported lazily |
| `infer` extra (GPU hosts: `gpu-host`; optionally the laptop's 4 GB GPU in an isolated env, `UV_PROJECT_ENVIRONMENT=_testbed/venv-infer`) | torch 2.8.0 + torchvision 0.23.0 + xformers 0.0.32.post2 from the PyTorch cu128 index; `rfdetr` @ git 1f2af2ea; `depth-anything-3` @ git 3d835ec; `addict` (imported by depth-anything-3 but not declared by it); `ultralytics==8.4.175` (YOLO26 and the optional `ultralytics_sam` refiner; **AGPL-3.0** — acceptable for this internal research prototype, optional extra); `transformers>=5.1,<6` (locked 5.19.0; previously only transitive via rfdetr; SAM-family refiners and SegFormer); `timm>=1.0.17` (locked 1.0.30; only for EdgeTAM's TimmWrapper backbone) |
| External code (GPU hosts) | TwinLiteNet+ clone `~/dev/TwinLiteNetPlus` @ 90f1b86 (MIT; not a package, imported with a temporary `sys.path` entry; weights in its `pretrained/`) |
| Model licences (task_0005) | Apache-2.0: SAM, SAM 2.1, SAM-HQ, EdgeTAM, MobileSAM, `timm`; "SAM License", gated: SAM 3 (`sam3_tracker`, optional, never a default); `other` (NVIDIA terms, read before non-academic use): SegFormer checkpoints; MIT: TwinLiteNet+; AGPL-3.0: ultralytics |
| Dev group | pytest, ruff |
| Video I/O | system ffmpeg/ffprobe |
| Inference host | `gpu-host`: NVIDIA RTX PRO 6000 Blackwell 96 GB (sm_120), no sudo |
| CI / Docker | none |

## 3. Project Structure

```
det-depth-distance/
├── pyproject.toml           # uv manifest; "infer" extra holds the GPU stack, "eval" extra matplotlib; 3 console scripts; pytest "gpu" marker
├── uv.lock
├── configs/                 # example run YAMLs: rfdetr_da3, yolo_o365_da3, multi_detector(_fused), *_seg_da3, rfdetr_sam2_da3, yolo_o365_sam2_da3, rfdetr_semantic_da3, rfdetr_lanes_scene_da3, rfdetr_da3_track, rfdetr_da3_calibrated, yolo_light_da3, rfdetr_da3_motion, rfdetr_sam2_lanes_motion_da3
│   ├── labels/              # versioned label maps: coco_driving, objects365_driving
│   └── references/          # example_references.yaml (fictitious field-reference template)
├── src/det_depth_distance/
│   ├── core/                # contracts + orchestration, no ML framework imports
│   │   ├── types.py         # Frame, Instances, DepthResult, FrameResult, RefinedMasks, SemanticMap, TrackEstimate, Trajectory, MotionEstimate, MapGeometry, CameraIntrinsics, ContractError
│   │   ├── options.py       # ConfigError, from_mapping (typed YAML -> dataclass, dotted error paths)
│   │   ├── components.py    # LoadContext, Component, DetectorOptions, Detector, DepthEstimator, RefinerOptions, MaskRefiner, SemanticSegmenter, TrackerOptions, Tracker (ABCs)
│   │   ├── registry.py      # @register, build, available, load_adapters (ADAPTER_MODULES)
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
│   ├── adapters/_hf_hub.py  # offline from_pretrained + exact download command
│   ├── adapters/_ultralytics_env.py  # YOLO_OFFLINE / YOLO_AUTOINSTALL / YOLO_CONFIG_DIR policy (YOLO and SAM)
│   ├── motion/              # pure numpy, per-track 2D motion: config.py (MotionConfig, MeasurementConfig, PredictionConfig, CameraMount), measurement.py (box -> (x, z) + covariance, project_points), models.py (MotionModel/MotionFilter, registered "cv" and "ctrv" EKF), ego.py (EgoMotionProvider, registered "csv", PlanarTransform), estimator.py (MotionEstimator, build_motion)
│   ├── io/                  # video.py (ffmpeg pipes), source.py (FrameSource, FrameTransform, VideoSource), sinks.py (CSV/video/npz/metrics), quicktime.py (stdlib MOV metadata reader: moov/meta + trak/udta metas)
│   ├── camera.py            # IntrinsicsProvider: FixedIntrinsics, EstimatedIntrinsics, CalibratedIntrinsics; Calibration, load/save_calibration, Undistorter, check_camera_metadata, CameraSetup, build_camera
│   ├── calibration.py       # TargetSpec, CalibrationTarget (ChessboardTarget, CharucoTarget), detect_views, select_views, coverage, calibrate_views, focal_px_from_35mm
│   ├── evaluation/          # references.py (ReferenceSet), run_data.py (RunData, load_run), matching.py, stats.py (ErrorStats, summarize), evaluate.py, report.py
│   ├── tools/               # calibrate.py (det-depth-distance-calibrate), evaluate.py (det-depth-distance-evaluate)
│   ├── config.py            # RunConfig + sections, load_config, apply_overrides (--set)
│   ├── distance.py          # pure numpy: box -> map pixels -> robust stats -> distance
│   ├── temporal.py          # pure numpy: SmoothingConfig, distance_sigma, white_accel_cov, KalmanFilter1D / EmaFilter / MedianFilter (FILTERS), TrackSmoother
│   ├── render.py            # PIL boxes/labels/HUD (colour by class or source), predicted trajectories, top-view (BEV) inset, depth panel + legend
│   ├── runtime.py           # DeviceMonitor (torch, lazy), package versions, git commit
│   ├── cli.py               # --config/--set/--list-models; composition root
│   └── __main__.py          # python -m det_depth_distance
├── tests/                   # pytest without GPU; tests/smoke/ (auto-marked "smoke", end-to-end); tests/gpu/ (auto-marked "gpu") runs on gpu-host
├── utils/sync_gpu_host.sh    # rsync push code / pull outputs with the GPU host
├── docs/                    # task files (docs/task_DDDD.md) + template; refs/ (diagrams, calibration and field-reference protocol)
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
- `ReferenceSet` (evaluation/references.py): field references of one video (YAML via `from_mapping`): `video`, `instruments` (uncertainty_m, uncertainty_rel), `references` (`Reference`: id, frame or time range, `Selector` box/point/track_id + label, constant `distance_m` or `series`, instrument or uncertainty, `quantity` range/z, target, notes). `RunData` (run_data.py): a run's CSV columns as numpy arrays + metrics.json.
- `RunConfig` (config.py): the YAML run (source, detectors, depth, camera, aggregation, refiners, semantic, tracking (flat tracker spec or null), smoothing (`SmoothingConfig` or null), motion (`MotionConfig` or null), agreement, distance incl. `exclude_semantic` "<segmenter>.<class>" labels, render incl. `semantic_alpha`/`semantic_overlay`/`show_smoothed`/`show_speed` and, for motion, `show_trajectory`/`trajectory_alpha`/`show_motion_speed`/`bev_inset`/`bev_size`/`bev_range_m`/`bev_width_m`, output, device).

Per run: `video.mp4`, `detections.csv` (one row per box; task_0001 columns then `source`, `raw_label`, `n_sources`, `sources`, `source_scores`, `raw_labels` — the last three `|`-joined, best member per detector — then `used_mask`, `mask_px`, `mask_erode`, `box_distance_m`, then `mask_source`, `mask_score`, then `track_id`, `distance_smooth_m`, `distance_smooth_std_m`, `rel_speed_mps`, `rel_speed_std_mps` (empty without tracking/smoothing; `distance_m` stays raw), then the 16 `motion_*` columns (`MOTION_FIELDS`: model, x/z and stds, vx/vz, speed and std, heading and std, yaw rate and std, abs speed, NIS, n_updates; empty without motion); no semantic column; 61 columns), `trajectories.jsonl` (with motion: one JSON line per frame x instance with its estimate and predicted trajectory), `metrics.json` (`components` incl. `refiners`/`semantic`/`tracker`, `pipeline` incl. `agreement`, `region`, `mask_erode`, `exclude_semantic`, `smoothing` and `motion` policy (motion models, classes, measurement, prediction, ego-motion with `clamped_frames`), focal, timings incl. `det.<name>`, `agree`, `aggregate`, `refine(.<name>)`, `track`, `semantic(.<name>)`, `smooth`, `motion`, `tracks` (with a tracker: per-track summary — label mode and changes, sources, span, coverage, gaps, distance range, raw/smoothed jitter, relative-speed percentiles and, with motion, a per-track `motion` entry: model, speed percentiles, mean NIS, restarts; tracks shorter than 1 s only counted), `detections_by_source`, `detections_by_n_sources`, `distance_regions`, `frame_intrinsics` (frame K given to the depth model or null), `camera` (`{}` without calibration; calibration, rescaling, lens check, undistortion, focal cross-check), `video.transform`, `mask_sources`, `refined` (targets/filled per refiner), `semantic` (with segmenters: per-map class fractions and excluded fraction, plus the excluded fraction of the union), `region_comparison` (mask minus box, with region "both"), `agreement` summary with >= 2 detectors, versions incl. `transformers`/`timm`, resolved `config`), optional `depth.npz` (plus `semantic_labels_<name>` on the depth-map grid and `semantic_classes_<name>` per segmenter).

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

No network API. The interface is the CLI `det-depth-distance` /
`python -m det_depth_distance --config run.yaml [VIDEO] [--set key=value]...`
and `--list-models`, plus two auxiliary console scripts:
`det-depth-distance-calibrate calibrate|board|inspect` and
`det-depth-distance-evaluate --refs REFS.yaml RUN [RUN ...]`. The YAML schema (sections source, detectors, depth,
camera, aggregation, refiners, semantic, tracking, smoothing, agreement, distance, render, output, device) is documented in the
README; the task_0001 flags were replaced without aliases.

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
user's existing token cache on the GPU host, outside this repository; the
gated `facebook/sam3` checkpoint additionally needs approved access and
`HF_TOKEN` there. Adapters never download at run time.

## 10. Development & Deployment

- Local: `uv sync`, `uv run pytest` (unit + smoke, GPU tests deselected), `uv run ruff check . && uv run ruff format --check .`
- Three test levels: unit (each module, fixed doubles); smoke (`-m smoke`, `tests/smoke/`, ~8 s): the real CLI entry point in process on a synthetic 320x180 grey-level video of known depth planes, with frame-reading test doubles (detector, box-prompt refiner, semantic segmenter) inserted into the registry by `monkeypatch` and the real `DA3Depth` with a fake network; a pairwise matrix of 9 configurations checks every output and every distance analytically, plus the all-off = task_0002 invariant and a calibrate -> run -> evaluate chain; `tests/run_outputs.py` checks any run folder's structure; gpu (`-m gpu`, on `gpu-host`).
- GPU tests (adapter contracts + 60-frame regression against task_0001 + a 5-frame smoke of 10 example configs, one per real adapter family plus the light 4 GB example, which skips where its weights are missing): `uv run --extra infer pytest -m gpu -v` on `gpu-host`.
- `utils/gpu_pending.sh` (run ON `gpu-host`): every pending GPU step of tasks 0002-0008 in dependency order, a log per step and `summary.tsv` in `_testbed/outputs/gpu_pending_<stamp>/`; `list`, `run --dry-run|--from|--only|--skip-long`; downloads only with `--allow-downloads` plus a per-item confirmation (task_0004 downloads only printed).
- Refiner and SegFormer weights: Hugging Face cache `weights/hf` in the repo (git-ignored and excluded from the rsync push), downloaded explicitly with `snapshot_download(..., allow_patterns=['*.json', 'model.safetensors'])` (SegFormer: `pytorch_model.bin`) — command in the README; MobileSAM `mobile_sam.pt` beside the YOLO weights; TwinLiteNet+ weights in its clone.
- Light example (task_0010, `configs/yolo_light_da3.yaml`, for the laptop's RTX A1000 4 GB): YOLO26s-seg in `~/dev/ultralytics/_testbed/weights`, SegFormer-B0 `nvidia/segformer-b0-finetuned-cityscapes-640-1280` and the DA3 checkpoints in `weights/hf`, TwinLiteNet+ `small.pth` in its clone. Estimated peak ~2.4 GB allocated (main phase) and ~3 GB in the focal phase, which is not recorded in `metrics.json`; VRAM on the 4 GB GPU still to be measured. DA3 downloads a missing checkpoint at run time (no `local_files_only`).
- Weights live outside the repo on the GPU host: RF-DETR (incl. Seg, ~135–155 MB each, downloaded and MD5-checked by `rfdetr` on first construction) in `~/.roboflow/models`; YOLO26 (`yolo26*.pt`, `*-objv1-150.pt`, `*-seg.pt`, `*-objv1-seg.pt`, from the `ultralytics/assets` v8.4.0 release, fetched manually — the adapter never downloads) in `~/dev/ultralytics/_testbed/weights`. GPU tests skip when a weight is missing.
- GPU host: `utils/sync_gpu_host.sh push`, then on `gpu-host`
  `~/.local/bin/uv sync --extra infer`; outputs fetched with
  `utils/sync_gpu_host.sh pull`.
- No deployment target beyond running the CLI on the GPU host.

## 11. Conventions Summary

- Code, docstrings, comments in English; task files, AGENTS.md, CLAUDE.md,
  README in pt-BR; CHANGELOG in pt-BR ASCII.
- Ruff defaults; snake_case functions/modules, PascalCase classes.
- Conventional Commits with pt-BR subject; scopes: core, adapters, io, distance, camera, render, cli, eval, configs, tests, utils, deps, docs.
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
- No fusion of masks across models (a fused instance keeps one member's mask) and no fusion of semantic maps (each segmenter keeps its own map); no FastSAM (not one-mask-per-box), no native `sam3` package (the SAM 3 tracker is used through `transformers`); no semantic CSV columns or video colour legend; no appearance-based (ReID) or ultralytics tracker, no 3D association (the tracker still matches by image IoU) and no per-track colour in the video. Motion: no IMM (one model per track), no learned trajectory predictor, no map/lane constraints and no visual ego-motion (only a CSV provider). Instance masks are never saved to disk (semantic labels are, on the depth-map grid, with `save_depth`).
- No score calibration across detectors (fusion weights are fixed per detector); no new dependency for fusion (no `ensemble-boxes`, `torchvision.ops` or `scipy` in the core).
- No interactive reference annotation tool (reference YAML is edited by hand), no per-frame intrinsics for stabilisation, no focal from the 35 mm equivalent, no `opencv-contrib`; no `pandas` in evaluation.
- No CI, no Docker, no git remote, no web UI or API.
