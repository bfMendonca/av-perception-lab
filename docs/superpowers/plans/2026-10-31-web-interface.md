# Web Interface Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A dark, English web UI to build a det-depth-distance pipeline visually, process a trimmed segment of an uploaded video, and browse/compare past runs.

**Architecture:** A FastAPI server (new package `det_depth_distance.web`, torch-free) exposes an options schema derived from the existing dataclasses, a video library, a run queue and a history index over `_testbed/outputs/`. Each "Process" spawns one worker subprocess that calls `cli.run()` and reports progress as JSON lines; the server relays it by SSE. A React + TypeScript front-end (`web/`, built to `web/dist`, served by FastAPI) renders the stage-lane editor, the video/segment panel and the runs/compare view.

**Tech Stack:** Python 3.11, FastAPI, uvicorn, pytest + FastAPI `TestClient`; React 18, TypeScript, Vite, vitest; `ffmpeg`/`ffprobe` (already used by `io/`).

**Spec:** `docs/superpowers/specs/2026-10-31-web-interface-design.md` (mockup: `docs/superpowers/specs/mockups/web-interface-mockup.html`, direction only).

## Global Constraints

- `core/` and `adapters/` never import the web layer; the CLI works without the `web` extra.
- **Future-proof (user requirement):** new models AND new pipeline stages must appear in the UI with **zero edits under `web/` or `src/.../web/`**. Single source of truth for stages, options and UI hints lives next to the pipeline code (`core/registry.py`, dataclass field metadata), never in a web-side table. The front-end renders unknown stage kinds, field types and artifacts through generic fallbacks, and offers an extension registry for custom views. The catalog carries `catalog_version`; adding fields/stages is non-breaking. A pipeline stage's *execution* still lives in `Pipeline`/`cli.run`; that is pipeline work, not UI work.
- Server must start, list modules and validate YAML **without importing torch**; framework imports stay inside adapters (`CLAUDE.md`).
- Single environment: `UV_PROJECT_ENVIRONMENT=_testbed/venv-infer`, every `uv run` with `--extra infer --extra web`. Plain `uv run pytest` (no `web` extra) must still pass: web tests start with `pytest.importorskip("fastapi")`.
- One GPU job at a time; each Process = one worker subprocess; v1 loads models on every run.
- UI language **English**, **dark theme only**; repo docs stay pt-BR.
- Server binds `127.0.0.1` by default; `--host 0.0.0.0` explicit. No auth.
- Time parameters are seconds. Saved YAML must load with `config.load_config`; the UI preserves unknown keys as "extras" and warns that comments/key order are lost.
- Runs write to the existing run-folder layout (`detections.csv`, `video.mp4`, `trajectories.jsonl`, `metrics.json`) in `output.out_root`, with a unique `output.tag`; old folders appear read-only in history, tolerating missing files.
- `web/dist` is not committed; FastAPI serves it when present.
- Repo process (`AGENTS.md`): each task below starts by creating `docs/task_DDDD.md` (numbers 0013–0017; 0012 is taken) before code, and ends with `DESCRIPTION.md`, `README.md`, `CHANGELOG.md` updated and the task file marked `(CONCLUÍDA)` in the same commit; work on a `feat/web_interface` branch, never `main`; commits end with the `Co-Authored-By` line.

## Review Focus

- A YAML with an unknown key, wrong type or missing required field returns 422 with the dotted path (`detectors[0].threshold: ...`), never a 500.
- An upload that is not a video (or is truncated) is rejected with 4xx and leaves no library entry or stray file.
- Cancel mid-run, and a worker that crashes (e.g. CUDA OOM), end the job as `cancelled`/`failed` with the log tail, free the queue, and leave a history entry flagged as incomplete.
- Two Process clicks queue: the second never starts while the first runs; a third while one is queued is accepted in order.
- Ids and paths from the client (`run_id`, `video_id`, "server path" import) cannot escape `out_root` / the library (`../`, symlinks, absolute paths outside allowed roots).
- `source.start` past the end of the video surfaces the `ConfigError` message ("no frames decoded...") as a readable job error.

## File Structure

```
pyproject.toml                       # + extra "web", script det-depth-distance-web
src/det_depth_distance/core/registry.py # + register_stage()/stages(): UI-facing stage descriptors
src/det_depth_distance/cli.py        # run(cfg, on_frame=None)
src/det_depth_distance/io/sinks.py   # VideoSink on_frame hook
src/det_depth_distance/web/
  __init__.py
  app.py          # create_app(root: WebPaths) -> FastAPI; routes only, thin
  paths.py        # WebPaths dataclass + safe_join()
  stages.py       # reads the stage registry from core.registry (no table here)
  schema.py       # options_schema(), module_catalog(), validate_config()
  library.py      # video library: add/list/get, probe, thumbnail, proxy
  jobs.py         # JobQueue (one runner), Job state machine, SSE event log
  worker.py       # python -m ...web.worker CONFIG.yaml: runs cli.run, JSON lines
  history.py      # scan out_root -> RunRecord; compare()
  gpu.py          # free_memory_mib() via short subprocess (torch only there)
  server.py       # console entry det-depth-distance-web (uvicorn)
docs/web_extending.md   # how to add a model / a stage / a custom view (checklist)
tests/web/        # test_*.py, conftest.py (fake worker, tmp roots)
web/              # Vite React TS app; src/{api,editor,video,runs,ui}/
```

---

### Task 1: Options schema and validation API (task_0013)

**Files:**
- Create: `src/det_depth_distance/web/{__init__,paths,stages,schema,app,server}.py`, `tests/web/{conftest,test_schema,test_api_config,test_extensibility}.py`, `docs/web_extending.md`
- Modify: `src/det_depth_distance/core/registry.py` (stage registry), `src/det_depth_distance/config.py` and the options dataclasses (UI hints as `dataclasses.field(metadata=...)` for the Kalman and tracker fields first), `tests/test_registry.py`, `pyproject.toml` (extra `web`: `fastapi`, `uvicorn`, `python-multipart`, `httpx` for tests; script `det-depth-distance-web = "det_depth_distance.web.server:main"`)

**Interfaces:**
- Produces:
  - In `core/registry.py` (no web import): `Stage` (frozen dataclass: `yaml_key: str`, `title: str`, `kind: str | None` (adapter family) , `config_cls: type | None` (for plain config sections), `multiple: bool`, `optional: bool`, `order: int`, `general: bool = False` for sections shown in "General settings"), `register_stage(stage: Stage) -> None` (duplicate `yaml_key` raises `ValueError`, like `register`) and `stages() -> tuple[Stage, ...]` sorted by `order`. The built-in stages (source, detectors, aggregation, refiners, depth, tracking, smoothing, semantic, motion; general: camera, distance, agreement, render, output) are registered once at import of `config.py`/`registry.py`.
  - UI hints are dataclass field metadata read by the schema: `field(default=..., metadata={"ui": {"label": str, "unit": str, "min": float, "max": float, "step": float, "advanced": bool, "help": str}})`; every key optional; missing hints fall back to the field name.
  - `options_schema(cls: type) -> dict` → `{"fields": [Field]}`, `Field = {"name", "type", "default", "required", "nullable", "choices"?, "items"?, "fields"?, "hint"?}`, `type ∈ bool|int|float|str|path|enum|list|map|object|any`.
  - `module_catalog() -> dict` → `{"catalog_version": 1, "stages": [...from registry.stages()], "modules": {kind: {type_name: {"schema": ...}}}, "sections": {yaml_key: {"schema": ...}}}`. Pure function of the registries: no hard-coded stage or module names in `web/`.
  - `validate_config(data: dict) -> list[dict]` → list of `{"path": str, "message": str}` (empty = valid); wraps `config.apply_overrides`-free path: `from_mapping(RunConfig, ...)` plus `registry.build` for every component spec, collecting dotted paths from `ConfigError`.
  - `create_app(paths: WebPaths) -> FastAPI` with `GET /api/catalog`, `POST /api/config/validate` (body `{config: dict}` → 200 `{errors: [...]}`), `POST /api/config/parse` (body `{yaml: str}` → `{config, extras}`), `POST /api/config/dump` (body `{config}` → `{yaml}`).
  - `WebPaths(out_root: Path, library_root: Path, data_roots: tuple[Path, ...])` and `safe_join(root: Path, rel: str) -> Path` (raises `ValueError` if the resolved path leaves `root`).

- [ ] **Step 1: Create `docs/task_0013.md` from `docs/task_TEMPLATE.md` and the branch `feat/web_interface`.**
- [ ] **Step 2: Write failing tests** in `tests/web/test_schema.py`: `test_catalog_lists_registered_adapters` (`"rfdetr" in catalog["modules"]["detector"]`, `"wbf" in catalog["modules"]["aggregator"]`), `test_literal_becomes_enum_with_choices`, `test_optional_field_is_nullable`, `test_nested_dataclass_becomes_object_with_fields`, `test_smoothing_section_exposes_accel_std_float` (field `accel_std` in `catalog["sections"]["smoothing"]`, `type == "float"`, default 2.0 for the config default), `test_validate_reports_dotted_path` (`detectors[0].treshold` typo → error path `detectors[0].treshold`), `test_validate_ok_for_recommended_config` (loads `configs/rfdetr_sam2_lanes_motion_da3.yaml`, expects `[]`), `test_server_imports_without_torch` (subprocess `python -I -c "import det_depth_distance.web.app, sys; assert 'torch' not in sys.modules"`). In `test_extensibility.py` (the proof of the future-proof requirement): `test_new_adapter_appears_in_catalog_without_web_changes` (register a fake detector with `monkeypatch.setitem(registry._REGISTRY["detector"], "fake", FakeDetector)`; it appears under `modules.detector.fake` with its fields), `test_new_stage_appears_in_catalog_in_order` (a fake `Stage(order=55)` registered via `register_stage` shows between the neighbours it was ordered between), `test_duplicate_stage_key_rejected`, `test_ui_metadata_flows_to_schema_hint` (field with `metadata={"ui": {"min": 0, "max": 10, "advanced": True}}`), `test_field_without_ui_metadata_has_empty_hint`, `test_catalog_carries_version`, and a grep-style test `test_web_package_has_no_hardcoded_adapter_names` (no registered `type_name` string literal appears in `src/det_depth_distance/web/*.py`). In `test_api_config.py`: `test_validate_endpoint_422_free` (bad config → 200 with errors, never 500), `test_parse_keeps_unknown_keys_as_extras`, `test_dump_roundtrips_through_load_config`, `test_safe_join_rejects_traversal` (`../x`, absolute path, symlink to outside).
- [ ] **Step 3: Run `uv run --extra infer --extra web pytest tests/web -q`** — expect FAIL (module missing).
- [ ] **Step 4: Implement** the modules above. `options_schema` walks `dataclasses.fields` with `typing.get_type_hints`, mirroring the annotation set `core.options._convert` supports (bool, int, float, str, Path, Literal, nested dataclass, Optional/union, tuple, dict, Any); an unsupported annotation raises `TypeError` like `from_mapping`. Seed the `ui` metadata on the Kalman fields (`smoothing.accel_std`, `std_scale`, `rel_floor`, `reset_after_s`, `tau_s`) and the tracker IoU fields. Write `docs/web_extending.md` (pt-BR) as a checklist: (a) new model = new `@register` module + `ADAPTER_MODULES` entry + optional `ui` metadata, nothing else; (b) new pipeline stage = stage code in `Pipeline`/`cli.run` + config field + `register_stage(...)`, nothing under `web/`; (c) custom inspector/view = front-end extension registry (Task 3).
- [ ] **Step 5: Run the tests** — expect PASS; also run plain `uv run pytest -q` (no `web` extra) — web tests are skipped, rest green.
- [ ] **Step 6: Docs and commit** (`feat(web): adicionar API de esquema e validação`), task file marked `(CONCLUÍDA)` with Detalhes de Implementação and Caderno de Testes.

---

### Task 2: Runs, job queue, live progress and history (task_0014)

**Files:**
- Create: `src/det_depth_distance/web/{jobs,worker,history,gpu}.py`, `tests/web/{test_jobs,test_worker,test_history,test_api_runs}.py`
- Modify: `src/det_depth_distance/cli.py` (`run(cfg: RunConfig, on_frame: Callable[[int, np.ndarray], None] | None = None) -> Path`), `src/det_depth_distance/io/sinks.py` (`VideoSink.__init__(..., on_frame=None)`; call `on_frame(index, image)` with the composed frame just before it is written), `src/det_depth_distance/web/app.py`, `tests/test_cli.py`, `tests/test_sinks.py`

**Interfaces:**
- Consumes: Task 1 `create_app`, `WebPaths`, `validate_config`, `safe_join`; `config.load_config`, `RunConfig.to_dict()`.
- Produces:
  - Worker CLI `python -m det_depth_distance.web.worker CONFIG.yaml` printing one JSON object per line: `{"event":"started","run_dir":str}`, `{"event":"frame","n":int}` (every frame; `preview.jpg` rewritten in the run dir at most every 0.5 s), `{"event":"done","run_dir":str,"frames":int}`, `{"event":"error","message":str}` (exit 2 on `ConfigError`, 1 otherwise). The worker also writes `config.yaml` (effective, from `to_dict`) and `web_run.json` (`{"status","video_id","started_at"}`) into the run dir.
  - `JobQueue(paths: WebPaths, worker_argv: Callable[[Path], list[str]])`: `submit(config: dict, video_id: str | None) -> Job`, `get(job_id) -> Job`, `cancel(job_id) -> None`, `events(job_id) -> AsyncIterator[dict]`; `Job.status ∈ queued|running|done|failed|cancelled`, `Job.log_tail: list[str]`. One runner task; cancel sends `kill()` to the worker (not SIGTERM; same lesson as the ffmpeg decoder in `CLAUDE.md`).
  - `RunRecord(id, created, video, detectors, depth, frames, wall_s, status, complete: bool, has_config: bool)` and `scan_runs(out_root) -> list[RunRecord]`; `load_run(out_root, run_id) -> RunRecord`.
  - Routes: `POST /api/runs` (`{config, video_id?}` → `{job_id}`, 422 with `errors` if invalid), `GET /api/jobs/{id}`, `GET /api/jobs/{id}/events` (SSE), `DELETE /api/jobs/{id}` (cancel), `GET /api/runs`, `GET /api/runs/{id}`, `GET /api/runs/{id}/{video|preview|config|detections}`, `GET /api/gpu` → `{free_mib: int | None}`.
  - `free_memory_mib() -> int | None` runs `python -c` with `torch.cuda.mem_get_info()` in a subprocess (server stays torch-free); `None` when torch/CUDA unavailable.

- [ ] **Step 1: Create `docs/task_0014.md`.**
- [ ] **Step 2: Write failing tests.** `test_cli.py::test_run_calls_on_frame_per_frame` (existing in-process fakes from `tests/smoke/smoke_fakes.py`; assert indices `0..n-1`); `test_sinks.py::test_video_sink_on_frame_receives_rendered_image`. `test_worker.py::test_worker_emits_json_lines_and_writes_effective_config` (fake registry adapters per `CLAUDE.md`: only `monkeypatch.setitem(registry._REGISTRY[kind], ...)`), `test_worker_exit_2_on_config_error`. `test_jobs.py` with a fake worker script (sleeps, prints events): `test_second_job_waits_for_first`, `test_cancel_running_job_kills_worker_and_marks_cancelled`, `test_worker_crash_marks_failed_with_log_tail`, `test_events_stream_ends_after_done`. `test_history.py`: `test_scan_tolerates_missing_metrics_and_config` (old folder with only `detections.csv`), `test_scan_marks_incomplete_run`, `test_run_id_traversal_rejected`. `test_api_runs.py`: `test_post_runs_invalid_config_returns_422_with_paths`, `test_start_past_end_surfaces_readable_error`.
- [ ] **Step 3: Run `uv run --extra infer --extra web pytest tests/web tests/test_cli.py tests/test_sinks.py -q`** — expect FAIL.
- [ ] **Step 4: Implement** `on_frame` hooks (keep `run()` default behaviour byte-identical), the worker, `JobQueue`, `history`, `gpu`, and routes. Job events are an append-only list plus an `asyncio.Condition`, so late SSE subscribers replay from the start. History derives `status` from `web_run.json` when present, else from the presence of `metrics.json`.
- [ ] **Step 5: Run the same tests** — expect PASS; run full `uv run --extra infer pytest -q` — green.
- [ ] **Step 6: Docs and commit** (`feat(web): adicionar fila de execuções, progresso e histórico`).

---

### Task 3: Front-end scaffold and pipeline editor (task_0015)

**Files:**
- Create: `web/` (Vite React TS app), `web/src/api/client.ts`, `web/src/editor/{Lanes,ModuleCard,Inspector,fieldControls}.tsx`, `web/src/editor/yamlState.ts`, `web/src/ui/{theme.css,Shell.tsx}`, `web/src/editor/*.test.ts(x)`
- Modify: `src/det_depth_distance/web/app.py` (serve `web/dist` at `/` with SPA fallback when it exists), `README.md` (build steps), `.gitignore` (`web/node_modules/`, `web/dist/`)

**Interfaces:**
- Consumes: `GET /api/catalog`, `POST /api/config/{validate,parse,dump}` from Task 1.
- Produces: front-end extension registry `registerStageView(yamlKey: string, view: StageViewProps => ReactElement)` and `registerFieldControl(match: (f: Field) => boolean, control: FieldControl)`; both optional, the defaults below are the fallbacks. `PipelineState` (TS) = parsed config dict + `extras`; `useEditor()` store with `addModule(stageKey, typeName)`, `removeModule(stageKey, index)`, `setOption(path: string[], value: unknown)`, `load(yaml: string)`, `toConfig(): object`, `dirty: boolean`, `diff(): {path, before, after}[]`; `fieldControls(field: Field): ReactElement` mapping `Field.type` → slider (numbers with hint min/max), select (`enum`), switch (`bool`), text/path (`str|path`), list editor (`list`), nested group (`object`); fields with `hint.advanced` go under a collapsed "Advanced" group.

- [ ] **Step 1: Create `docs/task_0015.md`.** Install `node` without sudo (version manager), document the exact commands in `README.md`; scaffold `web/` with Vite React TS, vitest, and the dark theme tokens as CSS variables (no light theme, no i18n).
- [ ] **Step 2: Write failing tests** (vitest): `unknown stage from the catalog renders as a generic lane with a card per module`, `unknown field type falls back to a raw YAML text editor with validation`, `registerFieldControl overrides the default control for a matching field`, `catalog_version greater than the supported one shows a non-blocking banner and still renders`, `fieldControls renders a slider for a float with hint range`, `fieldControls renders a select for enum with choices`, `advanced fields are collapsed by default`, `addModule appends a detector spec with schema defaults and marks dirty`, `removeModule on a single-slot stage is rejected`, `setOption updates nested path and diff() lists before/after`, `load keeps unknown keys in extras and toConfig re-emits them`, `server validation errors attach to the field by dotted path`.
- [ ] **Step 3: Run `npm test` in `web/`** — expect FAIL.
- [ ] **Step 4: Implement** the editor to match the mockup direction: left rail, top bar (Open YAML, Save YAML, Process disabled until Task 4), stage lanes (Source → Detectors → Aggregation → Refiners → Depth → Tracking & filter → Semantic → Motion), "+ add" per multi-slot stage, right inspector for the selected card (inspector content follows the click; a "General settings" gear opens `camera|distance|agreement|render|output`), YAML diff under the inspector, validation debounced at 400 ms. Save YAML downloads the dumped text and shows the comments-lost warning once.
- [ ] **Step 5: Run `npm test && npm run build`**, then `uv run --extra infer --extra web det-depth-distance-web` and open the page remotely (192.168.x:PORT with `--host 0.0.0.0`) to confirm the editor loads `configs/rfdetr_sam2_lanes_motion_da3.yaml`, a Kalman edit shows in the diff, and the saved YAML passes `load_config`.
- [ ] **Step 6: Docs and commit** (`feat(web): adicionar editor visual de pipeline`).

---

### Task 4: Video library, segment selection and Process (task_0016)

**Files:**
- Create: `src/det_depth_distance/web/library.py`, `tests/web/{test_library,test_api_videos}.py`, `web/src/video/{DropZone,Player,SegmentBar,JobPanel}.tsx`, tests for segment math
- Modify: `src/det_depth_distance/web/app.py`, `web/src/ui/Shell.tsx`

**Interfaces:**
- Consumes: Task 2 `POST /api/runs`, `GET /api/jobs/{id}/events`, `DELETE /api/jobs/{id}`, `GET /api/gpu`; Task 3 `useEditor().toConfig()`.
- Produces:
  - `Library(root: Path)`: `add(stream: BinaryIO, filename: str) -> VideoRecord`, `add_from_path(path: Path) -> VideoRecord` (path must be under an allowed `data_roots` entry), `list() -> list[VideoRecord]`, `get(video_id) -> VideoRecord`. `VideoRecord(id, name, width, height, fps, duration_s, size_bytes, original: Path, preview: Path | None, thumb: Path)`; `id` = sha256 prefix (12 hex) so re-uploads dedupe. Probing by `ffprobe`; non-video or truncated input raises `InvalidVideo`.
  - `.h264` + sibling `.meta.json` are remuxed to VFR `.mkv` using the algorithm of `data/s000032/remux_vfr.py` (SensorTimestamp), using PyAV when available; a browser-playable MP4 `preview` is produced in a background thread; the record lists `preview: None` until ready.
  - Routes: `POST /api/videos` (multipart), `POST /api/videos/import` (`{path}`), `GET /api/videos`, `GET /api/videos/{id}/{file|preview|thumb}`.
  - TS: `segmentToSource(start: number, duration: number, stride: number) -> {start, duration, stride}` used to fill `source.*` of the config on Process; `estimateSeconds(frames, fps_processing = 2) -> number`.

- [ ] **Step 1: Create `docs/task_0016.md`.**
- [ ] **Step 2: Write failing tests.** Python: `test_add_dedupes_by_hash`, `test_add_rejects_text_file_and_leaves_no_entry`, `test_add_rejects_truncated_video`, `test_import_outside_data_roots_rejected`, `test_import_symlink_escape_rejected`, `test_h264_with_meta_is_remuxed_with_sensor_timestamps` (tiny synthetic h264 + meta), `test_probe_reads_fps_size_duration`, `test_upload_endpoint_returns_record`. Vitest: `segmentToSource clamps duration to the video end`, `estimateSeconds uses stride`, `Process builds config with source.video, start, duration, stride and unique output.tag`, `Process asks for confirmation when /api/gpu reports < 4000 MiB free`.
- [ ] **Step 3: Run** `uv run --extra infer --extra web pytest tests/web/test_library.py tests/web/test_api_videos.py -q` and `npm test` — expect FAIL.
- [ ] **Step 4: Implement.** Drop anywhere on the page or paste a server path; the player starts as soon as the original (or the preview when ready) can play; the segment bar has two handles plus a `stride` field and a live estimate ("~35 s at 2 fps"); **Process** runs the segment, **Process entire video** runs `duration = 0`; JobPanel shows queue position, progress (frames / expected), the latest `preview.jpg` refreshed from SSE `frame` events, Cancel, and on `done` jumps to the run in the runs view; on `failed` shows the log tail.
- [ ] **Step 5: Run the tests again** — expect PASS; then manually process a 3 s segment of `data/s000070/0000_vfr.mkv` from the browser and confirm a new folder appears under `_testbed/outputs/` with `config.yaml` and `web_run.json`.
- [ ] **Step 6: Docs and commit** (`feat(web): adicionar biblioteca de vídeos, trecho e processamento`).

---

### Task 5: Runs view and comparison (task_0017)

**Files:**
- Create: `src/det_depth_distance/web/compare.py`, `tests/web/{test_compare,test_api_compare}.py`, `web/src/runs/{RunList,RunDetail,CompareView,DistanceChart}.tsx`, tests
- Modify: `src/det_depth_distance/web/{history,app}.py`

**Interfaces:**
- Consumes: Task 2 `scan_runs`, `load_run`, `GET /api/runs/{id}/detections`; Task 3 diff of configs.
- Produces:
  - `config_diff(a: dict, b: dict) -> list[{"path": str, "a": Any, "b": Any}]` (flat dotted paths, lists by index).
  - `run_summary(out_root, run_id) -> {"median_distance_m", "jitter_m", "fps", "frames", "boxes"}` from `detections.csv` and `metrics.json`; `jitter_m` = median over tracks of the absolute frame-to-frame change of `distance_m` (document the exact column names read, per `evaluation/run_data.py`).
  - `distance_series(out_root, run_id, track_id: int | None) -> {"t": [...], "raw": [...], "smoothed": [...]}`.
  - Run artifacts are discovered, not enumerated: `GET /api/runs/{id}/artifacts` lists the files in the run folder with sizes and media types, and `metrics.json` is rendered as a generic collapsible key/value tree, so a new sink's output or a new metrics section shows up without UI changes; known artifacts (annotated video, CSV chart) get dedicated views via the extension registry.
  - Routes: `GET /api/compare?a=ID&b=ID` → `{summary_a, summary_b, deltas, config_diff}`; `GET /api/runs/{id}/series?track=N`; `POST /api/runs/{id}/reopen` → effective config for the editor; `POST /api/runs/{id}/rerun` → `{job_id}`.
  - TS: Runs view = history list (left), annotated `video.mp4` player, KPI cards with signed deltas against the chosen comparison run, YAML diff, `DistanceChart` overlaying run B over run A with raw in grey; clicking a point seeks the player.

- [ ] **Step 1: Create `docs/task_0017.md`.**
- [ ] **Step 2: Write failing tests.** `test_config_diff_lists_changed_leaf_paths`, `test_config_diff_handles_added_and_removed_keys`, `test_summary_jitter_known_series` (hand-built CSV), `test_summary_tolerates_old_run_without_smoothed_columns`, `test_compare_missing_run_404`, `test_compare_rejects_traversal_ids`, `test_reopen_returns_effective_config`, `test_rerun_queues_same_config_and_video`, `test_artifacts_lists_unknown_files_in_run_dir`, `test_compare_ignores_columns_missing_in_old_runs`. Vitest: `unknown metrics section renders in the generic tree`. Vitest: `DistanceChart maps seconds to x and seeks on click`, `deltas show the sign and a good/bad class for jitter`.
- [ ] **Step 3: Run** `uv run --extra infer --extra web pytest tests/web/test_compare.py tests/web/test_api_compare.py -q` and `npm test` — expect FAIL.
- [ ] **Step 4: Implement** per the interfaces; reuse `evaluation/run_data` readers for the CSV instead of re-parsing columns.
- [ ] **Step 5: Run all** — `uv run --extra infer --extra web pytest -q`, `ruff check . && ruff format --check .`, `npm test && npm run build` — all green; manually compare two runs of the same segment that differ only in `smoothing.accel_std` and confirm the YAML diff shows exactly that key.
- [ ] **Step 6: Docs and commit** (`feat(web): adicionar histórico e comparação de execuções`); update `DESCRIPTION.md` (new package, endpoints, UI sections).

---

## Self-Review

- **Spec coverage:** stage-lane editor with add-module and per-module inspector (T3); fields generated from dataclasses, "Advanced" group, general settings panel (T1, T3); open/save YAML with extras and comment-loss warning (T1, T3); drag-drop video, paths, metadata, preview, dedupe, `.h264` conversion (T4); segment + Process + entire video + estimate + GPU warning (T4); live frame preview, cancel, queue (T2, T4); history incl. old folders, compare, YAML diff, chart, reopen/rerun (T2, T5); English/dark (Global Constraints, T3); torch-free server (T1 test); no core/adapters changes (only `cli.py` and `io/sinks.py` hooks, justified as the composition root and sink layer).
- **Future-proof:** T1 `test_extensibility.py` proves a fake adapter and a fake stage appear with no web edits; T3 covers generic fallbacks and the extension registry; T5 covers discovered artifacts; `docs/web_extending.md` documents the three extension paths.
- **Left to the builder, per spec:** library choices, history index details beyond the folder scan, exact SSE frame transport (T2 uses `preview.jpg` + events), caching and resident models (explicitly out).
- **Type consistency:** `WebPaths`, `safe_join`, `create_app`, `JobQueue.submit`, `RunRecord`, `scan_runs`, `segmentToSource` are named identically where consumed.
- **Proportion:** bodies are omitted; signatures, route names and test names/assertions carry the decisions.
