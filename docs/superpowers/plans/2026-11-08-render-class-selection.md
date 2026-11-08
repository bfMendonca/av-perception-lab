# Render class selection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let the user choose which semantic classes are drawn in the rendered video, from class lists the adapters declare before loading, through a YAML option and a web checkbox dropdown.

**Architecture:** A render-only allowlist `render.semantic_classes` (`<map>.<class>`, `None` = all) turns hidden classes transparent in the renderer's lookup table. `Component.declared_classes(options)` (classmethod, default `None`) gives the web editor each model's classes without loading; a load-time cache under the weights root covers detectors/segmenters with no cheap static source. A new `POST /api/classes` returns the list; a field control renders the dropdown.

**Tech Stack:** Python 3.x (dataclasses, numpy, FastAPI, pytest), React 18 + TypeScript + vitest.

**Spec:** `docs/superpowers/specs/2026-11-08-render-class-selection-design.md`

## Global Constraints

- Selection is presentation only: distance, tracking, CSV, NPZ and metrics values must not change.
- `render.semantic_classes: None` means every class of every drawn map; existing YAMLs render exactly as before.
- A non-`None` list is an allowlist; a map with no entry draws no classes.
- Entries are `<map>.<class>`; prefix checked in `RunConfig.__post_init__`, class checked after `pipeline.load` (same split as `exclude_semantic`).
- `POST /api/classes` never loads a model and never unpickles a checkpoint.
- Class cache path: `<weights root>/.classes/<key>.json`; the key changes when the weights file's path, size or mtime changes; a write failure is logged and ignored; `class_names` after `load()` stays authoritative.
- `background` is listed but not selectable (the renderer already never draws it).
- Imports of `torch`/framework only inside adapter `_load` and methods called after it; `declared_classes` must work without torch.
- Docs and the task file are in English. Never commit on `main`: branch `feat/render_class_selection`. `DESCRIPTION.md` is updated in the same commit as endpoint/option changes.
- Lint: `uv run ruff check --fix . && uv run ruff format .`; tests: `uv run pytest`; web: `cd web && npm test && npm run typecheck`.

## Review Focus

1. `render.semantic_classes: []` (empty list): draws no semantic classes at all, no error. Pinned in Task 1 (config) and Task 2 (renderer).
2. A class name containing a dot, or a space (`traffic light`): entries split on the **first** dot only. Pinned in Task 1.
3. A truncated or corrupt cache JSON file: treated as a miss, no crash. Pinned in Task 4.
4. Weights file replaced (new mtime/size) after caching: stale entry ignored. Pinned in Task 4.
5. Two maps sharing a class name (`segformer.car`, `other.car`): chosen independently. Pinned in Task 2.

---

### Task 0: Task file and branch

**Files:**
- Create: `docs/task_0035.md` (from `docs/task_TEMPLATE.md`, English)

- [ ] **Step 1:** `git switch -c feat/render_class_selection` from `main`.
- [ ] **Step 2:** Write `docs/task_0035.md` with Description, Scope (file table from Tasks 1-7), Plan, Impact analysis, Verification, Dependencies, summarising the spec. Do not mark it `(COMPLETED)` yet (Task 7 does).
- [ ] **Step 3: Commit**

```bash
git add docs/task_0035.md docs/superpowers
git commit -m "docs: task_0035 render class selection (spec, plan)"
```

### Task 1: `render.semantic_classes` config and name resolution

**Files:**
- Modify: `src/av_perception_lab/config.py` (`RenderConfig`, `RunConfig.__post_init__` near the `semantic_overlay` check)
- Modify: `src/av_perception_lab/core/labels.py` (next to `group_semantic_labels`)
- Test: `tests/test_config.py`, `tests/test_labels.py`

**Interfaces:**
- Produces: `RenderConfig.semantic_classes: tuple[str, ...] | None = None`
- Produces: `resolve_semantic_classes(selection: Sequence[str] | None, classes_by_map: Mapping[str, Sequence[str]]) -> dict[str, frozenset[str]] | None` in `core/labels.py`. `None` selection returns `None`. Otherwise returns, for every map in `classes_by_map`, the allowed class set (empty set for a map with no entry). Raises `ConfigError("render.semantic_classes: <map>.<class>: unknown class; <map> classes: a, b")` for an unknown class, and `ConfigError("render.semantic_classes: unknown semantic map '<map>'; configured: ...")` for an unknown prefix. Splits on the first `.` only.

- [ ] **Step 1: Write the failing tests**
  - `test_render_semantic_classes_default_none`: `RenderConfig().semantic_classes is None`.
  - `test_run_config_rejects_unknown_map_prefix`: a `RunConfig` with `semantic=[{name:"scene",...}]` and `render.semantic_classes=("nope.road",)` raises `ValueError` matching `unknown semantic map 'nope'`.
  - `test_yaml_list_form_is_accepted`: `semantic_classes: [scene.road]` loads to `("scene.road",)`. (A bare YAML string is not specified: accept it as a one-item list like `exclude_semantic` does, and say so in the field comment: builder's call.)
  - In `test_labels.py`: `test_resolve_none_means_all`; `test_resolve_empty_list_selects_nothing` (`[]` → `{"scene": frozenset()}`); `test_resolve_splits_on_first_dot` (class `"a.b"` in map `scene` via `"scene.a.b"`); `test_resolve_unknown_class_lists_valid_names`; `test_resolve_same_class_in_two_maps` (`"scene.car"` selects only in `scene`, not in `other`).
- [ ] **Step 2:** Run `uv run pytest tests/test_config.py tests/test_labels.py -v`; expect FAIL (missing field/function).
- [ ] **Step 3:** Add the field to `RenderConfig` with a comment stating allowlist semantics; validate the prefix in `RunConfig.__post_init__` via a prefix-only check beside the `semantic_overlay` one. Implement `resolve_semantic_classes` in `labels.py`.
- [ ] **Step 4:** Run the same command; expect PASS. Then `uv run pytest` for regressions (the web catalog/schema tests enumerate `RenderConfig` fields; update expectations if they list them).
- [ ] **Step 5: Commit** `feat(config): render.semantic_classes (task_0035)`.

### Task 2: Renderer, VideoSink and CLI validation

**Files:**
- Modify: `src/av_perception_lab/render.py` (`Renderer.render`, `_semantic_lut`, `_blend_semantic`)
- Modify: `src/av_perception_lab/io/sinks.py` (`VideoSink` frame method that builds `layers`)
- Modify: `src/av_perception_lab/cli.py` (right after `pipeline.load(source, info)`)
- Modify: `src/av_perception_lab/core/pipeline.py` (add `semantic_classes(self) -> dict[str, tuple[str, ...]]`)
- Test: `tests/test_render.py`, `tests/test_sinks.py`, `tests/test_cli.py`, `tests/test_pipeline.py`

**Interfaces:**
- Consumes: `resolve_semantic_classes` and `RenderConfig.semantic_classes` (Task 1).
- Produces: `Renderer.render(..., semantic: Sequence[tuple[np.ndarray, Sequence[str]]] = (), semantic_visible: Sequence[frozenset[str] | None] = (), ...)`: `semantic_visible[i]` applies to `semantic[i]`; missing or `None` = all classes visible.
- Produces: `Renderer._semantic_lut(class_names: tuple[str, ...], visible: frozenset[str] | None) -> np.ndarray`, cache keyed by `(class_names, visible)`.
- Produces: `Pipeline.semantic_classes() -> dict[str, tuple[str, ...]]` (segmenter name → `class_names`, valid after load).

- [ ] **Step 1: Write the failing tests**
  - `test_semantic_hidden_class_is_transparent`: a 2×2 labels array with classes `("background","road","car")`; `visible=frozenset({"road"})`; the `car` pixel in the output equals the input pixel, the `road` pixel differs.
  - `test_semantic_empty_visible_draws_nothing`: `visible=frozenset()` → output equals input.
  - `test_semantic_lut_cache_distinguishes_visible_sets`: two calls with the same names and different `visible` return different alpha columns.
  - `test_two_maps_share_class_name_independently`: two layers, both with class `car`, `visible=[frozenset({"car"}), frozenset()]` → only the first layer tints.
  - `test_render_without_visible_is_unchanged`: existing call signature and output identical to before.
  - Sink: `test_video_sink_passes_allowed_classes` using the existing fake-writer pattern in `tests/test_sinks.py`: with `semantic_classes=("scene.road",)` the renderer receives `{"road"}` for map `scene`.
  - CLI/pipeline: `test_unknown_semantic_class_is_config_error_before_first_frame`: through the in-process CLI of `tests/smoke/smoke_fakes.py`, a class not in the fake segmenter's `class_names` raises `ConfigError` naming the valid classes; `test_pipeline_semantic_classes_after_load`.
  - `test_distance_and_csv_unchanged_by_selection`: run the smoke pipeline twice, `semantic_classes=None` vs `("scene.road",)`; the CSV files are byte-identical.
- [ ] **Step 2:** Run `uv run pytest tests/test_render.py tests/test_sinks.py tests/test_cli.py tests/test_pipeline.py -v`; expect FAIL.
- [ ] **Step 3:** Implement in `render.py`: hidden classes get alpha 0 exactly like `SEMANTIC_BACKGROUND`. In `VideoSink`, resolve once per frame from `cfg.semantic_classes` and `result.semantic` (a map absent from the resolved dict is all-hidden when the selection is non-`None`). In `cli.py`, call `resolve_semantic_classes(cfg.render.semantic_classes, pipeline.semantic_classes())` after load so errors surface before frame 1 (same place the exclude check runs).
- [ ] **Step 4:** Run the same command; expect PASS; then `uv run pytest`.
- [ ] **Step 5: Commit** `feat(render): draw only the selected semantic classes (task_0035)`.

### Task 3: `declared_classes`, TwinLiteNet+ and SegFormer

**Files:**
- Modify: `src/av_perception_lab/core/components.py` (`ClassCatalog`, `Component.declared_classes`)
- Modify: `src/av_perception_lab/adapters/semantic/twinlitenet.py` (`declared_classes`, head check in `_load`)
- Modify: `src/av_perception_lab/adapters/semantic/segformer.py` (`declared_classes` reading `config.json`)
- Test: `tests/test_components.py`, `tests/test_twinlitenet.py`, `tests/test_segformer.py`

**Interfaces:**
- Produces: `ClassCatalog(names: tuple[str, ...], source: Literal["static", "metadata", "cache"])`, frozen dataclass, in `core/components.py`.
- Produces: `Component.declared_classes(cls, options: Any) -> ClassCatalog | None` (classmethod; base returns `None`).
- Produces: `SegFormer.declared_classes(options)` returns `ClassCatalog(names, "metadata")` from the `config.json` of `options.model`'s snapshot in `options.hf_cache` (use `resolve_cache` and the HF cache layout; read it as JSON, no `transformers` import); `None` when the file is missing or invalid.

- [ ] **Step 1: Write the failing tests**
  - `test_base_declared_classes_is_none`.
  - `test_twinlitenet_declared_classes_static`: equals `ClassCatalog(("background","drivable","lane"), "static")`; works with no torch installed.
  - `test_twinlitenet_load_rejects_head_with_other_channels`: a fake model whose head has 5 output channels raises `ContractError` mentioning the expected 3 classes (use the existing fakes pattern in the file).
  - `test_segformer_declared_classes_from_config_json`: a `tmp_path` HF cache with a `snapshots/<rev>/config.json` containing `id2label` for 3 classes; result `names == ("a","b","c")`, `source == "metadata"`; also `test_segformer_declared_classes_missing_file_is_none` and `..._invalid_ids_is_none` (non-contiguous ids).
  - `test_declared_classes_match_class_names_after_load` (contract test): for each of `FakeSemantic`-style fakes registered in the test, a non-`None` declaration equals `class_names` after `load()`.
- [ ] **Step 2:** Run `uv run pytest tests/test_components.py tests/test_twinlitenet.py tests/test_segformer.py -v`; expect FAIL.
- [ ] **Step 3:** Implement; reuse `_id2label_names` in `segformer.py` for the id check.
- [ ] **Step 4:** Run the same command; expect PASS.
- [ ] **Step 5: Commit** `feat(core): declared_classes for segmenters (task_0035)`.

### Task 4: Load-time class cache for detectors

**Files:**
- Create: `src/av_perception_lab/core/class_cache.py`
- Modify: `src/av_perception_lab/core/components.py` (`Detector`: `_weights_file`, `declared_classes`, write after load)
- Modify: `src/av_perception_lab/adapters/detectors/ultralytics_yolo.py`, `src/av_perception_lab/adapters/detectors/rfdetr.py` (`weights_file`)
- Test: `tests/test_class_cache.py`, `tests/test_components.py`

**Interfaces:**
- Produces in `class_cache.py`: `cache_key(weights: Path) -> str | None` (hash of resolved path, size and `st_mtime_ns`; `None` if the file is missing); `read(weights: Path) -> tuple[str, ...] | None`; `write(weights: Path, names: Sequence[str]) -> None` (never raises: logs and returns on `OSError`). Files live at `<weights root>/.classes/<key>.json` via `requirements.weights_root()`.
- Produces: `Detector.weights_file(cls, options) -> Path | None` (classmethod, base `None`); YOLO returns `resolve_weights(options.weights)`; RF-DETR returns `options.weights` if set, else `checkpoint_path(options.variant)`.
- Produces: `Detector.declared_classes(options)` returns `ClassCatalog(names, "cache")` from `class_cache.read(weights_file)` or `None`.
- Consumes: `ClassCatalog` (Task 3).

- [ ] **Step 1: Write the failing tests** (use `set_weights_root(tmp_path)` and restore it with a fixture)
  - `test_write_then_read_roundtrip`.
  - `test_replaced_weights_file_is_a_miss`: write cache, rewrite the weights file with different size → `read` returns `None`.
  - `test_touched_mtime_is_a_miss`.
  - `test_corrupt_json_is_a_miss`: truncate the cache file; `read` returns `None`, no exception.
  - `test_read_only_root_does_not_raise`: root made unwritable (`chmod 0o500`); `write` returns without error.
  - `test_missing_weights_file_has_no_key`.
  - `test_detector_load_writes_cache_and_declares_it`: a fake detector with `weights_file` pointing at a tmp file; before load `declared_classes` is `None`; after `load()` it is `ClassCatalog(names, "cache")`.
  - `test_cache_does_not_override_loaded_names`: a stale cache with other names does not change `class_names` after load.
- [ ] **Step 2:** Run `uv run pytest tests/test_class_cache.py tests/test_components.py -v`; expect FAIL.
- [ ] **Step 3:** Implement; `Detector._after_load` calls `class_cache.write` when `weights_file` is not `None`. Add `.classes/` to nothing in git (the weights root is already ignored).
- [ ] **Step 4:** Run the same command; expect PASS; then `uv run pytest`.
- [ ] **Step 5: Commit** `feat(core): load-time class cache for detectors (task_0035)`.

### Task 5: `POST /api/classes`

**Files:**
- Modify: `src/av_perception_lab/web/app.py` (route next to `/api/models/status`), `src/av_perception_lab/web/schema.py` or a new `src/av_perception_lab/web/classes.py` for the logic
- Test: `tests/web/test_api_classes.py`

**Interfaces:**
- Consumes: `Component.declared_classes`, `registry.build`-style option construction used by `schema.validate_config` (build the options dataclass only; never instantiate-and-load).
- Produces: `POST /api/classes` body `{"kind": "semantic"|"detector", "spec": {type, name?, **options}}` → `200 {"names": [...], "source": "static"|"metadata"|"cache"}` or `200 {"names": null, "source": null}`; invalid spec → `422 {"errors": [...]}` in the same shape as `/api/models/status`; unknown type → 422.

- [ ] **Step 1: Write the failing tests** (reuse `tests/web/conftest.py` fixtures and `model_fakes.py`)
  - `test_twinlitenet_returns_static_names`.
  - `test_unknown_classes_returns_null`: a fake adapter without a declaration.
  - `test_invalid_options_is_422`.
  - `test_never_loads_a_model`: a fake adapter whose `_load` raises; the endpoint still answers.
  - `test_path_outside_allowed_roots_is_rejected` using the existing `SERVER_CHECKED_PATHS` machinery for path options.
- [ ] **Step 2:** Run `uv run pytest tests/web/test_api_classes.py -v`; expect FAIL (404).
- [ ] **Step 3:** Implement with a pydantic body model (`extra="forbid"`, as `StatusBody`).
- [ ] **Step 4:** Run the same command; expect PASS; then `uv run pytest tests/web`.
- [ ] **Step 5: Commit** `feat(web): POST /api/classes (task_0035)`.

### Task 6: Dropdown in the editor

**Files:**
- Create: `web/src/api/classes.ts`, `web/src/editor/ClassSelect.tsx`, `web/src/editor/ClassSelect.test.tsx`
- Modify: `web/src/editor/registry.ts` usage (register a field control), `web/src/editor/EditorView.tsx` or wherever custom controls are registered (follow `docs/web_extending.md` §(c))

**Interfaces:**
- Consumes: `POST /api/classes` (Task 5); `registerFieldControl`-style entry in `registry.ts` (`match: (f: Field) => boolean`; match the `semantic_classes` field of the render section).
- Produces: `fetchClasses(kind: string, spec: unknown): Promise<{ names: string[] | null; source: string | null }>`; `ClassSelect` control with props `FieldControlProps`. It reads the config's `semantic` specs (find how the editor exposes the current config to custom controls; if `StageContext` does not carry it, add a field to it, never a new global), fetches each map's classes, and shows one group per map labelled by the map name, entries `<map>.<class>`.

- [ ] **Step 1: Write the failing tests** (vitest, API client faked like `presets.test.ts`)
  - `test_all_checked_by_default`: value `undefined` → every box checked.
  - `test_unchecking_writes_allowlist`: unchecking `scene.sky` calls `onChange` with the other entries.
  - `test_checking_everything_again_writes_undefined`: the control returns the value to "all", so the YAML key disappears.
  - `test_select_none_writes_empty_list`.
  - `test_search_filters_visible_entries`.
  - `test_background_is_disabled_with_note`.
  - `test_unknown_names_shows_message`: `names: null` renders "classes appear after the first load".
- [ ] **Step 2:** Run `cd web && npm test -- ClassSelect`; expect FAIL.
- [ ] **Step 3:** Implement the component and register it; keep it keyboard-accessible (labels bound to checkboxes, like `SwitchControl`).
- [ ] **Step 4:** Run `cd web && npm test && npm run typecheck`; expect PASS. Run `npm run build`.
- [ ] **Step 5: Commit** `feat(web): class dropdown for semantic render (task_0035)`.

### Task 7: Docs, changelog and closing the task

**Files:**
- Modify: `DESCRIPTION.md`, `docs/reference.md` (YAML field), `CHANGELOG.md`, `docs/web_extending.md`, `docs/task_0035.md`
- Create/modify: an example in `configs/rfdetr_semantic_da3.yaml` only if it shows the option as a commented line.

- [ ] **Step 1:** Update `DESCRIPTION.md` (endpoint, option, cache) in the same commit as the closing changes; add the `docs/reference.md` section for `render.semantic_classes` including allowlist semantics and the cache behaviour; add a `CHANGELOG.md` entry; extend `docs/web_extending.md` with `declared_classes` as a rule for new adapters.
- [ ] **Step 2:** Mark `docs/task_0035.md` `(COMPLETED)` and add `Implementation details` and `Test notebook`.
- [ ] **Step 3:** Run `uv run ruff check --fix . && uv run ruff format . && uv run pytest && cd web && npm test && npm run typecheck`; all pass.
- [ ] **Step 4: Commit** `docs: task_0035 render class selection (completed)`; integrate via pull request, never directly on `main`.
