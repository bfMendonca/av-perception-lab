# Model Readiness ("Prepare model") Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Every model declares what it needs (weights, external code, Python packages); the CLI and the web interface check readiness without torch and prepare missing pieces with one action, showing not-ready models as yellow dashed cards with an alert icon.

**Architecture:** A framework-free requirements model and provisioning engine live in `core/` (declared per component class through a `requirements(options)` hook, with registered source kinds so new sources need no engine edits). The CLI gets `--prepare`; the web layer adds a status endpoint, a separate prepare queue and generic UI states. v1 handles dependency conflicts by detection only (dry-run), with the other resolution paths documented.

**Tech Stack:** Python 3.11, `huggingface_hub`, stdlib `urllib`/`git` CLI, `uv` (dry-run), FastAPI (existing), React/TypeScript (existing), pytest, vitest.

**Spec:** `docs/superpowers/specs/2026-10-31-model-readiness-design.md` (mockup: `docs/superpowers/specs/mockups/model-readiness-mockup.html`, direction only).

## Global Constraints

- **Everything new is written in English**: documentation, `docs/task_DDDD.md` files (template headings translated), commit messages, UI text, code comments. Existing pt-BR documents are NOT migrated by this plan.
- `core/` and `adapters/` never import the web layer; the CLI works without the `web` extra; framework imports stay inside adapter `_load` methods.
- Readiness checks never import torch (file/folder existence, `importlib.util.find_spec`, `importlib.metadata`).
- Future-proof: adding a model or a source kind needs zero edits under `web/` or `src/.../web/`; the UI never lists model names; sources are fixed in code, never typed by the client.
- Downloads are atomic (`.part` then rename), resumable when the source supports it, cancellable, idempotent; one Prepare at a time, independent from the GPU job queue.
- Default location: `weights/<family>/<model>/` for assets and `weights/_code/<name>/` for clones, via `--weights-root` / `DDD_WEIGHTS_ROOT`; a path in the YAML is an override. Prepare never moves or deletes existing weights (`~/.roboflow`, `weights/hf`).
- The existing `infer` extra is not split; the server never installs the base stack (documented one-time command); per-model extras install only after a clean dry-run, with the union of detected extras, never a bare `uv sync`.
- Gated models: Prepare reports "gated — accept the licence at <link> and set `HF_TOKEN`"; no token entry in the UI.
- Google Drive is not in v1 (extension point only).
- Not-ready pipeline: YAML still loads/validates; UI blocks Process; server refuses submit with 409 listing models; CLI fails before loading with a message naming `--prepare`. DA3 stops downloading at run time.
- CLI shape: flag `--prepare` (no subcommands; existing invocations unchanged).
- Repo process (`AGENTS.md`): each task starts with `docs/task_DDDD.md` (numbers 0019–0025; 0012 is taken) before code and ends with `DESCRIPTION.md`, `README.md`, `CHANGELOG.md` updated and the task file marked `(COMPLETED)` in the same commit; branch `feat/model_readiness` created from `feat/web_interface`; never `main`; commits end with the `Co-Authored-By` line; environment `UV_PROJECT_ENVIRONMENT=_testbed/venv-infer` with `uv run --extra infer --extra web` for every command.

## Review Focus

- A download that is interrupted, truncated or fails its sha256 is never reported `ready`; a second Prepare resumes or restarts cleanly (no stale `.part` accepted).
- Asset targets and clone folder names derived from options cannot escape the weights root (`../`, absolute paths, symlinks); containment is checked on the resolved path.
- Cancelling a Prepare mid-download leaves no partial file counted as ready and frees the prepare queue; a failed Prepare ends as `failed` with a readable reason (offline, 404, gated), never a crash.
- On a fresh clone (no weights, no network, no torch imported) status for every shipped sample YAML computes without importing torch, and a `UserProvided` asset never triggers a download (message "provide this file").
- A conflict dry-run leaves the environment untouched (no files change) and shows the resolver's reason; the install command is built from detected extras and never removes the base stack.
- Two Prepare requests queue one at a time; a Prepare on an already-ready model is a no-op; a run submitted while a model is not ready is refused with 409 (web) or a clear error (CLI) instead of failing deep in `load()`.

## File Structure

```
src/det_depth_distance/core/requirements.py   # Asset kinds, CodeRepo, Requirements, ModelStatus, weights_root(), check()
src/det_depth_distance/core/provision/        # engine.py (prepare), sources.py (hf, url, git), packages.py (dry-run, install)
src/det_depth_distance/core/components.py     # + Component.requirements(options) classmethod
src/det_depth_distance/adapters/**            # each adapter declares requirements; defaults under weights root
src/det_depth_distance/cli.py                 # --prepare
src/det_depth_distance/web/models.py          # status + prepare routes, PrepareQueue
src/det_depth_distance/web/{app,server}.py    # wiring, --weights-root, 409 on not-ready submit
web/src/models/                               # status hook, banner, card states, inspector section
configs/*.yaml                                # machine-specific paths removed
docs/model_environments.md                    # recipes + every conflict-resolution path
tests/{test_requirements,test_provision_*,test_adapter_requirements}.py, tests/web/test_models*.py
```

---

### Task 1: Requirements model and torch-free status check (task_0019)

**Files:**
- Create: `src/det_depth_distance/core/requirements.py`, `tests/test_requirements.py`
- Modify: `src/det_depth_distance/core/components.py` (hook), `docs/task_TEMPLATE.md` (English headings), `AGENTS.md` (one line: documentation and task files are in English)

**Interfaces:**
- Produces:
  - `Asset` (frozen base dataclass: `key: str`, `target: str`, `size_bytes: int | None`, `license: str | None`, `gated: bool = False`) and registered kinds via `register_asset_kind(name: str)(cls)`; built-ins `HuggingFaceAsset(repo: str, files: tuple[str, ...] = (), patterns: tuple[str, ...] = (), revision: str | None = None)`, `UrlAsset(url: str, sha256: str | None = None)`, `UserProvided(option: str, hint: str)`.
  - `CodeRepo(name: str, url: str, commit: str)`; `Requirements(assets: tuple[Asset, ...] = (), code: tuple[CodeRepo, ...] = (), packages: tuple[str, ...] = (), extra: str | None = None)` (`packages` = import names; `extra` = the `pyproject` extra that provides them).
  - `weights_root() -> Path` (order: explicit setter, `DDD_WEIGHTS_ROOT`, repo `weights/`), `set_weights_root(path: Path) -> None`.
  - `ModelStatus(state: Literal["ready","missing","conflict","failed","preparing"], layers: dict[str, str], missing: tuple[str, ...], size_bytes: int | None, message: str | None)` and `check(req: Requirements, root: Path | None = None) -> ModelStatus`.
  - `Component.requirements(cls, options) -> Requirements` classmethod, default `Requirements()`.

- [ ] **Step 1: Create `docs/task_0019.md`** from the English `docs/task_TEMPLATE.md`; translate the template headings, add the AGENTS.md language line, create branch `feat/model_readiness` from `feat/web_interface`.
- [ ] **Step 2: Write failing tests** in `tests/test_requirements.py`: `test_empty_requirements_are_ready`, `test_missing_asset_file_is_missing_with_layer_detail`, `test_present_asset_is_ready`, `test_truncated_asset_with_known_size_or_sha_is_not_ready`, `test_part_file_is_never_ready`, `test_user_provided_missing_reports_hint_and_never_downloadable`, `test_code_repo_ready_only_when_commit_matches_head` (local git fixture), `test_package_layer_uses_find_spec_without_importing` (assert the module is not in `sys.modules` after check), `test_check_does_not_import_torch` (subprocess `-I`), `test_asset_target_escape_is_rejected` (`../x`, absolute, symlink), `test_register_asset_kind_makes_new_kind_checkable`, `test_weights_root_precedence`.
- [ ] **Step 3: Run `uv run --extra infer --extra web pytest tests/test_requirements.py -q`** — expect FAIL.
- [ ] **Step 4: Implement** the module and the hook. `check` aggregates three layers (assets → files exist, sizes/sha when known, no `.part`; code → folder exists and `git rev-parse HEAD` equals the pinned commit; packages → `importlib.util.find_spec`).
- [ ] **Step 5: Run tests** — expect PASS; full `uv run --extra infer pytest -q` green.
- [ ] **Step 6: Docs and commit** (`feat(core): add model requirements declaration and status check`).

---

### Task 2: Provisioning engine, sources and `--prepare` (task_0020)

**Files:**
- Create: `src/det_depth_distance/core/provision/{__init__,engine,sources}.py`, `tests/test_provision_engine.py`, `tests/test_provision_sources.py`
- Modify: `src/det_depth_distance/cli.py` (`--prepare`), `tests/test_cli.py`

**Interfaces:**
- Consumes: Task 1 `Requirements`, `Asset` kinds, `ModelStatus`, `check`, `weights_root`.
- Produces:
  - `register_source(asset_cls: type[Asset])(source_cls)`; `Source` protocol `prepare(asset: Asset, root: Path, progress: Callable[[int, int | None], None], cancel: threading.Event) -> None`; built-in sources for `HuggingFaceAsset` (`huggingface_hub.snapshot_download` with `allow_patterns`, `cache_dir=root/'hf'` layout), `UrlAsset` (stream to `<target>.part`, resume with `Range`, verify `sha256`, rename), `CodeRepo` (`git clone` + `git checkout <commit>` into `root/_code/<name>`), and `UserProvided` (raises `ProvisionError` with the hint).
  - `prepare(req: Requirements, root: Path | None, progress: Callable[[str, int, int | None], None], cancel: threading.Event) -> ModelStatus` — idempotent, no-op when ready, raises/returns `failed` with a readable reason for offline/404/gated/sha mismatch.
  - CLI: `det-depth-distance --config X.yaml --prepare` builds components from the config, prepares each (progress on stdout), prints a status table, exits 0 only if all ready.

- [ ] **Step 1: Create `docs/task_0020.md`.**
- [ ] **Step 2: Write failing tests** (no network: a local `http.server` thread and a local bare git repo as fixtures): `test_url_source_downloads_atomically_and_verifies_sha`, `test_url_source_resumes_from_part_file`, `test_url_sha_mismatch_ends_failed_and_removes_part`, `test_cancel_mid_download_leaves_no_ready_file`, `test_git_source_clones_pinned_commit`, `test_prepare_is_noop_when_ready`, `test_hf_source_passes_allow_patterns_and_cache_dir` (monkeypatched `snapshot_download`), `test_gated_error_message_names_licence_and_hf_token`, `test_offline_failure_is_readable`, `test_unknown_asset_kind_without_source_fails_clearly`, `test_register_source_adds_new_kind_without_engine_edit`, `test_cli_prepare_exit_code_and_table` (fake registered component), `test_cli_existing_invocations_unchanged`.
- [ ] **Step 3: Run the new tests** — expect FAIL.
- [ ] **Step 4: Implement** sources and engine; targets are resolved with the Task 1 containment helper before any write.
- [ ] **Step 5: Run tests** — expect PASS; full suite green.
- [ ] **Step 6: Docs and commit** (`feat(core): add provisioning engine and --prepare`).

---

### Task 3: Adapter migration and portable sample YAMLs (task_0021)

**Files:**
- Modify: `src/det_depth_distance/adapters/{depth/da3.py,detectors/rfdetr.py,detectors/ultralytics_yolo.py,refiners/hf_sam.py,refiners/ultralytics_sam.py,semantic/segformer.py,semantic/twinlitenet.py}`, `src/det_depth_distance/adapters/_hf_hub.py`, `configs/*.yaml`, `tests/test_adapter_requirements.py` (create), affected adapter tests

**Interfaces:**
- Consumes: Task 1 `Requirements`/assets, Task 2 sources.
- Produces: each adapter implements `requirements(options)`: Hugging Face adapters and DA3 declare `HuggingFaceAsset` into the `weights/hf` cache layout (existing caches keep working); DA3 loses the run-time download (missing weights → `ConfigError` naming `--prepare`); RF-DETR declares a `UrlAsset` for the variant's checkpoint and the adapter loads `rfdetr.from_checkpoint(<weights path>)` (existing `~/.roboflow` untouched); YOLO and Ultralytics SAM declare `UrlAsset` for known names (ultralytics assets) and `UserProvided("weights")` for custom `.pt`; TwinLiteNet+ declares `CodeRepo(name="TwinLiteNetPlus", url=..., commit="90f1b86…")` (weights ship in the clone; `~/dev/TwinLiteNetPlus` stays a valid override); `packages=("rfdetr","depth_anything_3",…)` with `extra="infer"`. Default option values resolve under `weights_root()`; every shipped `configs/*.yaml` drops machine paths (`~/dev/...`) for defaults.

- [ ] **Step 1: Create `docs/task_0021.md`.** Look up the real URLs, sizes, licences and commit hashes per model (record in the task file; leave `sha256=None` where unknown).
- [ ] **Step 2: Write failing tests**: `test_every_registered_adapter_declares_or_explicitly_empty_requirements`, `test_da3_requirements_use_hf_cache_layout_and_no_runtime_download`, `test_rfdetr_loads_from_explicit_checkpoint_path`, `test_yolo_custom_pt_is_user_provided`, `test_twinlitenet_declares_pinned_clone`, `test_existing_explicit_paths_in_yaml_still_override_defaults`, `test_every_shipped_config_loads_and_checks_without_importing_torch`, `test_no_shipped_config_contains_a_home_directory_path` (scan `configs/*.yaml` for `~/dev`).
- [ ] **Step 3: Run** — expect FAIL.
- [ ] **Step 4: Implement** per adapter; keep behaviour with explicit paths identical; update the adapter tests that asserted the old error texts.
- [ ] **Step 5: Run** the adapter tests and the full suite — green.
- [ ] **Step 6: Docs and commit** (`refactor(adapters): declare requirements and portable defaults`).

---

### Task 4: Web API, prepare queue and run gating (task_0022)

**Files:**
- Create: `src/det_depth_distance/web/models.py`, `tests/web/test_models.py`, `tests/web/test_models_gating.py`
- Modify: `src/det_depth_distance/web/{app,server}.py`, `src/det_depth_distance/web/security.py` (reuse containment for the weights root)

**Interfaces:**
- Consumes: Task 1–3 core/engine; existing `create_app`, `JobQueue`, security helpers, SSE style.
- Produces:
  - `POST /api/models/status` (body `{config: dict}` → `{models: [{stage, index, type, name, state, layers, missing, size_bytes, message}], total_size_bytes}`); components are built from the validated config exactly like `validate_config`, never loaded.
  - `PrepareQueue(weights_root: Path)`: `submit(models: list[key] | "all", config) -> PrepareJob`, `get`, `cancel`, `events` (SSE with per-model progress); one job at a time; independent from `JobQueue`; routes `POST /api/models/prepare`, `GET /api/models/prepare/{id}`, `GET /api/models/prepare/{id}/events`, `DELETE /api/models/prepare/{id}`.
  - `submit_run` returns 409 `{models_not_ready: [...]}` when any model of the config is not `ready`; server flag `--weights-root`.

- [ ] **Step 1: Create `docs/task_0022.md`.**
- [ ] **Step 2: Write failing tests**: `test_status_for_fresh_config_lists_missing_models_without_torch`, `test_status_ready_after_fake_asset_present`, `test_prepare_all_runs_one_model_at_a_time`, `test_second_prepare_waits_for_first`, `test_cancel_prepare_frees_queue_and_leaves_failed_not_ready`, `test_prepare_unknown_model_key_is_422`, `test_prepare_never_uses_client_supplied_urls` (body fields ignored/rejected), `test_submit_run_409_when_model_not_ready`, `test_submit_run_ok_when_all_ready`, `test_prepare_does_not_block_gpu_job_queue`, `test_weights_root_containment_enforced`.
- [ ] **Step 3: Run** `uv run --extra infer --extra web pytest tests/web -q` — new tests FAIL.
- [ ] **Step 4: Implement**; the prepare worker runs the engine in a thread (downloads are I/O), cancellation through the event; persist nothing beyond logs.
- [ ] **Step 5: Run** `tests/web` and the full suite — green.
- [ ] **Step 6: Docs and commit** (`feat(web): add model status and prepare API`).

---

### Task 5: Interface states and Prepare actions (task_0023)

**Files:**
- Create: `web/src/models/{useModelStatus.ts,ReadinessBanner.tsx,statusStyles.css,ModelRequirements.tsx}` and tests
- Modify: `web/src/editor/{ModuleCard.tsx,Lanes.tsx,Inspector.tsx}`, `web/src/video/` (Process gating), `web/src/api/`

**Interfaces:**
- Consumes: Task 4 routes, existing editor store (`toConfig()`), `ModuleCard`, job panel patterns.
- Produces: `useModelStatus(config)` (debounced `POST /api/models/status`, maps `{stage,index}` to a status); card variants — *ready* (solid), *missing* (yellow dashed outline + small alert icon at the top-right corner), *preparing* (progress bar), *failed* (red dashed + message), *conflict*; `ReadinessBanner` ("N models need preparing · X GB" + **Prepare all**); inspector section "Weights & requirements" (layers, source, size, licence, target folder, **Prepare model**, Advanced override); Process disabled with a reason when any model is not ready; all texts in English, no model names in the front-end.

- [ ] **Step 1: Create `docs/task_0023.md`.**
- [ ] **Step 2: Write failing tests** (vitest): `missing card renders dashed warning style and an alert icon`, `ready card has neither`, `preparing card shows progress from SSE events`, `banner shows count and total size and triggers prepare all`, `inspector shows Prepare model for a missing model and nothing for a model with no requirements`, `process is disabled while any model is not ready and enabled when all are ready`, `unknown state value renders the generic style`, `409 from submit shows the list of models`.
- [ ] **Step 3: Run `npm test`** — expect FAIL.
- [ ] **Step 4: Implement** following `docs/superpowers/specs/mockups/model-readiness-mockup.html` (direction only); verify in headless Chrome against a loopback server with a fake prepare worker.
- [ ] **Step 5: Run `npm test`, `npx tsc -b`, `npm run build`** — green.
- [ ] **Step 6: Docs and commit** (`feat(web): add model readiness states and prepare actions`).

---

### Task 6: Packages layer with dry-run conflict detection (task_0024)

**Files:**
- Create: `src/det_depth_distance/core/provision/packages.py`, `tests/test_provision_packages.py`, `tests/web/test_models_packages.py`
- Modify: `src/det_depth_distance/core/provision/engine.py`, `src/det_depth_distance/web/models.py`, `web/src/models/ModelRequirements.tsx`

**Interfaces:**
- Consumes: Task 1 `Requirements.packages/extra`, engine, prepare queue.
- Produces: `detect_extras(pyproject: Path) -> set[str]` (extras whose declared packages are importable via `importlib.metadata`/`find_spec`), `dry_run(extra: str, detected: set[str], uv: str = "uv") -> DryRunResult(ok: bool, message: str)` running `uv sync --dry-run` with `--extra` for every detected extra plus the new one (never a bare sync), `install(extra, detected, ...)` running the same without `--dry-run`; new state `conflict` carries the resolver's reason; web action **Install packages** (explicit confirmation, runs as a prepare job, afterwards "restart the server"); base-stack missing → the UI shows the exact one-time command with a copy button instead of a Prepare button.

- [ ] **Step 1: Create `docs/task_0024.md`.**
- [ ] **Step 2: Write failing tests** with a fake `uv` executable (script that exits 0 or prints a resolver error): `test_dry_run_ok_offers_install`, `test_dry_run_conflict_reports_reason_and_changes_nothing` (snapshot the environment directory listing before/after), `test_install_command_includes_every_detected_extra`, `test_install_never_runs_bare_sync`, `test_base_stack_missing_is_reported_with_exact_command_not_installed`, `test_detect_extras_uses_import_names_without_importing`, `test_conflict_state_reaches_the_api_and_ui_contract`.
- [ ] **Step 3: Run** — expect FAIL.
- [ ] **Step 4: Implement**; the install job streams `uv` output into the prepare events.
- [ ] **Step 5: Run** python tests, `npm test`, build — green.
- [ ] **Step 6: Docs and commit** (`feat(core): detect package conflicts before installing`).

---

### Task 7: Documentation (task_0025)

**Files:**
- Create: `docs/model_environments.md`
- Modify: `README.md` (English section "Model weights" and a `TODO` entry), `DESCRIPTION.md`, `CHANGELOG.md`, `docs/web_extending.md`, `docs/task_0025.md`

**Interfaces:** none (documentation only); every example in the documents must be runnable or verified against the code.

- [ ] **Step 1: Create `docs/task_0025.md`.**
- [ ] **Step 2: Write `docs/model_environments.md`** (English): the model recipe format with a worked example of adding a model (a hypothetical Metric3D: subclass, `requirements()`, an extra in `pyproject.toml`, a registered module), the three layers, default locations and overrides, the status states, and **each dependency-conflict resolution path in detail** — (1) detect before installing *(implemented)*, (2) mutually exclusive extras via `[tool.uv] conflicts`, (3) one environment per profile used by the worker subprocess, (4) the conflicting model served as its own process, (5) manual pinning or forking — each with when to use it, cost, limits and how it would plug into this design.
- [ ] **Step 3: README**: "Model weights" section (clone, `uv sync --extra infer --extra web`, load a sample, click Prepare; `--prepare` on the CLI; `--weights-root`; gated models) and a `TODO` list linking paths 2–5 to the document.
- [ ] **Step 4: Verify** every command in the new documents by running it (or `--help`/dry-run where it needs the network), and run `ruff` plus the full test suite.
- [ ] **Step 5: Commit** (`docs: document model readiness and conflict resolution paths`).

---

## Self-Review

- **Spec coverage:** declaration per class with three layers and registered kinds (T1, T2); defaults under the weights root and sample YAMLs portable (T1, T3); torch-free status and states (T1, T4, T5); one UI action Prepare / Prepare all and visuals from the user's spec (T5); conflict detection v1 and install-after-dry-run (T6); `--prepare` CLI (T2); not-ready gating 409/CLI error/Process disabled (T4, T5); `UserProvided`, gated messages (T1, T2); English docs and `docs/model_environments.md` with all five paths, README TODO, AGENTS.md line (T1, T7); future-proof with no model names in the UI (T5 tests).
- **Left to the builder, per spec:** exact class and module names beyond the interfaces above, the libraries for sources, the dry-run parsing details, resume specifics, the real URLs/hashes/licences (recorded in `task_0021`).
- **Type consistency:** `Requirements`, `Asset` kinds, `ModelStatus`, `check`, `prepare`, `weights_root`, `PrepareQueue`, `DryRunResult` are named identically where consumed.
- **Proportion:** bodies are omitted; interfaces, route names and test names carry the decisions.
