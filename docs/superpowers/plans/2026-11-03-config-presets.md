# Config Presets Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let the web editor load a named pipeline preset from a list (the `configs/*.yaml` files plus the user's own) and save the current pipeline under a name on the server.

**Architecture:** A new `web/presets.py` holds all preset logic (manifest, slug, list/read/save/delete) with no FastAPI import; four thin routes in `web/app.py` call it. The front-end gets an injected `PresetsApi` client (same pattern as `videoApi`/`runsApi`/`modelsApi`), a `presets/` folder with the menu and the save dialog, and one new store method that loads a preset while keeping the editor's own `source` and `output`.

**Tech Stack:** Python 3 + FastAPI + PyYAML + pytest (`tests/web/`, `web` extra); React 18 + TypeScript + vitest (`web/`).

**Spec:** `docs/superpowers/specs/2026-11-03-config-presets-design.md`

## Global Constraints

- A preset holds the pipeline only: `source` and `output` (whole sections) are removed on save and never touched on load; `device` stays in the preset.
- Files in `configs/` are never written, renamed or deleted; built-in ids (`builtin:<stem>`) answer 403 on delete.
- A client name becomes a file name only through `slugify` and `web.paths.safe_join`; slug charset `[a-z0-9-]`, at most 60 chars; typed name at most 80 chars, no control characters.
- Saved YAML is checked with `schema.validate_config` before writing and limited to `MAX_YAML_CHARS` (`web/app.py`).
- User presets live in `<presets-root>/<slug>.yaml`, default `_testbed/presets/`, option `--presets-root`; first line `# name: <typed name>`.
- Preset ids: `builtin:<file stem>` and `user:<slug>`.
- Documentation and the task file are in English; no commit on `main`; `DESCRIPTION.md`, `CHANGELOG.md` and the task file are updated in the same commit as the code they describe (task-by-task here, final state checked in Task 5).
- Commit messages end with `Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>`.
- Test commands: `uv run --extra web pytest tests/web/<file> -v` (add `--extra infer` if the GPU stack is installed in that environment, or `uv sync` removes it); front-end: `cd web && npm test` and `npm run typecheck`.

## Review Focus

Failure modes the spec implies; each has a test in the task named in brackets.

1. A name that slugs to nothing (`"!!!"`, `"   "`) must be rejected with a clear message and write no file. [Task 1, 2]
2. A preset id or name carrying `../x`, an absolute path, an encoded `%2e%2e` or a symlink must never read or write outside the presets root. [Task 1, 2]
3. One hand-edited invalid file in the user folder must not hide the other presets: it is listed, and loading it reports its error. [Task 1, 2]
4. Loading a preset into an editor that has no `source`/`output` yet (fresh editor) must not crash or invent them. [Task 3]
5. Deleting a preset that is already gone (second tab), or a missing presets folder, must not break the list: the list refreshes and the folder is created on first save. [Task 1, 4]

---

### Task 0: Branch and task file

**Files:**
- Create: `docs/task_0030.md` (from `docs/task_TEMPLATE.md`, in English)
- Add (already written): `docs/superpowers/specs/2026-11-03-config-presets-design.md`, this plan

- [ ] **Step 1:** `git switch -c feat/config_presets` from the current branch (it carries the `run_web.sh` commit; if that branch was merged, branch from `main`).
- [ ] **Step 2:** Create `docs/task_0030.md` following the template: Description, Scope (files table from Tasks 1–5), Plan (point to this plan), Impact analysis, Verification, Dependencies. Title without `(COMPLETED)` yet.
- [ ] **Step 3:** Commit.

```bash
git add docs/task_0030.md docs/superpowers
git commit -m "docs: add task_0030 plan for config presets"
```

---

### Task 1: Preset logic and manifest (backend, no HTTP)

**Files:**
- Create: `src/det_depth_distance/web/presets.py`
- Create: `configs/presets.yaml`
- Test: `tests/web/test_presets.py`

**Interfaces:**
- Produces (used by Task 2):
  - `EXCLUDED_KEYS: tuple[str, ...] = ("source", "output")`
  - `BUILTIN_DIR: Path` (the checkout's `configs/`, `Path(__file__).resolve().parents[3] / "configs"`)
  - `@dataclass(frozen=True, slots=True) class PresetInfo: id: str; name: str; description: str; kind: Literal["builtin", "user"]`
  - `class PresetNotFound(LookupError)`, `class PresetExists(Exception)`, `class PresetReadOnly(Exception)`, `class PresetInvalid(ValueError)` (bad name or id; message is shown to the user)
  - `slugify(name: str) -> str` — NFKD ascii-fold, lowercase, runs of non `[a-z0-9]` become `-`, trimmed, cut to 60; raises `PresetInvalid` if empty or name > 80 chars or has control characters.
  - `list_presets(builtin_dir: Path, user_root: Path | None) -> list[PresetInfo]` — built-ins in manifest order, then user presets sorted by name.
  - `read_preset(preset_id: str, builtin_dir: Path, user_root: Path | None) -> dict[str, Any]` — the YAML mapping with `EXCLUDED_KEYS` removed; raises `PresetNotFound`, `PresetInvalid` (unparsable YAML or not a mapping).
  - `save_preset(user_root: Path, builtin_dir: Path, name: str, data: Mapping[str, Any], overwrite: bool) -> PresetInfo` — strips `EXCLUDED_KEYS`, writes `# name: <name>` + `yaml.safe_dump(sort_keys=False, allow_unicode=True)`; raises `PresetInvalid` (bad name, or slug equal to a built-in stem or to the slug of a built-in name), `PresetExists` (file exists and not `overwrite`).
  - `delete_preset(preset_id: str, user_root: Path | None) -> None` — raises `PresetReadOnly` for `builtin:`, `PresetNotFound` if absent.

- [ ] **Step 1: Write the failing tests** in `tests/web/test_presets.py` (use `tmp_path` for both dirs; build a tiny builtin dir with a manifest and two YAMLs):
  - `test_slugify_cases`: `"RF-DETR + DA3"` → `"rf-detr-da3"`, `"Minha Câmera 4K"` → `"minha-camera-4k"`; `"!!!"`, `"   "`, `""`, `"a\nb"`, 81-char name raise `PresetInvalid`.
  - `test_list_builtin_follows_manifest_order_and_names`, `test_list_without_manifest_falls_back_to_stems` (and skips `presets.yaml`), `test_list_ignores_manifest_entry_whose_file_is_missing`.
  - `test_read_builtin_strips_source_and_output` (keeps `device`, `detectors`, `depth`).
  - `test_save_then_list_and_read_roundtrip` (name shown from the `# name:` comment; file is `<slug>.yaml`; saved file has no `source`/`output`).
  - `test_save_creates_missing_root`; `test_save_existing_without_overwrite_raises_exists`; `test_save_with_overwrite_replaces`.
  - `test_save_rejects_builtin_stem_and_builtin_name_slug`.
  - `test_invalid_user_file_is_listed_and_read_raises_invalid_without_hiding_others` (Review Focus 3).
  - `test_ids_with_traversal_are_rejected`: `read_preset("user:../x", ...)`, `"user:/etc/passwd"`, `"user:a/b"`, and a symlink `<root>/evil.yaml -> outside` raise `PresetNotFound`/`PresetInvalid` and never open the target (Review Focus 2).
  - `test_delete_user_removes_file`, `test_delete_builtin_raises_read_only`, `test_delete_missing_raises_not_found` (Review Focus 5).
- [ ] **Step 2:** Run `uv run --extra web pytest tests/web/test_presets.py -v` → FAIL (module not found).
- [ ] **Step 3:** Implement `presets.py` per the Interfaces block. Use `web.paths.safe_join(user_root, f"{slug}.yaml")` for every user path and reject a resolved path that is a symlink; write new files with mode `"x"` (exclusive) and overwrites via a temp file in the same folder plus `os.replace`; `yaml.safe_load` for reads; log (module `logging`) and fall back to stems if the manifest is malformed.
- [ ] **Step 4:** Create `configs/presets.yaml` with a top-level `presets:` list of `{file, name, description}`, one entry for each of the 16 configs, using the names and descriptions of the spec table, in the spec's order.
- [ ] **Step 5:** Add `test_repo_manifest_lists_every_config_and_every_file_exists` (against the real `configs/`): the set of manifest files equals the set of `configs/*.yaml` minus `presets.yaml`, each file exists, and `read_preset` on each returns a mapping with `detectors` and no `source`/`output`. Run the whole file → PASS.
- [ ] **Step 6:** Commit.

```bash
git add src/det_depth_distance/web/presets.py configs/presets.yaml tests/web/test_presets.py
git commit -m "feat(web): add preset storage with a names manifest"
```

---

### Task 2: HTTP routes and `--presets-root`

**Files:**
- Modify: `src/det_depth_distance/web/paths.py` (`WebPaths` gains `presets_root: Path | None = None`)
- Modify: `src/det_depth_distance/web/app.py` (routes; `create_app` gains `builtin_presets_dir: Path = presets.BUILTIN_DIR`; extract the config/extras split of `/api/config/parse` into `_split_config(data) -> dict[str, dict]` and reuse it)
- Modify: `src/det_depth_distance/web/server.py` (`--presets-root`, default `Path("_testbed/presets")`)
- Test: `tests/web/test_api_presets.py`

**Interfaces:**
- Consumes: Task 1 functions and exceptions.
- Produces (used by Task 3):
  - `GET /api/presets` → `[{"id","name","description","kind"}]`
  - `GET /api/presets/{preset_id}` → `{"config": {...}, "extras": {...}}` (same shape as `/api/config/parse`, no `source`/`output`); 404 unknown, 422 invalid file with `detail` message.
  - `PUT /api/presets` with body `SaveBody{name: str, config: dict, extras: dict = {}, overwrite: bool = False}` → `{"id","name","description","kind"}`; 422 `{detail: [{path,message}]}` or message string for an invalid name/config, 409 for `PresetExists` and for a built-in clash, 413 over `MAX_YAML_CHARS`, 404 when `paths.presets_root is None`.
  - `DELETE /api/presets/{preset_id}` → 204; 403 builtin; 404 missing.
  - The `tests/web/conftest.py` `client` fixture keeps working: `presets_root` defaults to `None` there, so add a fixture `preset_client` in the new test file that builds `WebPaths(..., presets_root=tmp_path / "presets")` and `create_app(paths, builtin_presets_dir=<tmp builtin dir>)`.

- [ ] **Step 1: Write the failing tests** in `tests/web/test_api_presets.py`:
  - `test_list_returns_builtin_and_user`, `test_list_works_when_presets_root_missing_or_none`.
  - `test_get_builtin_strips_source_output_and_splits_extras`.
  - `test_put_saves_and_get_roundtrips_without_source_output` (send a config containing `source: {video: x}` and `output: {out_root: y}`; the file on disk has neither).
  - `test_put_empty_slug_name_422_and_no_file` (`"!!!"`; Review Focus 1).
  - `test_put_invalid_config_422_lists_errors` (config `{}`), `test_put_existing_409_then_overwrite_200`, `test_put_builtin_name_409`, `test_put_oversize_413`.
  - `test_get_and_delete_traversal_ids_never_leave_root`: ids `user:..%2Fx`, `user:%2e%2e`, absolute path → 404/422 and an outside canary file is untouched (Review Focus 2).
  - `test_delete_user_204_then_404`, `test_delete_builtin_403`.
  - `test_invalid_user_file_listed_and_get_422` (Review Focus 3).
  - `test_state_changing_routes_require_local_origin`: PUT/DELETE with a foreign `Origin` header are rejected like the other routes (copy the existing check from `test_api_videos.py` or `test_api_runs.py`).
- [ ] **Step 2:** Run `uv run --extra web pytest tests/web/test_api_presets.py -v` → FAIL.
- [ ] **Step 3:** Implement the routes, mapping exceptions to the status codes above; the PUT validates with `schema.validate_config(merged_without_excluded)` and measures `len(yaml_text)` against `MAX_YAML_CHARS`. Wire `--presets-root` into `WebPaths` in `server.py` and add it to the module docstring of `app.py` next to the other route groups.
- [ ] **Step 4:** Run `uv run --extra web pytest tests/web -v` → PASS (all existing web tests still pass).
- [ ] **Step 5:** Update `DESCRIPTION.md` (endpoints, `--presets-root`, `configs/presets.yaml`) and add the `CHANGELOG.md` entry (Unreleased, backend part).
- [ ] **Step 6:** Commit.

```bash
git add src tests/web DESCRIPTION.md CHANGELOG.md
git commit -m "feat(web): add preset routes and the --presets-root option"
```

---

### Task 3: Front-end client and store method

**Files:**
- Create: `web/src/api/presets.ts`
- Modify: `web/src/editor/yamlState.ts` (new `loadPreset`)
- Test: `web/src/api/presets.test.ts`, `web/src/editor/yamlState.test.ts` (add cases)

**Interfaces:**
- Consumes: the routes of Task 2.
- Produces (used by Task 4):
  - `export interface PresetInfo { id: string; name: string; description: string; kind: "builtin" | "user" }`
  - `export interface PresetsApi { list(): Promise<PresetInfo[]>; get(id: string): Promise<ParsedConfig>; save(name: string, config: ConfigDict, extras: ConfigDict, overwrite: boolean): Promise<PresetInfo>; remove(id: string): Promise<void> }`
  - `export function createPresetsApi(base = "", fetchImpl: typeof fetch = (...a) => fetch(...a)): PresetsApi` — reuses `ApiError` from `./client`; ids go through `encodeURIComponent`; a 409 surfaces as `ApiError` with `status === 409` (the dialog branches on it).
  - Store: `loadPreset(parsed: ParsedConfig): void` on the object returned by `createEditorStore` — sets the pipeline to `parsed.config` plus the current pipeline's `source` and `output` when they exist (never creates them), sets `baseline` to a clone, `dirty: false`, `fileName: null`, clears selection/errors/notice, schedules validation (same as `loadParsed`).

- [ ] **Step 1: Write the failing tests:**
  - `presets.test.ts` with a fake `fetch`: `list` GETs `/api/presets`; `get("user:a b")` requests `/api/presets/user%3Aa%20b`; `save` PUTs JSON `{name, config, extras, overwrite}`; `remove` sends DELETE and accepts 204; a 409 body `{detail: "exists"}` rejects with `ApiError` status 409 and message `exists`.
  - `yamlState.test.ts`: `loadPreset keeps the current source and output` (editor loaded with `source.video = "a.mp4"`, `output.out_root = "o"`; preset config has other `source`/`output` values and a new `detectors` → result has the editor's `source`/`output` and the preset's `detectors`); `loadPreset on a fresh editor adds no source or output` (Review Focus 4); `loadPreset leaves the editor clean (dirty false)`.
- [ ] **Step 2:** `cd web && npm test -- presets yamlState` → FAIL.
- [ ] **Step 3:** Implement `createPresetsApi` and `loadPreset`.
- [ ] **Step 4:** `npm test` and `npm run typecheck` → PASS.
- [ ] **Step 5:** Commit.

```bash
git add web/src
git commit -m "feat(web): add the presets client and the loadPreset editor action"
```

---

### Task 4: Presets menu, save dialog and delete

**Files:**
- Create: `web/src/presets/usePresets.ts`, `web/src/presets/PresetsMenu.tsx`, `web/src/presets/SavePresetDialog.tsx`
- Modify: `web/src/App.tsx` (new `presetsApi?: PresetsApi` prop defaulting to `createPresetsApi()`; "Presets" and "Save as preset…" buttons beside `Open YAML`/`Save YAML`; `openPreset` reuses the dirty confirmation text of `openFile`: `Discard the unsaved changes and open preset <name>?`)
- Modify: `web/src/ui/icons.tsx` only if a needed icon is missing (reuse existing names first)
- Test: `web/src/presets/Presets.test.tsx`

**Interfaces:**
- Consumes: `PresetsApi`, `PresetInfo` (Task 3), `store.loadPreset`, `ApiError`.
- Produces: `usePresets(api: PresetsApi): { presets: PresetInfo[]; status: "loading" | "ready" | "error"; error: string | null; refresh(): Promise<void> }`; `<PresetsMenu presets onPick(id) onDelete(id) />` (two groups, "Built-in" and "Mine", each row shows name and description; trash button only on `kind === "user"`); `<SavePresetDialog open existingNames onSave(name, overwrite) onCancel />`.

- [ ] **Step 1: Write the failing tests** in `Presets.test.tsx` (fake `PresetsApi`, render `App` or the components, same helpers as `Editor.test.tsx`):
  - `lists built-in and user presets in two groups with name and description`.
  - `picking a preset loads it and keeps the video and output of the editor`.
  - `picking a preset with unsaved changes asks first and does nothing when declined`.
  - `save as preset asks for a name; a 409 offers overwrite or another name, and overwrite sends overwrite=true`.
  - `an empty or symbols-only name shows the server message and keeps the dialog open`.
  - `delete asks for confirmation, removes the row, and a 404 (already gone) just refreshes the list` (Review Focus 5).
  - `built-in rows have no delete button`.
  - `the list shows an error line with a retry when listing fails`.
- [ ] **Step 2:** `cd web && npm test -- Presets` → FAIL.
- [ ] **Step 3:** Implement the hook and components; the menu is a popover opened from the "Presets" button (follow the existing dropdown near `App.tsx:419` for markup and keyboard handling); after a successful save call `refresh()` and `editor.notify({tone: "info", text: "Saved preset <name>."})`.
- [ ] **Step 4:** `npm test`, `npm run typecheck`, `npm run build` → PASS.
- [ ] **Step 5:** Commit.

```bash
git add web/src
git commit -m "feat(web): add the presets menu, save dialog and delete"
```

---

### Task 5: Docs, end-to-end check and task closing

**Files:**
- Modify: `README.md` (one short paragraph in the web section), `docs/reference.md` (section on presets: manifest format, `--presets-root`, what is and is not saved), `docs/web_extending.md` (only if it lists the endpoints), `DESCRIPTION.md` and `CHANGELOG.md` (front-end part), `AGENTS.md` only if it lists `configs/` contents or the file layout
- Modify: `docs/task_0030.md` (mark `(COMPLETED)`, add `Implementation details` and `Test notebook`)

- [ ] **Step 1:** Update the docs above; add a note in `configs/presets.yaml`'s header comment that a new config in `configs/` must be added there (the Task 1 manifest test enforces it).
- [ ] **Step 2:** Run the full suites: `uv run --extra web pytest` and `cd web && npm test && npm run typecheck && npm run build` → all PASS; `uv run ruff check --fix . && uv run ruff format .` → clean.
- [ ] **Step 3:** Manual check with `utils/run_web.sh`: open the Presets list, load "Light (4 GB GPU)" with a video already selected (video unchanged), change a field, "Save as preset…" under a new name, reload the page and load it, overwrite it, delete it; confirm `_testbed/presets/` holds a YAML without `source`/`output` and that `configs/` is unchanged (`git status`).
- [ ] **Step 4:** Close `docs/task_0030.md` and commit everything pending.

```bash
git add -A docs README.md DESCRIPTION.md CHANGELOG.md AGENTS.md configs/presets.yaml
git commit -m "docs: document config presets and close task_0030"
```
