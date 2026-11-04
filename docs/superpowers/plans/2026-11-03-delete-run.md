# Delete a Run Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let the user delete one run at a time from the web history, with a confirmation that shows the space it frees.

**Architecture:** Two history helpers (`run_size`, `delete_run`) do the filesystem work behind `safe_join`; two routes (`GET /api/runs/{id}/size`, `DELETE /api/runs/{id}`) expose them and refuse active runs via a new `JobQueue.holds_run`; the front end adds `size`/`remove` to `RunsApi`, a `DeleteRunDialog`, and trash buttons in `RunList` and `RunDetail`.

**Tech Stack:** Python 3 / FastAPI / pytest (`tests/web/`), React + TypeScript + Vitest (`web/`).

**Spec:** `docs/superpowers/specs/2026-11-03-delete-run-design.md`

## Global Constraints

- Branch `feat/delete_run`; never commit to `main`.
- `docs/task_0032.md` (English, from `docs/task_TEMPLATE.md`) is written **before** code; mark `(COMPLETED)` + `Implementation details` + `Test notebook` in the implementation commit.
- Run ids resolve only through `history.run_folder` / `safe_join`; symlinks inside a run are unlinked, never followed.
- Active (queued/running) run: DELETE answers 409.
- Update `CHANGELOG.md` and `DESCRIPTION.md` (new endpoints) in the same commit as the implementation.
- Routes `/api/runs/{id}/size` must be declared before the `/api/runs/{run_id}/{name}` file route (`app.py` ~line 817).
- Lint: `uv run ruff check --fix . && uv run ruff format .`; tests: `uv run pytest tests/web`, `npm test` in `web/`.

## Review Focus

- Run id `../x`, `.web_jobs`, `a/b`, absolute path → 404 on both routes, nothing deleted.
- Symlink inside a run folder pointing outside `out_root` → target untouched after delete; size ignores it.
- Run whose job is still queued/running → 409, folder intact.
- `job-<id>` entry (no folder) → delete removes only its `.web_jobs/<id>.json`; the entry disappears from `/api/runs`.
- Deleting the open run or the comparison run → UI leaves the dead route (`onRoute(null, …)` / compare cleared) instead of showing "Run not found".

---

### Task 1: Branch, task file, history helpers

**Files:**
- Create: `docs/task_0032.md`
- Modify: `src/av_perception_lab/web/history.py`
- Test: `tests/web/test_history.py`

**Interfaces:**
- Produces: `history.run_size(out_root: Path, run_id: str) -> int` (bytes; regular files only, symlinks not followed or counted; for `job-<id>` the record file size; errors as `load_run`/`run_folder`: `ValueError` invalid id, `FileNotFoundError` unknown).
- Produces: `history.delete_run(out_root: Path, run_id: str) -> None` (same errors; folder → `shutil.rmtree`, which unlinks symlinks without following; `job-<id>` → unlink `.web_jobs/<id>.json`).

- [ ] **Step 1:** `git switch -c feat/delete_run`; create `docs/task_0032.md` from the template (English; Scope = this plan's goal; Plan = these tasks).
- [ ] **Step 2: Write failing tests** in `tests/web/test_history.py`: `test_run_size_sums_regular_files` (two files of 10 and 5 bytes → 15), `test_run_size_ignores_symlinks` (symlink to a 1000-byte file outside → size unchanged), `test_delete_run_removes_folder_not_symlink_target` (outside file still exists), `test_delete_job_record` (failed job JSON with no folder → file gone, `scan_runs` empty; write the JSON as `test_history` already does for job entries), `test_run_ops_reject_bad_ids` (parametrized `"../x"`, `".web_jobs"`, `"a/b"`, `""` → `ValueError`; unknown name → `FileNotFoundError`).
- [ ] **Step 3:** Run `uv run pytest tests/web/test_history.py -q` → new tests FAIL (`AttributeError`).
- [ ] **Step 4: Implement** both functions in `history.py` using `run_folder` for folders and the `job-` branch logic from `load_run` for records (check the record with `_job_record` so only listed job entries are touchable). Sum with `os.walk(followlinks=False)` + `is_symlink()` skip.
- [ ] **Step 5:** Run the same command → PASS.
- [ ] **Step 6:** Commit `feat(web): history helpers to size and delete a run (task_0032)`.

### Task 2: Active-run guard and routes

**Files:**
- Modify: `src/av_perception_lab/web/jobs.py` (add method near `active`, ~line 158)
- Modify: `src/av_perception_lab/web/app.py` (next to `get_run`, ~line 648)
- Test: `tests/web/test_api_runs.py`, `tests/web/test_jobs.py`

**Interfaces:**
- Consumes: `history.run_size`, `history.delete_run` (Task 1).
- Produces: `JobQueue.holds_run(self, folder: Path) -> bool` — True when a job not in `TERMINAL` has `run_dir` resolving to `folder` (compare resolved paths).
- Produces: `GET /api/runs/{run_id}/size` → `{"bytes": int}`; `DELETE /api/runs/{run_id}` → 204. Both 404 for invalid/unknown ids (reuse the `run_folder_or_404` pattern but also accept `job-<id>`); DELETE → 409 `detail` "run is still queued or running" when `holds_run`.

- [ ] **Step 1: Write failing tests.** `test_jobs.py::test_holds_run`: submit a job through the existing fake-worker fixtures, assert `holds_run(Path(job.run_dir))` is True while non-terminal and False for another folder and after the job ends. `test_api_runs.py` (using the `api` fixture and `run_folders.make_run` to create folders under `web_paths.out_root`): `test_size_route` (200, `{"bytes": n}`), `test_delete_route_removes_run` (204 then `GET /api/runs/{id}` is 404 and the id is absent from `/api/runs`), `test_delete_traversal_404` (`/api/runs/..%2Fx`, `/api/runs/.web_jobs` → 404, a sibling file outside `out_root` still exists), `test_delete_active_run_409` (start a slow fake job, DELETE its run folder id → 409, folder exists), `test_delete_job_entry` (failed job record → 204, entry gone).
- [ ] **Step 2:** `uv run pytest tests/web/test_jobs.py tests/web/test_api_runs.py -q` → new tests FAIL.
- [ ] **Step 3: Implement** `holds_run` in `jobs.py`; the two routes in `app.py` (catch `ValueError`/`FileNotFoundError` → 404, same message style as `get_run`). Declare the `/size` route above the `/{run_id}/{name}` file route.
- [ ] **Step 4:** Same command → PASS; then `uv run pytest tests/web -q` → all PASS.
- [ ] **Step 5:** Commit `feat(web): size and delete routes for runs (task_0032)`.

### Task 3: Front end

**Files:**
- Modify: `web/src/api/runs.ts` (`RunsApi` + `createRunsApi`), `web/src/test/fixtures.ts` (fake `RunsApi`/`VideoApi` gain `size`/`remove`)
- Create: `web/src/runs/DeleteRunDialog.tsx`
- Modify: `web/src/runs/RunList.tsx`, `web/src/runs/RunDetail.tsx`, `web/src/runs/RunsView.tsx`
- Test: `web/src/runs/RunsView.test.tsx`, `web/src/runs/RunDetail.test.tsx`

**Interfaces:**
- Produces: `RunsApi.size(runId: string): Promise<number>` (bytes) and `RunsApi.remove(runId: string): Promise<void>` (DELETE; rejects `ApiError`, 409 for an active run; 204 has no body, so do not `.json()` it — mirror `presets.ts` delete at line ~57).
- Produces: `DeleteRunDialog({ open, run: RunRecord, api: RunsApi, onDeleted: () => void, onCancel: () => void })` built on `ui/Dialog`: on open calls `api.size`, shows "Delete <runTitle(run)> permanently? This frees <size>." (use a byte formatter added to `runs/format.ts`: `formatBytes(n: number): string`, e.g. `1.2 GB`), Cancel / Delete (danger) buttons; on failure shows the error message in the dialog and stays open; `job-<id>` entries skip the size call.
- Consumes: `RunListProps` gains `onDelete: (run: RunRecord) => void`; `RunDetailProps` gains the same; trash `icon-btn` (`Icon name="trash"`, `aria-label="Delete <title>"`) added beside the compare button in `RunList` and in `Actions` of `RunDetail`.

- [ ] **Step 1: Write failing tests** (`RunsView.test.tsx` style): `deletes a run after confirming` (click `Delete <title>`, dialog shows formatted size, click Delete → `remove` called with the id, list refreshed without it), `cancel leaves the run`, `shows the server error and stays open on 409`, `deleting the selected run clears the route` (`onRoute` called with `(null, …)`), `deleting the compare run clears only the comparison` (`onRoute(runId, null)`), `formatBytes` unit cases (`0`→`0 B`, `1536`→`1.5 KB`, `1.2e9`→`1.2 GB`).
- [ ] **Step 2:** `cd web && npm test -- --run` → new tests FAIL.
- [ ] **Step 3: Implement** the API methods, `formatBytes`, the dialog, the buttons, and the state in `RunsView` (`pendingDelete: RunRecord | null`; on success: `refresh()`, and route fix-up per the two tests above).
- [ ] **Step 4:** `npm test -- --run` and `npm run build` (or the project's typecheck script in `package.json`) → PASS. Rebuild `web/dist` only if the repo tracks it (check `git status`).
- [ ] **Step 5:** Commit `feat(web): delete button for runs (task_0032)`.

### Task 4: Docs and final verification

**Files:**
- Modify: `CHANGELOG.md`, `DESCRIPTION.md` (endpoint list), `docs/task_0032.md` (mark `(COMPLETED)`, add `Implementation details`, `Test notebook`), `docs/web_extending.md` only if it lists run routes.

- [ ] **Step 1:** Add the CHANGELOG entry and the two endpoints in DESCRIPTION.md; complete the task file.
- [ ] **Step 2:** `uv run ruff check --fix . && uv run ruff format . && uv run pytest -q` and `cd web && npm test -- --run` → all PASS; report the real output.
- [ ] **Step 3:** Manual check: start the server (`run` skill / README command) against a scratch `out_root`, delete a run in the browser, confirm the folder is gone from disk.
- [ ] **Step 4:** Amend into the implementation commit per task-file rule (docs in the same commit as the implementation: squash Tasks 1–4 commits on the branch before the PR, or keep them and make the last commit carry the docs — ask the user's preference only if unsure), then open a PR (no direct merge to `main`).
