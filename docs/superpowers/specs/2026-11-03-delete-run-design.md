# Delete a run from the history — design (small change)

## Intent
Let the user clean up old runs from the web interface, one at a time, to free
disk space on the workstation.

## Goals
- A delete button on each history row and on the run detail page.
- Confirmation dialog (existing `Dialog`) that shows the space the run uses.
- Permanent removal of the run folder (or, for a `job-<id>` entry, its record
  in `.web_jobs`).

## Non-goals
- Bulk / multi-select / "older than X" cleanup.
- Trash or undo (does not free space).

## Constraints
- Run ids go through `history.run_folder` / `safe_join` (no path traversal).
- A queued or running job's run cannot be deleted (409).
- Task file `docs/task_0032.md` (English) before code; branch `feat/delete_run`;
  update CHANGELOG.md and DESCRIPTION.md (new endpoints) in the same commit.

## Approaches
A. Permanent delete after confirmation, with size shown (chosen).
B. Move to a trash folder, purge later — rejected: frees no space, two features.
C. Delete without confirmation/size — rejected: too easy to misclick.

## Design
- `history.run_size(out_root, run_id) -> int` (bytes, sums regular files, no
  symlink following) and `history.delete_run(out_root, run_id)`.
- `GET /api/runs/{id}/size` -> `{"bytes": n}`; declared before the
  `/api/runs/{run_id}/{name}` file route. `DELETE /api/runs/{id}` -> 204.
  Both 404 on unknown ids; DELETE answers 409 while the run's job is
  queued/running (checked via the job queue and `web_run.json` status).
- `job-<id>`: size 0 / record file size; delete unlinks the JSON record.
- Symlinks inside a run folder are unlinked, never followed.
- Front end: trash button in `RunList` rows and `RunDetail`; dialog fetches
  size on open, confirm calls DELETE, then refreshes the list; if the deleted
  run was open or in the compare selection, it is closed/removed. A 409 shows
  an inline message.

## Left to the builder
Exact icon/styling, wording of messages, byte formatting (reuse `format.ts`).

## Verification
Backend tests (delete, 404, traversal ids, 409 on active job, job-record
delete, size); front-end tests in `RunsView.test.tsx`/`RunDetail.test.tsx`;
`uv run pytest`, ruff, and the web test suite.
