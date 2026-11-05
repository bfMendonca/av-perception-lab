# Contributing

Thanks for looking. This is a research prototype with one maintainer, so small and focused
changes are the easiest to review. The rules below are the short version of
[`AGENTS.md`](AGENTS.md) (in Portuguese), which is the full reference.

## Set up

Python 3.11 is installed by [uv](https://docs.astral.sh/uv/); you also need ffmpeg and, for
the web interface, Node.js 22.12 or newer.

```bash
uv sync                      # light environment: no GPU stack, enough for the tests
uv run pytest -q             # unit and smoke tests, about a minute, no GPU
uv run ruff check . && uv run ruff format --check .
```

If you have installed the GPU stack, always pass `--extra infer --extra web` to `uv sync` and
`uv run`: a plain `uv sync` removes the GPU stack. The web interface tests live in `tests/web/`
and the front end in `web/` (`cd web && npm ci && npm test && npm run typecheck`).

Tests marked `gpu` need a GPU and downloaded weights; they are skipped by default.

## Making a change

1. **Branch from `main`.** Never commit to `main`. Name the branch `feat/<snake_case>`,
   `fix/<snake_case>` or `bug/<snake_case>`.
2. **Write the task file first.** Every implementation has a `docs/task_DDDD.md` (copy
   `docs/task_TEMPLATE.md`: description, scope, plan, impact analysis, verification,
   dependencies). New task files and new documentation are in English. When the work is done,
   mark the title `(COMPLETED)` and add `Implementation details` and a `Test notebook`, in the
   same commit as the code.
3. **Test it.** New behaviour comes with a test that fails without the change.
4. **Update the docs that describe it.** `DESCRIPTION.md` changes in the same commit when you
   change models, endpoints, dependencies, middleware, authentication or the main sections of
   the interface. Add an entry to `CHANGELOG.md` for anything users will notice.
5. **Commit with Conventional Commits**, subject in English, at most 50 characters:
   `feat(web): delete a run from the history (task_0032)`. The types and scopes are listed in
   `AGENTS.md`.
6. **Open a pull request** against `main` with what changed, why, and how you verified it.

## Adding a model

A model is a registered class that declares where its weights come from. You do not edit the
web interface: see [`docs/model_environments.md`](docs/model_environments.md) and
[`docs/web_extending.md`](docs/web_extending.md).

## Reporting problems

Open an issue with the config you used (`config.yaml` from the run folder), the command and
the output. For security problems, follow [`SECURITY.md`](SECURITY.md) instead.
