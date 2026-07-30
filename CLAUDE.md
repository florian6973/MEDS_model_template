# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

A [Copier](https://copier.readthedocs.io) **template** (not an application) that generates MEDS model
repositories exposing a shared six-command interface (`meds-model <command>`), contributable to
[MEDS-DEV](https://github.com/Medical-Event-Data-Standard/MEDS-DEV).

Two distinct codebases live here, and they are built/tested differently:

| | Path | Role |
|---|---|---|
| **The template repo** | `pyproject.toml`, `src/meds_model_template/`, `tests/`, `copier.yml` | A tiny bootstrap package (`meds-model-new` shells out to `copier copy`) plus the render tests. This is what `uv sync`/`ruff`/`pytest` at the repo root operate on. |
| **The Copier payload** | `template/` | The files that become a *generated* repo. `ruff` **excludes** this directory (`extend-exclude = ["template"]` in `pyproject.toml`) — it contains Jinja placeholders that aren't valid Python until rendered. |

README.md carries a warning that the repo is AI-generated and unreviewed; treat existing code as
unverified rather than as settled precedent.

**[`design-interface.md`](design-interface.md) is the authoritative spec** for the command graph, artifact
layout, manifests, and the arbitration/coverage rules. `docs/DESIGN.md` predates it and carries a banner
saying so — it is useful for background (why Copier, the verified ecosystem APIs) but its five-step CLI
vocabulary is obsolete.

## Commands

Template-repo development (repo root):

```bash
uv sync --group dev
uv run ruff check .                 # only lints src/ + tests/ — `template/` is excluded
uv run ruff format --check .
uv run pytest tests/ -q             # renders all profiles, ~seconds, no torch
uv run pytest tests/test_render.py::test_render_profile -k probe   # a single profile
```

Render a repo by hand and exercise it (mirrors the `rendered-smoke` CI job):

```bash
uvx copier copy --vcs-ref=HEAD --defaults --trust \
  --data model_slug=demo_model --data model_name="Demo Model" \
  --data profile=supervised_basic . /tmp/demo_model
cd /tmp/demo_model
uv venv && uv pip install torch --index-url https://download.pytorch.org/whl/cpu
uv pip install -e . --group dev
uv run pytest -m "not slow" -q      # `-m slow` runs the designed-signal learnability test
```

`--vcs-ref=HEAD` renders **committed** HEAD, so commit changes to `template/` before rendering this way.
`--trust` is needed because `copier.yml` declares post-copy `_tasks` (`git init`, best-effort `uvx ruff`).

Inside a generated repo, pytest runs with `--doctest-modules` over `src`, so docstring examples in
`meds_model_base` are executed as tests. `doctest_optionflags = ["ELLIPSIS", "NORMALIZE_WHITESPACE"]` is
set in the generated `pyproject.toml` — several doctests elide long error messages with `...` and will
fail without it.

If `copier`/`uv` are unavailable, `template/` can still be verified with a plain jinja2 render harness:
walk the tree, render `*.jinja` and path segments, then `compileall` the result and `yaml.safe_load` every
config. That catches broken Jinja, unused imports, and malformed YAML without installing torch.

## Architecture

### Ownership split inside a generated repo

- `src/meds_model_base/` — the **vendored, template-managed contract**: command ABCs and arbitration, the
  dispatcher, the manifest layer, default command implementations, schemas, profile `LightningModule`s,
  test helpers. Copied *verbatim* by Copier and re-rendered by `copier update`.
- `src/<model_slug>/` — the **user-owned surface**: `model.py`, `commands.py`, `configs/`. `model.py`,
  `commands.py`, `configs/model/**`, `configs/paths/**` and `configs/profile/**` are protected from
  `copier update` (`_skip_if_exists` in `copier.yml`); `__main__.py` and the other config groups are not.

### The command contract

`template/src/meds_model_base/commands/base.py` defines `CommandName` (six commands) and the
`MEDSModelCommand` ABC hierarchy. Each command class carries `name`, `config_name` (its Hydra root, named
after the command), `sources` (alternative input parameters), `supported_sources` (the subset this
implementation handles), and `require_source`.

`MEDSModelCommand.__call__` runs `validate()` — which arbitrates the sources — and caches the result on
`self.source` before dispatching to `run()`. **The dispatcher always invokes `__call__`, never `run`**, so
arbitration cannot be bypassed. Preserve that when adding commands.

A model declares support via `COMMANDS: dict[CommandName, type[MEDSModelCommand]]` in
`src/<model_slug>/commands.py`. This is the only thing that has to branch on the profile in Python,
because the dispatcher needs it before Hydra composes anything.

Adding or renaming a command touches: the ABC in `commands/base.py`, a `configs/<command>.yaml` root, the
`entries` table in `commands.py.jinja`, and `PROFILE_COMMANDS` + `ALL_COMMANDS` in `tests/test_render.py`.

### Manifests are read, not just written

`meds_model_base/manifest.py` is load-bearing, not bookkeeping. `write_artifact()` is a context manager
that stages into a temp sibling and renames into place, writing `manifest.yaml` last — so a visible
artifact is always complete. `read_manifest(dir, require_type=..., require_kind=...)` is how commands
reject a wrong-typed input *before* doing work.

Three places consume manifests for real behavior, not just provenance:

- `_runtime.load_trained_module` reads `module_class` from the model artifact, **not** `cfg.model._target_`
  — a checkpoint is loadable only by the class that wrote it.
- `preprocess_task` recovers the raw MEDS location from the `patients/` manifest for ACES extraction.
- `ProbePredictCommand` recovers which embeddings to score from the probe's own manifest, and
  `predict.task_definition_path()` recovers the ACES YAML from the task manifest for zero-shot models.

That last pattern is why `predict` needs no extra task-definition or embeddings argument: adding one would
invite exactly the mismatch the manifest exists to prevent.

### Import-weight discipline (load-bearing)

`commands/base.py`, `manifest.py` and `dispatch.py` are torch-free so `meds-model commands` and `--help`
stay cheap; `commands/__init__.py` uses a module-level `__getattr__` to lazily import the heavy default
implementations, and `dispatch._run_with_hydra` defers `register_structured_configs` until just before
Hydra composes. The tier-1 CLI smoke test depends on this. Don't hoist those imports.

### Schemas

`meds_model_base/schemas.py` is the single source of truth for column names — it re-exports canonical
`meds` / `meds_evaluation` classes rather than hardcoding strings. It intentionally omits
`from __future__ import annotations`: `flexible_schema` reads live `Required(...)` descriptors off class
annotations, and PEP 563 stringization breaks schema construction.

### Failure modes the contract deliberately catches

Each of these otherwise produces output that looks correct, so don't "simplify" them away:

- `predict` enforces **coverage** (`_check_coverage`) and records `n_expected`/`n_written` per split.
- `predict` **never reads ground truth** — `_runtime.load_index` drops `boolean_value`.
- `train.load_pretrained_weights` **raises when a warm start matches zero parameters**.
- `arbitrate_sources` rejects two sources and rejects an unsupported one; there is no precedence order.
- Zero-shot `resolve` is **abstract**. A placeholder returning a constant would yield a schema-valid
  predictions file of pure noise that every downstream check would pass.

### Profiles and Jinja gating

`copier.yml`'s `profile` question presets `implements_*` booleans (only *asked* for `custom`). Unlike the
previous design, those gate only `commands.py.jinja`, `model.py.jinja`, `configs/profile/default.yaml.jinja`
and `model.yaml.jinja` — every command module and config root ships unconditionally.

`commands.py.jinja` builds an `entries` list in Jinja and derives its import block from it, so a profile
can never import a class it doesn't register (which would fail ruff's F401 in the generated repo).

`_templates_suffix: .jinja` means **only** `.jinja` files are rendered; everything else under `template/`
is copied byte-for-byte (path segments like `{{ model_slug }}` are still substituted). Adding Jinja to a
file without the suffix silently ships literal braces.

### Test tiers in a generated repo

1. `test_cli_smoke.py` — `--help` exits 0 per supported command; `commands` lists exactly `COMMANDS`; an
   unsupported command fails clearly.
2. `test_smoke_pipeline.py` — end-to-end over `meds_testing_helpers` fixtures via subprocess. Which source
   each command receives is **derived from the registered class's `supported_sources`**, so a profile whose
   wiring and implementation disagree fails rather than silently exercising a different chain.
3. `test_property.py` (`@pytest.mark.slow`) — designed-signal learnability with a negative control.

`tests/test_render.py` here is a fourth, cheaper tier: render each profile, assert the registry matches,
assert every config root exists and parses, and `compileall` the rendered `src/`.

## Known gaps

- **ACES extraction is unverified.** `tasks.extract_with_aces` calls the documented `es-aces` API but has
  never been run against a live install; results are normalized through `normalize_label_columns` rather
  than assuming column names. The pre-materialized-labels path has no ACES dependency and is what the
  smoke tests exercise.
- **Zero-shot prediction is not covered end-to-end** by the smoke tests: it needs a task *definition*, and
  the `meds_testing_helpers` fixture supplies pre-materialized labels, so that leg is skipped explicitly.
- **No MEDS-DEV mapping has been validated.** `model.yaml.jinja` folds `preprocess_data` into the train
  commands and runs `preprocess_task` over the supplied `labels_dir`, but this has not been run against a
  live MEDS-DEV checkout. See the open questions in `design-interface.md`.
