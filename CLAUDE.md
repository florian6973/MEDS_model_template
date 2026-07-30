# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

A [Copier](https://copier.readthedocs.io) **template** (not an application) that generates MEDS model
repositories exposing a mandated five-step CLI (`meds-model <step>`), contributable to
[MEDS-DEV](https://github.com/Medical-Event-Data-Standard/MEDS-DEV).

Two distinct codebases live here, and they are built/tested differently:

| | Path | Role |
|---|---|---|
| **The template repo** | `pyproject.toml`, `src/meds_model_template/`, `tests/`, `copier.yml` | A tiny bootstrap package (`meds-model-new` shells out to `copier copy`) plus the render tests. This is what `uv sync`/`ruff`/`pytest` at the repo root operate on. |
| **The Copier payload** | `template/` | The files that become a *generated* repo. `ruff` **excludes** this directory (`extend-exclude = ["template"]` in `pyproject.toml`) — it contains Jinja placeholders that aren't valid Python until rendered. |

README.md carries a warning that the repo is AI-generated and unreviewed; treat existing code as
unverified rather than as settled precedent.

## Commands

Template-repo development (repo root):

```bash
uv sync --group dev
uv run ruff check .                 # only lints src/ + tests/ — `template/` is excluded
uv run ruff format --check .
uv run pytest tests/ -q             # renders all four profiles, ~seconds, no torch
uv run pytest tests/test_render.py::test_render_profile -k zero_shot_ar   # a single profile
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
`meds_model_base` (e.g. `schemas.py`, `dispatch.py`) are executed as tests — keep them correct.

## Architecture

### Ownership split inside a generated repo

- `src/meds_model_base/` — the **vendored, template-managed contract**: step ABCs, the dispatcher, default
  step implementations, schemas, profile `LightningModule`s, test helpers. Copied *verbatim* by Copier and
  re-rendered by `copier update`, so contract fixes propagate to downstream repos.
- `src/<model_slug>/` — the **user-owned surface**: `model.py`, `steps.py`, `__main__.py`, `configs/`.
  Only `model.py` and `configs/model/**` are protected from `copier update` (`_skip_if_exists` in
  `copier.yml`); `steps.py`, `__main__.py`, and the other config groups **are** overwritten on update.
  Changing that boundary means editing `_skip_if_exists`.

### The step contract

`template/src/meds_model_base/steps/base.py` defines `StepName` (the five steps, in pipeline order) and
the `MEDSModelStep` ABC hierarchy. Each step class carries two `ClassVar`s — `name` and `config_name`
(the Hydra root config, e.g. `_prediction`) — and one `run(cfg) -> Path`. Per-step ABCs declare the
override hooks (`build_module` for training, `predict` for prediction).

A model declares what it implements via `STEPS: dict[StepName, type[MEDSModelStep]]` in
`src/<model_slug>/steps.py`. `make_cli(STEPS, config_dir)` (`dispatch.py`) parses
`meds-model <step> [hydra overrides]`, handles `steps`/`--help`, and hands off to `hydra.main` using the
step's `config_name` against the model package's `configs/` dir (resolved via `importlib.resources`).

Adding or renaming a step touches four places in lockstep: the ABC in `steps/base.py`, a
`configs/_<step>.yaml` root, the `STEPS` entry in `steps.py.jinja`, and `PROFILE_STEPS` in
`tests/test_render.py`.

### Import-weight discipline (load-bearing)

`steps/base.py` and `dispatch.py` are deliberately torch-free so `meds-model steps` and `--help` stay
cheap; `steps/__init__.py` uses a module-level `__getattr__` to lazily import the torch-heavy default
step classes, and `dispatch._run_with_hydra` defers `register_structured_configs` (which imports
meds-torch-data) until just before Hydra composes. The tier-1 CLI smoke test in a generated repo depends
on this. Don't hoist those imports to module top level.

### Schemas

`meds_model_base/schemas.py` is the single source of truth for every column name — it re-exports the
canonical `meds` / `meds_evaluation` classes and constants rather than hardcoding strings. It
intentionally omits `from __future__ import annotations`: `flexible_schema` reads live `Required(...)`
descriptors off class annotations, and PEP 563 stringization breaks schema construction. It adds two
template-specific *open* schemas the standard lacks — `IndexSchema` and `TaskAgnosticOutputSchema` —
because `meds.LabelSchema` is closed.

Contract invariant: the `prediction` step never reads ground-truth labels. `load_index()` drops
`boolean_value` even when handed a MEDS-DEV `labels_dir`, and output is validated through
`PredictionSchema.align()` before writing `predictions.parquet`.

### Profiles and Jinja gating

`copier.yml`'s `profile` question presets the `implements_*` booleans (only *asked* when
`profile == custom`). Those booleans gate Jinja conditionals across `steps.py.jinja`, `model.py.jinja`,
`pyproject.toml.jinja`, the config roots, `model.yaml.jinja`, and the rendered tests. The four profiles
(`supervised_basic`, `zero_shot_ar`, `every_query`, `motor_finetune`) each map to a reference
`LightningModule` in `meds_model_base/profiles/`, which `src/<model_slug>/model.py` subclasses.

`_templates_suffix: .jinja` means **only** `.jinja` files are Jinja-rendered; everything else under
`template/` is copied byte-for-byte (path segments like `{{ model_slug }}` are still substituted). Adding
Jinja syntax to a file without the `.jinja` suffix silently ships the literal braces.

### Test tiers in a generated repo

1. `test_cli_smoke.py` — `meds-model <step> --help` exits 0 for each implemented step.
2. `test_smoke_pipeline.py` — end-to-end over `meds_testing_helpers` fixtures via subprocess CLI calls;
   the load-bearing assertion is `PredictionSchema.align()` on the output parquet.
3. `test_property.py` (`@pytest.mark.slow`) — designed-signal learnability with a negative control, backed
   by `meds_model_base/testing/{synthetic,property}.py`.

This repo's own `tests/test_render.py` is a fourth, cheaper tier: render each profile, assert the expected
files exist, assert `STEPS` registers exactly the profile's steps, and `compileall` the rendered `src/`.

## docs/DESIGN.md is aspirational in places

`docs/DESIGN.md` is the design rationale and is useful for *intent* (step semantics, the (d)-vs-(e)
distinction, MEDS-DEV integration, verified ecosystem API facts). But its repository-layout section
describes files that do not exist in the tree: `aces_labels.py`, `meds_model_base/configs/`,
`pipelines/preprocess.yaml`, `lightning/{datamodule,writer}.py`, `testing/fixtures.py`, the
`SearchPathPlugin`, and a `meds-model export-meds-dev` helper. In the current implementation all Hydra
configs live under `src/{{ model_slug }}/configs/`. Check the tree before acting on that document.
