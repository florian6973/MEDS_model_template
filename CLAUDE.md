# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

A [Copier](https://copier.readthedocs.io) **template** (not an application) that generates MEDS model
repositories exposing a shared five-command interface (`meds-model <command>`), contributable to
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
  --data profile=supervised . /tmp/demo_model
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

**Run the rendered suite for real when you change the payload.** The fast `tests/` job never installs
torch, so it can only check structure; several bugs are only reachable by actually running a generated
repo (a relative import in a test module, a task layout the reader does not handle). It takes ~1 minute
per profile with a warm uv cache:

```bash
uv run python -c "
from copier import run_copy
run_copy('.', '/tmp/demo', data={'model_slug':'demo_model','model_name':'Demo','profile':'probe'},
         defaults=True, unsafe=True, quiet=True)"
cd /tmp/demo && uv venv \
  && uv pip install torch --index-url https://download.pytorch.org/whl/cpu \
  && uv pip install -e . --group dev \
  && uv run pytest -m "not slow" -q -rs
```

Expect ~29 passed and 1 skipped: the skip is `skip_if_stub`, and it is the correct result — everything
that does not need a model runs, including real MTD tensorization and task materialization.

If `copier`/`uv` are unavailable, `template/` can still be checked with a plain jinja2 render harness:
walk the tree, render `*.jinja` and path segments, then `compileall` the result and `yaml.safe_load` every
config. That is strictly weaker — it is what missed both bugs above.

## Architecture

### Ownership split inside a generated repo

- `src/meds_model_base/` — the **vendored, template-managed contract**: command ABCs and arbitration, the
  dispatcher, the manifest layer, default command implementations, schemas, artifact plumbing, and a small
  MEDS-batch adapter layer. Copied *verbatim* and re-rendered by `copier update`. **No models.**
  It also carries two pieces of template-managed *tooling* that are deliberately not part of the command
  contract: `testing/` (the conformance harness) and `meds_dev.py` (`meds-model-add-to-meds-dev`, which
  installs `model.yaml` into a MEDS-DEV checkout). Neither may become a sixth `CommandName`.
- `src/<model_slug>/` — the **user-owned surface**: `model.py` (a stub), `predict.py` (a stub, for the
  `zero_shot_direct` and `packaged` DAGs), `commands.py`, `configs/`. All of those plus
  `configs/model/**`, `configs/paths/**` and `configs/profile/**` are protected from `copier update`
  (`_skip_if_exists`); `__main__.py` and the other config groups are not.

### The template ships no model

`model.py` is a **stub**: it declares the hooks the chosen DAG calls and raises `NotImplementedError` from
each, marked with a class attribute `is_stub = True`. This is deliberate and load-bearing — do not "helpfully"
fill it in.

The history is worth knowing, because the pull is to re-add a model each time something needs one:

1. `meds_model_base/profiles/` held four complete `LightningModule`s that `model.py` subclassed in three
   lines. That inverted the ownership split — the file users are told to edit was a stub while their real
   model lived where `copier update` overwrites it.
2. Those moved into a fully-rendered `model.py`. Better, but the template was still choosing architectures.
3. Now the architecture is gone entirely. `profile` selects a **DAG shape**, nothing more.

What stayed in the contract is only what must know the MEDS batch format or the command contract:
`CodeEmbedder`, `padding_mask`, `masked_mean`, `BaseLightningModule` (whose `infer_step` / `inference_kind`
the `infer` command reads), and the inference-artifact plumbing a probe joins against.

**The cost, which is real:** nothing proves a chain trains and predicts end to end. `tests/` here is
structural, and the rendered conformance tests skip on a stub. `meds_model_base.testing.skip_if_stub`
is what gates them; a repo is green but honest, and `-rs` in CI shows the skips. Closing this gap needs a
reference implementation living *outside* the payload (an `examples/` tree that a test renders a profile
and drops in). That is the agreed follow-up — not a reason to put a model back in `template/`.

### The command contract

`template/src/meds_model_base/commands/base.py` defines `CommandName` (five commands) and the
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
- `resolve_workspace` recovers `input_data_dir` from the source model's manifest when `predict` is not
  given one — which is what makes MEDS-DEV's rolling `model_initialization_dir` usable without
  re-tensorizing.
- `ProbePredictCommand` recovers which embeddings to score from the probe's own manifest.

That pattern is why `predict` needs no workspace or embeddings argument: adding one would invite exactly
the mismatch the manifest exists to prevent.

**There is no `preprocess_task` command and no task artifact.** Tasks arrive as `external_labels_dir` — a
MEDS labels directory, which is what `meds-dev-task` produces and all MEDS-DEV ever passes a model. Each
command that needs one materializes the split layout meds-torch-data wants into its own work directory.
The template never parses ACES: resolving a task *definition* needs dataset-specific predicates it does
not have, and a zero-shot model that needs to know *what* it is predicting must arrange that itself.

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

### Profiles are DAGs

There is one profile per chain in `design-interface.md`: `supervised`, `finetune`, `probe`,
`zero_shot_direct`, `zero_shot_materialized`, `packaged`, plus `custom`. Adding a chain to the spec means
adding a profile, and `tests/test_render.py::test_every_command_class_is_reachable` fails if a command
class exists that no profile registers — which is how `MaterializedPredictCommand` was caught sitting dead.

`copier.yml`'s `implements_*` booleans now gate only the `custom` profile. Nine files branch on `profile`:
`commands.py.jinja`, `model.py.jinja`, the conditional `predict.py` filename, `configs/model/default.yaml`,
`configs/model/probe.yaml`, `configs/supervised_train.yaml`, `configs/profile/default.yaml`,
`model.yaml.jinja`, and `README.md.jinja`.

`commands.py.jinja` builds an `entries` list in Jinja and derives its import block from it, so a profile
can never import a class it does not register (an F401 in the generated repo).

`_templates_suffix: .jinja` means **only** `.jinja` files are rendered; everything else under `template/`
is copied byte-for-byte (path segments like `{{ model_slug }}` are still substituted). A file whose
rendered *name* is empty is skipped — that is how `predict.py` is made conditional.

### Tests

`tests/` in **this** repo is structural and never touches a model. Beyond rendering each DAG and
byte-compiling it, it asserts four properties that each caught a real bug:

- `test_model_is_a_stub` — the payload ships no implementation (guards against regression 1–3 above).
- `test_declared_chain_matches_registry` — `configs/profile/default.yaml`'s `chain` equals `commands.py`.
- `test_every_consumed_artifact_is_produced` — required sources have producers, and `infer` output is
  consumed. Parsed with `ast`, not regex: `supported_sources` is declared in some classes and inherited in
  others, and a regex loose enough to span a class body matches the *next* class's declaration.
- `test_rendered_repo_passes_ruff` — this repo excludes `template/` from linting, so without it the
  vendored contract is never linted here at all.

`tests/` in a **generated** repo is the conformance suite for the user's model: CLI/workspace/arbitration
tests that run immediately, plus `test_smoke_pipeline` and `test_property` (designed signal + negative
control) that `skip_if_stub` until `model.py` is implemented. Both drive the chain through
`conftest.run_chain`, which reads `COMMANDS` and each class's `supported_sources` rather than hardcoding a
chain — so a DAG whose wiring and implementation disagree fails instead of silently testing something else.

## Known gaps

- **No end-to-end MEDS-DEV run.** `model.yaml.jinja` is written against MEDS-DEV's actual placeholder
  contract (checked against its source and two real models), but has never been executed by MEDS-DEV.
  `test_meds_dev_helper_writes_where_the_loader_looks` narrows this a little by replaying MEDS-DEV's own
  discovery glob (`rglob("*/model.yaml")` keyed on the parent directory) against what the helper writes,
  but it stubs the checkout — MEDS-DEV itself is never imported.
- **Nothing trains.** A generated repo ships no model, so no chain has ever run training or prediction —
  only preprocessing, dispatch and config composition. Closing that needs a reference implementation
  outside the payload.
- **Zero-shot task resolution is unspecified by design.** MEDS-DEV passes no task definition and has no
  task-name placeholder, so a zero-shot model must obtain it some other way. The template deliberately
  takes no position; revisit once the supervised chains are validated.
