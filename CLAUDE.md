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

**[`docs/design-interface.md`](docs/design-interface.md) is the authoritative spec** for the command graph, artifact
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

Expect ~37 passed and 2 skipped, and both skips are the correct result — everything that does not need a
model runs, including real MTD tensorization and task materialization. One skip is `skip_if_stub`; the
other is `test_unsupported_command_fails_clearly`, whose parameter set is empty for `probe` because that
DAG registers all five commands.

Add `'use_wandb': True, 'use_mlflow': True` to that `data` dict to render the optional logger configs —
otherwise `logger=csv` is the only thing the generated suite ever composes.

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

### The SLURM helpers derive the chain, they do not restate it

`template/slurm/` renders into every generated repo. `submit.sh` gets its *stage list* from
`meds-model commands` at submit time rather than from a rendered copy, which is what keeps it out of the
set of places the DAG is written down (`commands.py`, `configs/profile/default.yaml`, `CLAUDE.md` — three
already, guarded by `test_claude_md_states_the_same_chain`).

What it does duplicate is the *argument wiring* per command, which is the same knowledge as
`meds_model_base.testing.harness.run_chain`. That is guarded instead:
`test_slurm_submitter_covers_every_command` asserts `stage_args()`'s case arms equal `ALL_COMMANDS` and
that every source parameter in `SOURCE_PRODUCER` appears. Adding or renaming a command therefore also
touches `template/slurm/submit.sh.jinja`.

Shell has no equivalent of the byte-compile the Python payload gets, so
`test_slurm_scripts_render_as_runnable_bash` runs `bash -n` over the rendered scripts for every profile and
checks the executable bit survived rendering (Copier preserves file mode; the payload files are `chmod +x`).

One Jinja trap, already paid for once: `${` immediately followed by `#` — as in `${#array[@]}` — opens a
Jinja comment tag and breaks the render of *every* profile. Use `${array[0]:-}` or an explicit `{{ '{#' }}`.

### Agent-facing docs are part of the payload

`template/CLAUDE.md.jinja` and `template/docs/PORTING-A-MODEL.md.jinja` render into every generated repo.
That is the point of them: the rules an agent breaks first (the stub is deliberate, `src/meds_model_base/`
is overwritten by `copier update`, the dispatcher calls `__call__` not `run`) and the porting procedure
were only ever written *here*, where a port never looks. `meds-eic-ar-ft` was done with neither in reach.

They divide by lifetime, and the split is worth preserving. The rendered `CLAUDE.md` is loaded into every
session in that repo forever, so it holds only what stays true — ownership, the contract's load-bearing
rules, how to read the skips. `docs/PORTING-A-MODEL.md` is a one-time procedure with a deliverable, so it
is a linked document with a hard trigger sentence in `CLAUDE.md` rather than inline text every later
session pays for.

Neither is in `_skip_if_exists`: contract changes have to reach existing repos, and `copier update`
3-way merges them. Note also that the rendered `CLAUDE.md` only *asks* for the implementation report — a
document cannot enforce it. The check that would (a report linter gating CI on `is_stub` being gone) does
not exist yet.

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
`entries` table in `commands.py.jinja`, the chain block in `CLAUDE.md.jinja`, and `PROFILE_COMMANDS` +
`ALL_COMMANDS` in `tests/test_render.py`.

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

There is one profile per chain in `docs/design-interface.md`: `supervised`, `finetune`, `probe`,
`zero_shot_direct`, `zero_shot_materialized`, `packaged`, plus `custom`. Adding a chain to the spec means
adding a profile, and `tests/test_render.py::test_every_command_class_is_reachable` fails if a command
class exists that no profile registers — which is how `MaterializedPredictCommand` was caught sitting dead.

`copier.yml`'s `implements_*` booleans now gate only the `custom` profile. Ten files branch on `profile`:
`commands.py.jinja`, `model.py.jinja`, the conditional `predict.py` filename, `configs/model/default.yaml`,
`configs/model/probe.yaml`, `configs/supervised_train.yaml`, `configs/profile/default.yaml`,
`model.yaml.jinja`, `README.md.jinja`, and `CLAUDE.md.jinja`.

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
- `test_rendered_repo_is_clean_as_written` — this repo excludes `template/` from linting, so without it
  the vendored contract is never linted here at all. It renders with `skip_tasks=True`: `copier.yml`'s
  post-copy `uvx ruff check --fix` would otherwise repair the payload a second before the test measured
  it, which is exactly how four defects survived (unsorted imports in `train.py`, `test_cli_smoke.py` and
  `test_smoke_pipeline.py`; four unused imports in the `packaged` stub; and, once `ruff format --check`
  was asserted alongside `check`, 12 more files). The trailing-newline test skips the tasks for the same
  reason. Formatting is worth asserting rather than leaving to the post-copy task because Jinja produces
  artefacts nothing else catches: a conditional paragraph inside a docstring leaves `"""Summary.\n """`,
  which ruff collapses to one line in exactly the profiles where the branch is off, so `model.py` was
  malformed for four of the seven profiles and correct for the other three.

A fifth, `test_claude_md_states_the_same_chain`, guards drift rather than a bug already made: the DAG is
now written down three times (`commands.py`, `configs/profile/default.yaml`, `CLAUDE.md`), and the third
is the one an agent reads instead of the code.

`tests/` in a **generated** repo is the conformance suite for the user's model: CLI/workspace/arbitration
tests that run immediately, plus `test_smoke_pipeline` and `test_property` (designed signal + negative
control) that `skip_if_stub` until `model.py` is implemented. Both drive the chain through
`conftest.run_chain`, which reads `COMMANDS` and each class's `supported_sources` rather than hardcoding a
chain — so a DAG whose wiring and implementation disagree fails instead of silently testing something else.

`test_meds_dev_e2e` is a third `skip_if_stub` tier, and the only one that leaves this repository's
environment: it clones MEDS-DEV, installs it editable into its own venv, registers the model with
`meds-model-add-to-meds-dev`, and runs `meds-dev-model mode=full dataset_type=full` — one invocation that
covers every profile, because MEDS-DEV itself walks whatever stages the model declares and rolls
`model_initialization_dir` forward. Naming the marker is the whole opt-in — it clones MEDS-DEV itself, and
a bare `pytest` skips it rather than cloning unannounced. `MEDS_DEV_DIR` is an optional override that
reuses a local checkout (cloned to a temp dir, never mutated) for offline runs:

```bash
UV_TORCH_BACKEND=cpu uv run pytest -m meds_dev -rs                        # clones MEDS-DEV
MEDS_DEV_DIR=~/Git/MEDS-DEV UV_TORCH_BACKEND=cpu uv run pytest -m meds_dev -rs
```

Two things it must keep doing, both learned by running it: install MEDS-DEV from a **git clone** (it
versions itself with `setuptools-scm`, so a `.git`-less copy cannot be built at all), and let MEDS-DEV
build its own isolated venv rather than sharing this one — MEDS-DEV pins `meds==0.3.3` against the
template's `meds~=0.4`.

## Known gaps

- **No *complete* MEDS-DEV run**, for the same reason nothing trains: the chain reaches the model and stops.
  `tests/test_meds_dev_e2e.py` in a generated repo is the real thing (clone MEDS-DEV → install editable →
  register via the helper → `meds-dev-model mode=full` → assert `predictions.parquet` against
  `PredictionSchema`), and it `skip_if_stub`s like the rest of the conformance suite. Everything in it
  *up to the model* has been executed for real: MEDS-DEV built its isolated venv from the generated
  `requirements.txt` with a working `meds-model` entry point, filled its placeholders, and ran
  `preprocess_data` through to a published workspace. The final assertions have never run.
- **Nothing trains.** A generated repo ships no model, so no chain has ever run training or prediction —
  only preprocessing, dispatch and config composition. Closing that needs a reference implementation
  outside the payload.
- **Zero-shot task resolution is unspecified by design.** MEDS-DEV passes no task definition and has no
  task-name placeholder, so a zero-shot model must obtain it some other way. The template deliberately
  takes no position; revisit once the supervised chains are validated.
