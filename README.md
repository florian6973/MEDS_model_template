# MEDS Model Template

> [!WARNING]
> **This repository is entirely AI-generated (as of this writing) and has not yet been human-reviewed.**
> It was scaffolded end-to-end by an AI agent: the design, the vendored contract, every model profile,
> the tests, and this README. It renders and passes its own smoke/property tests locally, but treat every
> line as unreviewed — audit before relying on it, and expect breaking changes. Not yet validated against a
> live MEDS-DEV run.

A [Copier](https://copier.readthedocs.io) template for building **standards-conformant
[MEDS](https://github.com/Medical-Event-Data-Standard/meds) models** that expose a shared six-command
interface and are directly contributable to
[MEDS-DEV](https://github.com/Medical-Event-Data-Standard/MEDS-DEV).

The template generates the **command DAG, not the model**. `model.py` is a stub whose hooks raise until
you write them; what you get for free is the interface, the artifact discipline, and a test suite that is
the specification your model has to satisfy.

Every model generated from this template has the **same usage pattern**:

```bash
pip install .                    # or: uv sync
meds-model preprocess_data   external_meds_dir=$MEDS_ROOT output_data_dir=runs/data
meds-model preprocess_task   input_data_dir=runs/data external_task_file=$TASK
meds-model pretrain          input_data_dir=runs/data output_pretrained_model_dir=runs/models/pretrained
meds-model infer             input_data_dir=runs/data input_pretrained_model_dir=runs/models/pretrained
meds-model supervised_train  input_data_dir=runs/data output_supervised_model_dir=runs/models/supervised
meds-model predict           input_data_dir=runs/data input_supervised_model_dir=runs/models/supervised \
                             output_predictions_dir=runs/predictions
```

A given model supports a *subset* of these; `meds-model commands` prints which. The commands form a **DAG,
not a fixed pipeline** — see [`design-interface.md`](design-interface.md) for the full specification.

## The six commands

| Command | What it does | Key input | Output |
|---|---|---|---|
| `preprocess_data` | external MEDS → model-ready tensors | `external_meds_dir` | `<data_dir>/patients/` |
| `preprocess_task` | materialize or validate a task | ACES YAML **or** labels | `<data_dir>/tasks/<name>/` |
| `pretrain` | foundation-model pretraining | patient data | a pretrained model dir |
| `infer` | materialize reusable outputs | patient data + pretrained model | `<data_dir>/inference/<name>/` |
| `supervised_train` | task training, optionally on a prior artifact | patient data + task | a supervised model dir |
| `predict` | standardized predictions | task + exactly one source | `predictions.parquet` |

`data_dir` is a shared workspace: `patients/` is written once and never modified; `preprocess_task` and
`infer` append sibling subdirectories. Every artifact carries its own `manifest.yaml` and is published by
atomic rename, so a directory that exists is a directory that finished. There is no aggregate manifest to
keep in sync, which is what lets independent jobs materialize different tasks concurrently.

## Three invariants the contract enforces

These exist because each one otherwise produces output that looks correct:

1. **`predict` never reads ground truth.** The task's `boolean_value` is dropped on load; the repository
   ends at predicted probabilities and scoring is a separate, shared tool.
2. **Predictions cover the whole index.** A model that can only score part of a task fails rather than
   writing a short file, and `n_expected` / `n_written` are recorded per split in the manifest.
3. **A warm start that matches no parameters is an error.** Non-strict checkpoint loading is what makes
   fine-tuning a fresh head possible; it also silently yields a randomly initialized model when an encoder
   is renamed. The matched-parameter count is checked and recorded.

Likewise, where a command accepts several alternative sources, supplying two is an error rather than a
precedence decision, and supplying one the implementation does not handle is an error rather than a
silently ignored argument.

## Quick start

```bash
uv tool install copier
copier copy gh:mmcdermott/MEDS_model_template ./my-model
cd my-model && uv sync --all-extras
uv run pytest -m "not slow"
```

Or via the bootstrap package:

```bash
uv tool install meds-model-template
meds-model-new ./my-model
```

Pick a **DAG** at generation time — one profile per chain in
[`design-interface.md`](design-interface.md):

| Profile | Chain | `predict` consumes | Like |
|---|---|---|---|
| `supervised` | preprocess ×2 → `supervised_train` → `predict` | supervised model | a classic task classifier |
| `finetune` | + `pretrain`, then fine-tune | supervised model | MOTOR |
| `probe` | + `pretrain`, `infer` embeddings, head on frozen features | supervised model (the head) | linear probing |
| `zero_shot_direct` | `pretrain` → `predict` straight from the model | pretrained model | EveryQuery |
| `zero_shot_materialized` | `pretrain` → `infer` scores → `predict` from them | inference artifacts | MEDS-EIC-AR |
| `packaged` | preprocess ×2 → `predict` | nothing (weights ship with the repo) | PFN-style |

A profile is a **shape, not a model**. Every generated repository ships all six commands; the profile
decides which are registered and how their artifacts connect. The render tests assert each one is a
*runnable* DAG: everything a command requires is produced by another command in the chain, and nothing
produced is left unconsumed.

## What you get

- **`src/<your_model>/model.py`** — **a stub.** The hooks your DAG calls, documented, each raising
  `NotImplementedError`. This is the part the template deliberately does not write for you.
- **`src/meds_model_base/`** — the *vendored, template-managed* contract: command ABCs, source
  arbitration, the `meds-model` dispatcher, the manifest layer, default command implementations
  (MEDS-transforms + meds-torch-data + Lightning + ACES + meds-evaluation), schema validators, and a small
  MEDS-batch adapter layer. `copier update` re-renders it. **It contains no models.**
- **Hydra configs** (one root per command, plus a shared `paths` group), **CI**, **pre-commit**, a
  **`model.yaml`/`requirements.txt`** for MEDS-DEV, and a **conformance test suite**: CLI and workspace
  tests that run immediately, plus end-to-end and designed-signal learnability tests that skip while
  `model.py` is a stub and start running the moment you implement it.

## Configuration

Every command is a Hydra application reading a packaged `configs/` tree, so anything is overridable on the
command line (`meds-model supervised_train trainer.max_epochs=50 model.d_model=256`) or via config files.
Shared locations live in one `paths/default.yaml`; each command has its own root config named after it.

## Updating a generated repo

```bash
cd my-model
copier update            # 3-way merge: pulls the new contract into src/meds_model_base/, keeps your model
```

## Docs

- [`design-interface.md`](design-interface.md) — the authoritative command and artifact specification.
- [`docs/DESIGN.md`](docs/DESIGN.md) — background on the MEDS-ecosystem APIs this is built on. Note that
  its command vocabulary predates `design-interface.md`.

## License

MIT
