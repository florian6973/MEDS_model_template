# MEDS Model Template

> [!WARNING]
> **This repository is entirely AI-generated (as of this writing) and has not yet been human-reviewed.**
> It was scaffolded end-to-end by an AI agent: the design, the vendored contract, all model profiles,
> the tests, and this README. It renders and passes its own smoke/property tests locally, but treat every
> line as unreviewed — audit before relying on it, and expect breaking changes. Not yet validated against a
> live MEDS-DEV run.

A [Copier](https://copier.readthedocs.io) template for building **standards-conformant
[MEDS](https://github.com/Medical-Event-Data-Standard/meds) models** that expose a shared six-command
interface and are directly contributable to
[MEDS-DEV](https://github.com/Medical-Event-Data-Standard/MEDS-DEV).

Every model generated from this template — a supervised classifier, a zero-shot autoregressive generator
(à la [MEDS-EIC-AR](https://github.com/mmcdermott/MEDS_EIC_AR)), a query-based pretrained model (à la
[EveryQuery](https://github.com/payalchandak/EveryQuery)), a MOTOR-style time-to-event model, or a frozen
representation probe — has the **same usage pattern**:

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

Pick a **profile** at generation time (or `custom` to choose each command):

| Profile | Chain | Analogue |
|---|---|---|
| `supervised_basic` | preprocess ×2 → `supervised_train` → `predict` | a classic supervised classifier |
| `zero_shot_ar` | + `pretrain`, `infer` trajectories, zero-shot `predict` | MEDS-EIC-AR |
| `every_query` | + `pretrain`, `predict` by native query | EveryQuery |
| `motor_finetune` | + `pretrain`, then fine-tuned `supervised_train` | MOTOR |
| `probe` | + `pretrain`, `infer` embeddings, head on frozen features | linear probing |

A profile is **configuration, not a code path**: every generated repository ships all six commands. Only
`commands.py` (which the dispatcher reads before Hydra composes) and the model class differ.

## What you get

- **`src/meds_model_base/`** — the *vendored, template-managed* contract: command ABCs, source arbitration,
  the `meds-model` dispatcher, the manifest layer, default implementations (MEDS-transforms +
  meds-torch-data + Lightning + ACES + meds-evaluation), schema validators, and a reusable pytest harness.
  `copier update` re-renders this to pull in contract improvements.
- **`src/<your_model>/`** — the *user-owned* surface: `model.py` (your `LightningModule`), `commands.py`
  (which commands you support), and `configs/` (`model/`, `paths/`, `profile/`). Protected from
  `copier update`.
- **Hydra configs** (one root per command, plus a shared `paths` group), **CI**, **pre-commit**, a
  **`model.yaml`/`requirements.txt`** for MEDS-DEV, and a three-tier **test suite**: CLI smoke tests, an
  end-to-end pipeline smoke test on
  [meds-testing-helpers](https://github.com/mmcdermott/meds_testing_helpers) data, and a
  **model-specific synthetic-data property test** with a negative control.

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
