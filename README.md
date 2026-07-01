# MEDS Model Template

A [Copier](https://copier.readthedocs.io) template for building **standards-conformant
[MEDS](https://github.com/Medical-Event-Data-Standard/meds) models** that expose a *mandated* five-step
CLI and are directly contributable to
[MEDS-DEV](https://github.com/Medical-Event-Data-Standard/MEDS-DEV).

Every model generated from this template — whether a simple supervised classifier, a zero-shot
autoregressive generator (à la [MEDS-EIC-AR](https://github.com/mmcdermott/MEDS_EIC_AR)), a query-based
pretrained model (à la [EveryQuery](https://github.com/payalchandak/EveryQuery)), or a MOTOR-style
time-to-event foundation model — has the **same usage pattern**:

```bash
pip install .                       # or: uv sync
meds-model preprocess               input_dir=$MEDS_ROOT output_dir=data
meds-model unsupervised_train       datamodule.config.tensorized_cohort_dir=data output_dir=runs/pretrain
meds-model supervised_train         ... labels_dir=$LABELS output_dir=runs/model
meds-model task_agnostic_inference  ... index_df=$INDEX  output_dir=runs/embeddings
meds-model prediction               ... task=$ACES_YAML  output_dir=runs/preds   # -> predictions.parquet
```

A given model implements a *subset* of these steps; `meds-model steps` prints which.

## The five steps

| Step | What it does | Key input | Output |
|---|---|---|---|
| `preprocess` | raw MEDS → model-ready tensors | `$MEDS_ROOT` | tensorized cohort (default: [meds-torch-data](https://github.com/mmcdermott/meds-torch-data)) |
| `unsupervised_train` | self-supervised pretraining | preprocessed dir | checkpoint dir |
| `supervised_train` | supervised (fine-)tuning | preprocessed dir + labels | checkpoint dir |
| `task_agnostic_inference` | inference at given timepoints | **index df** `(subject_id, prediction_time)` | open output (embeddings / scores) |
| `prediction` | task-specific scored predictions | **[ACES](https://github.com/justin13601/ACES) task YAML** | `predictions.parquet` (meds-evaluation schema) |

`task_agnostic_inference` takes only *where* to predict; `prediction` additionally takes *what* to
predict (an ACES task definition), runs ACES in-process, and emits a
[meds-evaluation](https://github.com/kamilest/meds-evaluation)-conformant `predictions.parquet`.

## Quick start

Generate a new model with Copier (recommended):

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

Pick a **profile** at generation time (or `custom` to choose each step):

| Profile | Steps | Analogue |
|---|---|---|
| `supervised_basic` | `{preprocess, supervised_train, prediction}` | a classic supervised classifier |
| `zero_shot_ar` | `{preprocess, unsupervised_train, task_agnostic_inference, prediction}` | MEDS-EIC-AR |
| `every_query` | `{preprocess, unsupervised_train, prediction}` | EveryQuery |
| `motor_finetune` | `{preprocess, unsupervised_train, supervised_train, prediction}` | MOTOR |

## What you get

A generated repo contains:

- **`src/meds_model_base/`** — the *vendored, template-managed* contract: step ABCs, the `meds-model`
  dispatcher, default step implementations (MEDS-transforms + meds-torch-data + Lightning + ACES +
  meds-evaluation), schema validators, and a reusable pytest harness. `copier update` re-renders this to
  pull in contract improvements.
- **`src/<your_model>/`** — the *user-owned* surface: `model.py` (your `LightningModule`), `steps.py`
  (which steps you implement), and `configs/model/` (your Hydra overrides). Protected from
  `copier update`.
- **Hydra configs**, **CI**, **pre-commit**, a **`model.yaml`/`requirements.txt`** for MEDS-DEV, and a
  three-tier **test suite**: CLI smoke tests, an end-to-end pipeline smoke test on
  [meds-testing-helpers](https://github.com/mmcdermott/meds_testing_helpers) data, and a
  **model-specific synthetic-data property test**.

## Configuration

Extra arguments are supplied through Hydra. Every step is a Hydra application reading a packaged
`configs/` tree, so anything is overridable on the command line
(`meds-model supervised_train trainer.max_epochs=50 model.hidden_size=256`) or via config files.

## Updating a generated repo

```bash
cd my-model
copier update            # 3-way merge: pulls new template into src/meds_model_base/, keeps your model.py
```

See [`docs/DESIGN.md`](docs/DESIGN.md) for the full design rationale and the verified MEDS-ecosystem API
surface this template is built on.

## License

MIT
