# MEDS Model Template — Design

This document is the authoritative design for `MEDS_model_template`: a
[Copier](https://copier.readthedocs.io) template that generates **standards-conformant MEDS model
repositories** exposing a *mandated* five-step CLI, so that any model built from it has the same usage
pattern and is directly contributable to
[MEDS-DEV](https://github.com/Medical-Event-Data-Standard/MEDS-DEV).

It is grounded in a deep read of the reference model
[`MEDS_EIC_AR`](https://github.com/mmcdermott/MEDS_EIC_AR) and the current (verified) APIs of the MEDS
ecosystem.

---

## 0. Locked decisions

| Decision | Choice | Rationale |
|---|---|---|
| Scaffolding tool | **Copier** (`>=9`) | `copier update` propagates template changes into already-generated repos; typed questions; conditional file inclusion. |
| Artifact shape | **Pure Copier template with a vendored base package** | Each generated repo contains a *template-managed* `src/meds_model_base/` (the contract) **and** a *user-owned* `src/<model_slug>/` (the model). No separate PyPI library to publish; contract evolves via `copier update`. |
| CLI namespace | `meds-model` (command), `meds_model_base` (base package) | Neutral; avoids collision with MEDS-DEV's own `meds-dev-model`. |
| v1 scope | **All four profiles**, built natively on one modern stack | supervised-basic, zero-shot-AR, EveryQuery, MOTOR-style TTE fine-tune. |
| Stack | Hydra + PyTorch Lightning + `meds-torch-data` + MEDS-transforms + ACES + meds-evaluation | Modernizes EIC-AR: one dispatched entrypoint, real `supervised_train` + real `prediction`. |

### Why "vendored base" (and what that means)

Everything contract-critical (step ABCs, the dispatcher, default step implementations, schema
validators, the test harness, base Hydra configs) lives in `src/meds_model_base/`. In a generated repo
this package is **template-managed**: `copier update` re-renders it, so a contract change reaches a
downstream repo with one `copier update`. The *swappable surface* — `src/<model_slug>/model.py`, the
`configs/model/` group, `steps.py` — is **user-owned** and protected from `copier update` via
`_skip_if_exists`. MEDS-DEV installs each model into an isolated `uv` venv, so vendoring `meds_model_base`
never collides across models.

---

## 1. The mandated CLI contract

One dispatched entrypoint; each step is optional per model.

```
meds-model <step> [hydra.overrides ...]
meds-model steps          # prints the steps THIS model implements (from the STEPS registry)
```

`<step> ∈ {preprocess, unsupervised_train, supervised_train, task_agnostic_inference, prediction}`

| Step | Purpose | Input | Output |
|---|---|---|---|
| **a. preprocess** | Raw MEDS → model-ready tensors | `$MEDS_ROOT` (`meds.DataSchema` shards + `metadata/`) | model-ready dir (default: MTD tensorized cohort) |
| **b. unsupervised_train** | Self-supervised pretraining | preprocessed dir; splits via `SubjectSplitSchema` | `model_initialization_dir/{best_model.ckpt, config.yaml, ...}` |
| **c. supervised_train** | Supervised (fine-)tuning | preprocessed dir + `labels_dir` (`meds.LabelSchema`) + optional pretrained dir | fine-tuned `model_initialization_dir` |
| **d. task_agnostic_inference** | Inference at arbitrary timepoints | preprocessed dir + model dir + **index df** `(subject_id, prediction_time)` | `TaskAgnosticOutputSchema` parquet (open; embeddings/scores) |
| **e. prediction** | Task-specific scored predictions | preprocessed dir + model dir + **ACES task YAML** (or pre-extracted labels) | `meds_evaluation.PredictionSchema` → `predictions.parquet` |

**Steps (d) vs (e).** Both consume patient timepoints. `(d)` takes only an *index dataframe*
(`subject_id, prediction_time`) and emits an open, un-scored output (e.g. embeddings or zero-shot
scores). `(e)` additionally takes an **ACES configuration** which is run in-process to derive the index
*and* the ground-truth `boolean_value`; its output is a scored, meds-evaluation-conformant
`predictions.parquet`.

### 1.1 Interface (`meds_model_base.steps.base`)

```python
class StepName(StrEnum):
    preprocess = "preprocess"
    unsupervised_train = "unsupervised_train"
    supervised_train = "supervised_train"
    task_agnostic_inference = "task_agnostic_inference"
    prediction = "prediction"

class MEDSModelStep(ABC):
    name: ClassVar[StepName]
    config_name: ClassVar[str]          # Hydra root config, e.g. "_prediction"
    @abstractmethod
    def run(self, cfg: DictConfig) -> Path: ...   # execute; return primary output dir
```

Per-step ABCs pin the IO contract and expose the override hook(s) the default implementations fill
(`build_module`, `infer`, `predict`, `resolve`). A model declares which steps it implements via a
`STEPS: dict[StepName, type[MEDSModelStep]]` registry in `src/<model_slug>/steps.py`. The dispatcher
consults it; a non-implemented step exits cleanly. Copier's `profile` question presets the
`implements_*` booleans that gate which step modules/configs/registry-entries/`model.yaml` cells render.

### 1.2 Schemas (single source of truth)

`meds_model_base.schemas` re-exports canonical classes; it never hardcodes column names.

```python
from meds import DataSchema, LabelSchema, SubjectSplitSchema, CodeMetadataSchema
from meds_evaluation.schema import PredictionSchema     # subclass of LabelSchema
# IndexSchema             = LabelSchema restricted to (subject_id, prediction_time)   [validate-only]
# TaskAgnosticOutputSchema = flexible_schema OPEN class over those two keys + model columns
```

Key facts baked in (verified against installed versions):
- `LabelSchema` is **CLOSED** (`allow_extra_columns=False`); value columns are `Optional` but **non-nullable if present**. `DataSchema` / `CodeMetadataSchema` are OPEN.
- `PredictionSchema` adds `predicted_boolean_value: bool?` and `predicted_boolean_probability: float32?`; at least one must be present & non-all-null. Validate/coerce with `PredictionSchema.align(table)`.
- `prediction_time` is an **inclusive as-of (`<=`) cutoff**, not necessarily an observed event time.
- Hydra **enum overrides are UPPERCASE** (e.g. `datamodule.config.seq_sampling_strategy=TO_END`).

---

## 2. Default step implementations (all overridable)

| Step | Default (libs / functions) | Override |
|---|---|---|
| preprocess | `DefaultPreprocessStep`: run packaged MEDS-transforms pipeline (`pipelines/preprocess.yaml`) via `MEDS_transforms.runner`, then MTD tensorization via `MTD_preprocess MEDS_dataset_dir=… output_dir=… do_reshard=…`. Enforces the two invariants `validate()` misses (one-file-per-subject; complete `codes.parquet`). | point `preprocess.pipeline` at another YAML; add `@Stage.register` stages via entry points; replace the step class |
| unsupervised_train | `DefaultUnsupervisedTrainStep`: `instantiate(cfg.datamodule)` (MTD `Datamodule`, no `task_labels_dir`) → `build_module()` → `instantiate(cfg.trainer)` → `trainer.fit`; write `best_model.ckpt` + `resolved_config.yaml`; `do_overwrite`/`do_resume`. | swap `configs/model/`; override `build_module()`; wrap external trainer |
| supervised_train | `DefaultSupervisedTrainStep`: datamodule with `task_labels_dir` (⇒ `seq_sampling_strategy=TO_END`); optional pretrained-encoder load; Lightning fit with a classification head (BCE on `MEDSTorchBatch.boolean_value`). | `freeze_encoder` flag; override `build_module()`/head |
| task_agnostic_inference | `DefaultTaskAgnosticInferenceStep`: datamodule uses the **index df** as `task_labels_dir` (`boolean_value=None`), `SequentialSampler`; `trainer.predict` streamed through a DDP-safe writer; emits `TaskAgnosticOutputSchema`. | override `infer(batch)` (embeddings vs zero-shot scores) |
| prediction | `SupervisedPredictionStep` (sigmoid→prob, threshold→bool, join `boolean_value`) and `ZeroShotPredictionStep` (generate/query at the ACES times, abstract `resolve(...)→probs`). Labels via `aces_labels.extract()` *or* a `labels_dir`. Output validated by `PredictionSchema.align()`. | subclass, implement `predict()`/`resolve()`; `run()` (validate + write) is fixed |

### Profiles → wiring

- **supervised-basic `{a,c,e}`** — encoder + classification head; `DefaultSupervisedTrainStep` + `SupervisedPredictionStep`.
- **zero-shot-AR `{a,b,d,e}`** — autoregressive "everything-is-code" GPT; unsupervised next-token pretraining; `task_agnostic_inference` = generate; `ZeroShotPredictionStep` resolves ACES windows over generated futures.
- **EveryQuery `{a,b,e}`** — encoder pretrained with a query (does code X occur within Δ?) objective; `ZeroShotPredictionStep` queries the task's `(code, window)`.
- **MOTOR-style `{a,b,c,e}`** — time-to-event pretraining (piecewise-exponential hazard head) then supervised fine-tune of a task head; `SupervisedPredictionStep`. Native reimplementation on the modern stack (FEMR interop is an optional documented escape hatch behind a `[femr]` extra).

---

## 3. Hydra config layout

Base defaults ship in `meds_model_base/configs/` (trainer, callbacks, logger, optimizer, lr_scheduler,
datamodule, paths, hydra, extras). Generated `src/<model_slug>/configs/` carries only **roots** (one per
implemented step) + the **`model/`** override group. A `SearchPathPlugin` registered by
`meds_model_base` makes `pkg://meds_model_base/configs` resolvable, so roots inherit base defaults.

```yaml
# src/<model_slug>/configs/_supervised_train.yaml     # @package _global_
defaults:
  - _self_
  - datamodule: supervised
  - model: default          # ← user-owned surface, lives in THIS package
  - trainer: default
  - callbacks: default
  - logger: csv
  - optimizer: adamw
  - lr_scheduler: cosine
  - paths: default
  - hydra: default
output_dir: ???
labels_dir: ???
model_initialization_dir: null
do_overwrite: false
do_resume: true
```

`MEDSTorchDataConfig.add_to_config_store("datamodule/config")` is called at import so the datamodule is
a type-checked structured config.

---

## 4. Testing strategy (three tiers)

Shipped by `meds_model_base.testing` (a `pytest11` plugin) and rendered into `tests/`.

1. **CLI smoke** — for each implemented step, `meds-model <step> --help` exits 0 (catches
   resolver-registration / config-path regressions cheaply).
2. **End-to-end pipeline smoke** — a session-scoped fixture chain over `meds_testing_helpers`
   (`simple_static_MEDS`, `simple_static_MEDS_dataset_with_task`) driving the *actual* CLI steps; the
   load-bearing assertion is `PredictionSchema.align(predictions.to_arrow())`.
3. **Model-specific synthetic-data property test** (`@pytest.mark.slow`) — the flagship. A generalization
   of EIC-AR's grammar-FSM test: the contributor supplies `build_signal_dataset()` (synthetic MEDS with
   a *designed* signal deterministically predictive of the label) and `assert_learns(model)` (the
   learnability contract for the model class), always paired with a **negative control** so the test
   can't be vacuously satisfied. A companion Hypothesis test fuzzes cohort shape and asserts every
   `prediction` output round-trips `PredictionSchema.align()`.

---

## 5. MEDS-DEV integration

A generated repo ships `model.yaml` + `requirements.txt` for MEDS-DEV's 2×2 command grid
(`{unsupervised, supervised} × {train, predict}`; only `supervised.predict` is mandatory). MEDS-DEV runs
each command in an isolated `uv` venv, so the only cross-boundary contract is the on-disk parquet
columns: it passes `labels_dir` (`subject_id, prediction_time, boolean_value`) and reads
`predictions.parquet` (`predicted_boolean_probability` / `predicted_boolean_value`). There is no
first-class preprocess/`task_agnostic_inference` slot, so preprocess is folded into the train command
(writing to `{output_dir}/data`). A `meds-model export-meds-dev <checkout>` helper drops the model dir
into a MEDS-DEV checkout.

**Version-skew note.** MEDS-DEV's own env currently pins `meds==0.3.3` / `es-aces==0.6.1` /
`meds-evaluation==0.0.3`, older than the model's own venv (meds 0.4.x). The parquet column names are
stable across that skew, so it is tolerable — but the template ships a CI job asserting the on-disk
contract, and this should be confirmed against a live MEDS-DEV run.

---

## 6. Repository layout

```
MEDS_model_template/                     # the template repo (this repo)
  copier.yml                             # questions: model_name, model_slug, author, python, profile, implements_*
  pyproject.toml                         # bootstrap pkg `meds-model-template` (`meds-model new` shells to copier)
  docs/DESIGN.md
  tests/                                 # template's own tests: render each profile, run its suite
  template/                              # copier _subdirectory
    {{ _copier_conf.answers_file }}.jinja
    pyproject.toml.jinja  README.md.jinja  .gitignore.jinja
    .pre-commit-config.yaml  .github/workflows/ci.yml
    env.sh.jinja  model.yaml.jinja  requirements.txt.jinja
    src/
      meds_model_base/                   # VENDORED base — copied verbatim (template-managed)
        dispatch.py schemas.py aces_labels.py
        steps/{base,preprocess,train,inference,predict}.py
        lightning/{datamodule,writer,modules}.py
        configs/…  pipelines/preprocess.yaml
        testing/{__init__,fixtures,synthetic,property}.py
        profiles/                        # the four reference LightningModules + registries
      {{ model_slug }}/                  # USER-OWNED surface (rendered from the chosen profile)
        __init__.py  __main__.py.jinja  steps.py.jinja  model.py.jinja
        configs/…                        # roots per implemented step + model/ group
    tests/{conftest,test_cli_smoke,test_smoke_pipeline,test_property}.py.jinja
```

---

## 7. Milestones

M0 skeleton · M1 contracts · M2 dispatch · M3 preprocess · M4 training · M5 infer+predict ·
M6 test harness · M7 four profiles · M8 copier authoring + MEDS-DEV export · M9 render+run verification.
