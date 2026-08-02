# MEDS Model Template: Proposed Command and Artifact Interface

## Goal

Define a small, composable interface that supports:

* supervised models trained from scratch;
* pretrained models with fine-tuning;
* frozen representation probes;
* direct zero-shot prediction;
* prediction from materialized trajectories or scores;
* packaged models such as PFNs.

The interface should distinguish:

1. artifacts supplied externally;
2. artifacts produced by the pipeline;
3. shared paths;
4. command-specific configuration.

Model-specific behavior, including EveryQuery query generation, remains inside the generated model implementation and is not part of the global command contract.

---

## External artifacts

Arguments prefixed with `external_` identify artifacts supplied from outside the model pipeline.

### `external_meds_dir`

Canonical MEDS dataset, **sharded by split**:

```text
data/train/0.parquet
data/tuning/0.parquet
data/held_out/0.parquet
metadata/codes.parquet
metadata/subject_splits.parquet
```

Expected splits:

```text
train
tuning
held_out
```

The shard layout is a precondition, not a preference, and `preprocess_data` refuses input without it.
meds-torch-data recovers split membership from the shard path and never opens `subject_splits.parquet`,
so a dataset sharded another way tensorizes into an artifact with no splits at all.

This is what `meds-dev-dataset` and the standard MEDS ETL produce, so nothing reachable through MEDS-DEV
needs any preparation. Anything else is resharded upstream with a one-stage MEDS-transforms pipeline
(`reshard_to_split`); the error names the pipeline to run. Resharding cannot be done on the caller's
behalf during `preprocess_data`, because `reshard_to_split` reads `metadata/subject_splits.parquet` from
its own input and a MEDS-transforms `pipeline:` does not carry that file through — so the two could never
be combined.

`metadata/subject_splits.parquet` is required of the source dataset (MEDS requires it, and it is what
resharding consumes), but it is *not* copied into `patients/`. The published artifact records split
membership as its shard layout, which is the same place meds-torch-data reads it from.

### `external_labels_dir`

A directory of parquet files in the MEDS label format, containing at least:

```text
subject_id
prediction_time
boolean_value
```

Model-specific task columns are allowed and ignored.

This is deliberately **not** an ACES task YAML. Extracting labels from a task definition is
`meds-dev-task`'s job upstream — it resolves dataset-specific predicates, which a model cannot do — and
MEDS-DEV passes models only the materialized result. Duplicating that here would mean reimplementing
predicate resolution against a spec we do not own.

A consequence worth stating: a model receives *where* to predict, never *what*. Nothing in the labels
identifies the task, and MEDS-DEV has no task-name placeholder. A zero-shot model that needs to know which
task it is answering must obtain that itself — shipped alongside its weights, or via its own argument. The
interface takes no position on how.

---

## Internal artifact types

### Data directory

`data_dir` is a shared workspace containing patient data and derived artifacts.

```text
data_dir/
├── patients/
│   ├── metadata/
│   │   ├── codes.parquet
│   │   └── subject_splits.parquet
│   └── manifest.yaml
└── inference/
    └── <inference_name>/
        ├── artifacts.parquet
        └── manifest.yaml
```

The patient data is created once; `infer` adds subdirectories without copying it.

`subject_splits.parquet` is copied in by `preprocess_data` because meds-torch-data does not carry it
through tensorization, and every command that materializes labels needs it to partition them. Copying a
few kilobytes is what makes `patients/` self-contained — nothing downstream needs the raw dataset again.

There is **no `tasks/` subdirectory**. A task is not a pipeline stage: the commands that need one take
`external_labels_dir` and materialize the split layout meds-torch-data expects into their own work
directory, which is an implementation detail rather than an artifact.

There is deliberately **no `data_dir/manifest.yaml`**. Every artifact directory is self-describing, and the
state of a `data_dir` is derived by scanning `tasks/*/manifest.yaml` and `inference/*/manifest.yaml`. A
root manifest would have to be rewritten every time a task or inference subdirectory is added, which is a
write-write race as soon as two jobs materialize different tasks against the same `data_dir` concurrently
— the normal case on a cluster. Scanning has no such hazard: each subdirectory is written once, atomically,
by exactly one job.

### Pretrained model directory

```text
pretrained_model_dir/
├── checkpoint
├── resolved_config.yaml
└── manifest.yaml
```

### Supervised model directory

```text
supervised_model_dir/
├── checkpoint
├── resolved_config.yaml
└── manifest.yaml
```

### Predictions directory

```text
predictions_dir/
├── predictions.parquet
└── manifest.yaml
```

---

## Mutation rules

`preprocess_data` creates `data_dir`.

Later commands may enrich the same `data_dir` by adding new task or inference subdirectories.

Rules:

* `patients/` is immutable after `preprocess_data`.
* Existing task and inference subdirectories are immutable.
* Commands must fail when the requested output subdirectory already exists, unless overwrite is explicitly enabled.
* Writes should use a temporary directory followed by an atomic rename.
* Every artifact directory records its own provenance in a `manifest.yaml`. Because the directory is renamed into place atomically, a visible artifact always has a complete manifest — a half-written artifact is never observable.

### Write access

| Command | `data_dir` access |
| --- | --- |
| `preprocess_data` | creates it |
| `infer` | appends `inference/<inference_name>/` |
| `pretrain`, `supervised_train`, `predict` | read-only |

The read-only commands write only to their own `output_*_dir` roots. This matters operationally: `pretrain`,
`supervised_train` and `predict` can run against a read-only mount or a bucket with no write policy.

---

## Manifests

A manifest is not documentation. Commands **read the manifests of their inputs and validate them before
doing any work**, so a mismatched artifact fails in a second rather than after an hour on a GPU. That
validation is the reason the format is worth specifying; a manifest nobody reads will rot.

The base package should provide `write_manifest()` (temporary directory + atomic rename) and
`read_manifest(path, require_type=...)`, so input validation is one line at the top of each command and no
model implementation can skip it.

### Common fields

Present in every manifest, whatever the artifact type:

```yaml
manifest_version: 1
artifact:
  type: inference                 # data | task | inference | pretrained_model | supervised_model | predictions
  name: pretrained_embeddings
created_at: 2026-07-30T14:02:11Z  # UTC, ISO 8601
command: infer
provenance:
  model_package: {name: my_model, version: 0.1.0}
  template_commit: 9d37393        # from .copier-answers.yml `_commit`
  git: {commit: abc1234, dirty: false}
  env:
    python: "3.12"
    packages: {meds: "0.4.1", meds-torch-data: "0.9.0", torch: "2.6.0"}
inputs:                           # every input artifact, by role — this is what makes the DAG verifiable
  - {role: input_data_dir, path: /runs/ex/data, manifest_digest: "sha256:…"}
  - {role: input_pretrained_model_dir, path: /runs/ex/models/pretrained, manifest_digest: "sha256:…"}
config:
  resolved: resolved_config.yaml
  digest: "sha256:…"
outputs:
  - {file: artifacts.parquet, rows: 41233, digest: "sha256:…"}
```

Recording each input's `manifest_digest` means a downstream command can detect that it is being run against
a *different* artifact than the one its sibling used, rather than silently producing an incoherent result.

### Type-specific fields

| Artifact type | Additional fields |
| --- | --- |
| `data` (`patients/`) | source `external_meds_dir`; subject count per split; vocabulary size; tensorization parameters |
| `inference` | **`kind`** — `embeddings \| trajectories \| hazards \| scores \| token_probabilities`; column schema (name → dtype → shape); the task subdirectory used, if any |
| `pretrained_model`, `supervised_model` | the `external_labels_dir` used (path + content digest) and its per-split label summary; checkpoint digest; monitored metric and best score; epochs, steps, seed; if warm-started: source directory **and the number of parameters actually matched** |
| `predictions` | **which prediction source was used** (see the arbitration rule under `predict`); `n_expected` and `n_written` per split; splits covered |

Two of those exist specifically to convert silent-wrong-answer failures into loud ones. Recording the
matched-parameter count on a warm start catches a checkpoint that loaded nothing and left the model randomly
initialized. Recording `n_expected` against `n_written` catches predictions that silently cover only part of
the task index.

Note that *which commands a model supports* is a property of the model package, not of any artifact, so it is
declared in the implementation and surfaced by `meds-model commands` — not in a manifest (see *Profiles*).
Because `patients/` is a model-specific representation, its manifest records the `model_package` that
produced it under `provenance`, which is what lets a command reject a `data_dir` built by a different model.

---

## Standard commands

### 1. `preprocess_data`

Convert external MEDS data into the model-specific patient representation.

Required parameters:

```yaml
external_meds_dir: ...
output_data_dir: ...
```

Example:

```bash
meds-model preprocess_data --config-name preprocess_data
```

Output:

```text
output_data_dir/patients/
output_data_dir/patients/metadata/
output_data_dir/patients/manifest.yaml
```

---

### 2. `pretrain`

Train a foundation model from patient data.

Required parameters:

```yaml
input_data_dir: ...
output_pretrained_model_dir: ...
```

Example:

```bash
meds-model pretrain --config-name pretrain
```

The global interface does not include `external_labels_dir`.

Models such as EveryQuery may perform their own model-specific query or target preprocessing internally. That behavior is outside the shared interface.

---

### 3. `infer`

Materialize reusable outputs from a pretrained model.

Required parameters:

```yaml
input_data_dir: ...
input_pretrained_model_dir: ...
output_inference_subdir: inference/<inference_name>
```

Optional parameter:

```yaml
external_labels_dir: ...
```

Example:

```bash
meds-model infer --config-name infer
```

The generated model determines what inference produces. Examples include:

* embeddings;
* generated trajectories;
* hazards;
* native scores;
* token probabilities.

No standard `inference.kind` parameter is required.

The artifact type must be recorded in:

```text
input_data_dir/inference/<inference_name>/manifest.yaml
```

---

### 4. `supervised_train`

Train a supervised model.

Required parameters:

```yaml
input_data_dir: ...
external_labels_dir: ...
output_supervised_model_dir: ...
```

Optional parameters:

```yaml
input_pretrained_model_dir: null
input_inference_subdir: null
```

Supported cases:

#### Train from scratch

```text
task data → supervised model
```

Neither optional input is provided.

#### Fine-tune a pretrained model

```text
task data + pretrained model → supervised model
```

`input_pretrained_model_dir` is provided.

#### Train a probe or downstream model

```text
task data + inference artifacts → supervised model
```

`input_inference_subdir` is provided.

#### Argument arbitration

Let `n` be the number of non-null values among `input_pretrained_model_dir` and `input_inference_subdir`.

* `n > 1` → error, naming both. Specialized hybrid methods that genuinely consume both must opt in explicitly by declaring so in the implementation; the default is to reject.
* `n == 1` → the source must be one the implementation declares support for. A from-scratch-only model handed an `input_pretrained_model_dir` must error, not silently ignore it.
* `n == 0` → train from scratch.

The check belongs in the shared command wrapper, before dispatch to the model hook, so no implementation can
bypass it. The resolved source is recorded in the output model manifest.

---

### 5. `predict`

Produce standardized predictions.

Required parameters:

```yaml
external_labels_dir: ...
output_predictions_dir: ...
```

Optional, and usually omitted:

```yaml
input_data_dir: null    # recovered from the source artifact's manifest when unset
```

`input_data_dir` is optional because a caller may not be able to name it. MEDS-DEV's
`model_initialization_dir` is a single rolling pointer that becomes the most recent *training* output,
which for a pretrain-then-finetune chain is not where the workspace lives. Since every training artifact
records the workspace it was built from, `predict` recovers it rather than requiring the caller to know —
and rather than re-tensorizing. A model with no source artifact (`packaged`) must be given it explicitly.

Optional prediction sources:

```yaml
input_pretrained_model_dir: null
input_supervised_model_dir: null
input_inference_subdir: null
```

Optional parameter:

```yaml
splits: null    # null → every split present in the task subdirectory
```

Supported cases:

#### Supervised prediction

```text
task data + supervised model → predictions
```

#### Direct zero-shot prediction

```text
task data + pretrained model → predictions
```

#### Materialized zero-shot prediction

```text
task data + inference artifacts → predictions
```

#### Packaged-model prediction

```text
task data + implementation-owned model → predictions
```

This covers PFN-style models whose weights are packaged with the repository. No external model-path argument is required in that case, but the implementation must still declare the packaged model, and the identifier of that model is recorded in the predictions manifest.

#### Argument arbitration

Let `n` be the number of non-null values among `input_pretrained_model_dir`, `input_supervised_model_dir`
and `input_inference_subdir`.

* `n > 1` → error, naming the conflicting sources. There is no precedence order; ambiguity is a caller bug.
* `n == 1` → the source must be one the implementation declares support for. A supervised-only model handed an `input_pretrained_model_dir` must error rather than ignore it.
* `n == 0` → valid **only** if the implementation declares a packaged model. Otherwise error, listing the sources it does support.

As with `supervised_train`, the check runs in the shared command wrapper before dispatch, and the resolved
source is recorded in the predictions manifest.

#### Coverage

`predict` produces one row per row of the selected splits of `external_labels_dir`. It must fail if it cannot,
rather than emitting a short file: silently dropping index rows that the model happens not to cover turns a
partial run into a plausible-looking complete one. The manifest records `n_expected` and `n_written` per
split so the invariant is checkable after the fact.

`splits` exists because a caller may want held-out predictions only; the default is every split present.

The output must contain a standardized:

```text
predictions.parquet
```

with at least:

```text
subject_id
prediction_time
predicted_boolean_probability
```

---

## Supported pipeline chains

### Simple supervised

```text
preprocess_data
→ supervised_train
→ predict
```

### Pretraining and fine-tuning

```text
preprocess_data
→ pretrain

preprocess_data
→ supervised_train using pretrained model
→ predict
```

### Pretrained representation probe

```text
preprocess_data
→ pretrain
→ infer embeddings

supervised_train using inference artifacts
→ predict
```

### Direct zero-shot

```text
preprocess_data
→ pretrain

predict using pretrained model
```

### Materialized zero-shot

```text
preprocess_data
→ pretrain
→ infer trajectories or native scores

predict using inference artifacts
```

### PFN-style

```text
preprocess_data
→ predict using packaged model
```

---

## Profiles

A profile is a **configuration**, not a code-generation branch.

Every generated repository ships all five commands. What distinguishes one profile from another is only which
commands are called and which optional inputs are supplied — and the parameter matrix below already captures
both. So a profile reduces to one config file:

```yaml
# configs/profile/finetune.yaml
supervised_train:
  input_pretrained_model_dir: ${models.pretrained}
predict:
  input_supervised_model_dir: ${models.supervised}
```

This matters because the alternative is what the current template does: a `profile` question presets a set of
`implements_*` booleans that then gate conditionals in the steps registry, the model module, the dependency
list, which config roots render, the MEDS-DEV specification, and the tests — with the profile-to-step mapping
duplicated once more in the template's own test suite. Adding a profile means editing all of them in sync.
Under the DAG, none of that branching is needed for the *interface*.

Two things remain genuinely model-specific and are still generated:

* the model class in the user-owned module (an autoregressive model and a supervised classifier are different code);
* the MEDS-DEV specification, whose command strings differ per profile.

The implementation also declares which commands it supports — the concern the current template's `STEPS`
registry serves — but as a flat list surfaced by `meds-model commands`, rather than as conditional imports.
An unsupported command exits with a clear error naming what the model does support.

---

## Configuration organization

Use one shared paths config and one separate config per command.

```text
configs/
├── paths/
│   └── default.yaml
├── preprocess_data.yaml
├── pretrain.yaml
├── infer.yaml
├── supervised_train.yaml
└── predict.yaml
```

### Shared paths config

```yaml
# configs/paths/default.yaml

data_dir: /runs/example/data

tasks:
  mortality: tasks/mortality_24h

inference:
  embeddings: inference/pretrained_embeddings

models:
  pretrained: /runs/example/models/pretrained
  supervised: /runs/example/models/supervised

predictions:
  mortality: /runs/example/predictions/mortality
```

### Command-specific config

```yaml
# configs/infer.yaml

defaults:
  - paths: default
  - _self_

input_data_dir: ${data_dir}
external_labels_dir: ${labels.mortality}
input_pretrained_model_dir: ${models.pretrained}
output_inference_subdir: ${inference.embeddings}
```

Each command remains independently executable:

```bash
meds-model preprocess_data --config-name preprocess_data
meds-model pretrain --config-name pretrain
meds-model infer --config-name infer
meds-model supervised_train --config-name supervised_train
meds-model predict --config-name predict
```

CLI overrides should remain supported:

```bash
meds-model pretrain \
  --config-name pretrain \
  trainer.max_epochs=50
```

Configuration precedence:

```text
template defaults
< model profile defaults
< shared paths config
< command-specific config
< CLI overrides
```

---

## Final parameter matrix

| Command            | Required parameters                                                       | Optional parameters                                                                  |
| ------------------ | ------------------------------------------------------------------------- | ------------------------------------------------------------------------------------ |
| `preprocess_data`  | `external_meds_dir`, `output_data_dir`                                    | —                                                                                    |
| `pretrain`         | `input_data_dir`, `output_pretrained_model_dir`                           | —                                                                                    |
| `infer`            | `input_data_dir`, `input_pretrained_model_dir`, `output_inference_subdir` | `external_labels_dir`                                                                  |
| `supervised_train` | `input_data_dir`, `external_labels_dir`, `output_supervised_model_dir`    | `input_pretrained_model_dir`, `input_inference_subdir`                               |
| `predict`          | `external_labels_dir`, `output_predictions_dir`                           | `input_data_dir`,  `input_pretrained_model_dir`, `input_supervised_model_dir`, `input_inference_subdir`, `splits` |

For `supervised_train` and `predict`, the optional model/inference sources are mutually exclusive; see the
argument arbitration rules in each command's section.

---

## Core design principles

1. External inputs use the `external_` prefix.
2. Independent artifact roots use `*_dir`.
3. Components stored inside `data_dir` use `*_subdir`.
4. Patient data is created once and is not copied by later commands.
5. Task and inference artifacts enrich the shared data directory.
6. Every command has a separate Hydra config.
7. Shared paths are factored into a reusable Hydra config group.
8. Model-specific preprocessing does not expand the global interface.
9. `infer` is optional and may occur before supervised training or prediction.
10. The command graph is a DAG, not one mandatory linear pipeline.
11. Each artifact directory carries its own manifest; there is no aggregate manifest to keep in sync.
12. Where a command accepts several alternative sources, exactly one is valid and the rest are an error — never a silent precedence.
13. A profile is a configuration, not a code path.

---

## Open questions

**MEDS-DEV mapping.** MEDS-DEV drives a `{unsupervised, supervised} × {train, predict}` command grid, passes
a `labels_dir` directly, and has no slot for a separate task-materialization step. Six commands and a DAG need
an explicit projection onto that grid. Verified against MEDS-DEV as of this writing:

* `labels_dir` is gated on **dataset type, not slot** — it is available in `supervised.train` as well as
  `supervised.predict`, so no separate task-preparation step is needed.
* `model_initialization_dir` rolls forward to the most recent non-predict run's output directory, which is
  why `predict` recovers `input_data_dir` from its source manifest instead of being told.
* The available placeholders are `dataset_dir`, `model_dir`, `demo`, `output_dir`, `labels_dir`,
  `model_initialization_dir` and `split`. There is **no task-name placeholder**, which is what makes
  zero-shot task resolution a model-side concern.

Still unvalidated: an actual end-to-end MEDS-DEV run against a generated repository.

**`infer` from non-pretrained sources.** `input_pretrained_model_dir` is required, so representations cannot
be materialized from a supervised model or a packaged model. This is deliberate: the packaged-model path is
served by `predict`, and probing a task-supervised model's representations is rare enough not to design for
now. Revisit only if a concrete model needs it.
