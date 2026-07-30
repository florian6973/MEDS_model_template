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

Canonical MEDS dataset with existing subject splits.

Expected splits:

```text
train
tuning
held_out
```

### `external_task_file`

Either:

* an ACES task YAML; or
* an already-materialized parquet file containing at least:

```text
subject_id
prediction_time
boolean_value
```

Model-specific task columns are allowed.

---

## Internal artifact types

### Data directory

`data_dir` is a shared workspace containing patient data and derived artifacts.

```text
data_dir/
├── patients/
│   ├── metadata/
│   └── manifest.yaml
├── tasks/
│   └── <task_name>/
│       ├── train.parquet
│       ├── tuning.parquet
│       ├── held_out.parquet
│       └── manifest.yaml
└── inference/
    └── <inference_name>/
        ├── artifacts.parquet
        └── manifest.yaml
```

The patient data is created once. Task preprocessing and inference add new subdirectories without copying the patient data.

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
| `preprocess_task` | appends `tasks/<task_name>/` |
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
| `task` | source `external_task_file`; `materialization: aces_extracted \| passed_through` (which of the two `external_task_file` forms was supplied); label count and positive rate per split; `prediction_time` range |
| `inference` | **`kind`** — `embeddings \| trajectories \| hazards \| scores \| token_probabilities`; column schema (name → dtype → shape); the task subdirectory used, if any |
| `pretrained_model`, `supervised_model` | checkpoint digest; monitored metric and best score; epochs, steps, seed; if warm-started: source directory **and the number of parameters actually matched** |
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

### 2. `preprocess_task`

Materialize or validate a task against already-preprocessed patient data.

Required parameters:

```yaml
input_data_dir: ...
external_task_file: ...
output_task_subdir: tasks/<task_name>
```

Example:

```bash
meds-model preprocess_task --config-name preprocess_task
```

Output:

```text
input_data_dir/tasks/<task_name>/
```

`output_task_subdir` is relative to `input_data_dir`.

This command must not copy or rewrite `patients/`.

---

### 3. `pretrain`

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

The global interface does not include `input_task_subdir`.

Models such as EveryQuery may perform their own model-specific query or target preprocessing internally. That behavior is outside the shared interface.

---

### 4. `infer`

Materialize reusable outputs from a pretrained model.

Required parameters:

```yaml
input_data_dir: ...
input_pretrained_model_dir: ...
output_inference_subdir: inference/<inference_name>
```

Optional parameter:

```yaml
input_task_subdir: tasks/<task_name>
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

### 5. `supervised_train`

Train a supervised model.

Required parameters:

```yaml
input_data_dir: ...
input_task_subdir: tasks/<task_name>
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

### 6. `predict`

Produce standardized predictions.

Required parameters:

```yaml
input_data_dir: ...
input_task_subdir: tasks/<task_name>
output_predictions_dir: ...
```

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

`predict` produces one row per row of the selected splits of `input_task_subdir`. It must fail if it cannot,
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
→ preprocess_task
→ supervised_train
→ predict
```

### Pretraining and fine-tuning

```text
preprocess_data
→ pretrain

preprocess_data
→ preprocess_task
→ supervised_train using pretrained model
→ predict
```

### Pretrained representation probe

```text
preprocess_data
→ pretrain
→ infer embeddings

preprocess_task
→ supervised_train using inference artifacts
→ predict
```

### Direct zero-shot

```text
preprocess_data
→ pretrain

preprocess_task
→ predict using pretrained model
```

### Materialized zero-shot

```text
preprocess_data
→ pretrain
→ infer trajectories or native scores

preprocess_task
→ predict using inference artifacts
```

### PFN-style

```text
preprocess_data
→ preprocess_task
→ predict using packaged model
```

---

## Profiles

A profile is a **configuration**, not a code-generation branch.

Every generated repository ships all six commands. What distinguishes one profile from another is only which
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
├── preprocess_task.yaml
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
input_task_subdir: ${tasks.mortality}
input_pretrained_model_dir: ${models.pretrained}
output_inference_subdir: ${inference.embeddings}
```

Each command remains independently executable:

```bash
meds-model preprocess_data --config-name preprocess_data
meds-model preprocess_task --config-name preprocess_task
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
| `preprocess_task`  | `input_data_dir`, `external_task_file`, `output_task_subdir`              | —                                                                                    |
| `pretrain`         | `input_data_dir`, `output_pretrained_model_dir`                           | —                                                                                    |
| `infer`            | `input_data_dir`, `input_pretrained_model_dir`, `output_inference_subdir` | `input_task_subdir`                                                                  |
| `supervised_train` | `input_data_dir`, `input_task_subdir`, `output_supervised_model_dir`      | `input_pretrained_model_dir`, `input_inference_subdir`                               |
| `predict`          | `input_data_dir`, `input_task_subdir`, `output_predictions_dir`           | `input_pretrained_model_dir`, `input_supervised_model_dir`, `input_inference_subdir`, `splits` |

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
an explicit projection onto that grid — in particular where `preprocess_task` runs, and whether the
`data_dir` survives between the train and predict invocations. This should be settled against a live
MEDS-DEV run before the interface is frozen.

**`infer` from non-pretrained sources.** `input_pretrained_model_dir` is required, so representations cannot
be materialized from a supervised model or a packaged model. This is deliberate: the packaged-model path is
served by `predict`, and probing a task-supervised model's representations is rare enough not to design for
now. Revisit only if a concrete model needs it.
