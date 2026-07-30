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
├── tasks/
│   └── <task_name>/
│       ├── train.parquet
│       ├── tuning.parquet
│       └── held_out.parquet
├── inference/
│   └── <inference_name>/
│       ├── artifacts.parquet
│       └── manifest.yaml
└── manifest.yaml
```

The patient data is created once. Task preprocessing and inference add new subdirectories without copying the patient data.

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
* Every artifact should record its provenance in a manifest.

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
output_data_dir/metadata/
output_data_dir/manifest.yaml
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

Both optional inputs may be supported for specialized hybrid methods, but ordinary profiles should use at most one.

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

This covers PFN-style models whose weights are packaged with the repository. No external model-path argument is required in that case, but the implementation must still declare the packaged model in its manifest.

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
| `predict`          | `input_data_dir`, `input_task_subdir`, `output_predictions_dir`           | `input_pretrained_model_dir`, `input_supervised_model_dir`, `input_inference_subdir` |

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
