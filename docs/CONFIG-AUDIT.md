# Hydra configuration audit — what needs inspecting

A generated repository's behaviour is configured almost entirely through Hydra. Some of those keys are
inert values handed to a constructor; others select a **branch inside the vendored contract** and change
what the commands actually do. Only the second kind can fail silently, and it is the second kind that is
neither documented in the generated README nor exercised by any test.

This document is the inventory. It is a checklist to work through, not a design proposal.

**Status of this audit:** derived 2026-08-02 against the template at `e95a667` plus two uncommitted fixes.
It was prompted by `pipeline=` being found broken — a branch that had never executed since it was written.

## How to re-derive it

Three greps reproduce everything below, so this list can be refreshed rather than trusted:

```bash
cd template

# 1. every config key the contract reads
grep -rhoE 'cfg\.get\("[a-z_]+"' src/meds_model_base/ | sort | uniq -c | sort -rn
grep -rhoE 'cfg\.[a-z_]+' src/meds_model_base/ | sort -u

# 2. every branch those keys drive
grep -rnE 'if [a-z_]*\bcfg\.|if bool\(cfg|if cfg\.' --include="*.py" src/meds_model_base/

# 3. what the generated repo tells its user about them
for k in <keys>; do grep -c "$k" README.md.jinja ../docs/design-interface.md; done
```

## 1. Inert keys — Hydra plumbing, no branch

Passed straight to a constructor or to another CLI. A wrong value fails immediately and legibly (a
`TypeError`, a Hydra composition error, an OOM). Nothing here can be *quietly* wrong.

`max_seq_len`, `batch_size`, `num_workers`, `seed` (as a value), most of `trainer.*`, `optimizer`,
`model.*`, every `output_*` path, and the `paths/` interpolations.

**Inspection needed: none beyond documentation.** Adding tests here tests Hydra, not the template. The
generated README mentions none of them, which is worth fixing for discoverability but is not a
correctness gap.

Three keys look inert and are not:

- **`logger`** — group selection. Which members exist depends on the copier answers (`csv` always;
  `wandb` / `mlflow` when asked for), and `instantiate_group` skips anything without a `_target_` child
  rather than raising — so a malformed member logs nothing and the run still succeeds. The generated
  repo's `test_every_rendered_logger_composes` and this repo's
  `test_logger_group_members_nest_under_their_own_name` are what cover it.
- **`callbacks`** — `instantiate_group` branches on the node being empty, not on its value. Benign, but
  note that dropping `model_checkpoint` silently changes which weights get published (see
  `_persist_checkpoint`, §2).
- **`trainer.deterministic` / `trainer.benchmark`** — the exception to "nothing here can be quietly
  wrong", and the reason `trainer.*` is qualified above. Neither appears in
  `configs/trainer/default.yaml`, so Lightning runs with `deterministic=False` and cuDNN autotunes its
  algorithms by timing. Nothing raises; the run simply is not reproducible, which makes every number it
  produces unusable as evidence. Note the trap in the other direction: `deterministic=true` *without*
  `CUBLAS_WORKSPACE_CONFIG=:4096:8` in the environment raises, but only on CUDA and only at the first
  matmul — so flipping the flag alone is not a fix.

## 2. Live keys — each selects a branch in the contract

These are the ones that matter. "Branch" column gives the exact site.

| key | commands | branch | non-default behaviour | in README | tested |
|---|---|---|---|---|---|
| `do_resume` | pretrain, supervised_train | `utils.py:51-55` | reuse scratch checkpoints from a previous attempt, or discard them | ✗ | ✗ |
| `work_dir` | pretrain, supervised_train | `utils.py:37` | relocate the scratch directory away from the output sibling | ✗ | ✗ |
| `clean_work_dir` | pretrain, supervised_train | `train.py:87`, `train.py:256` | keep the scratch tree instead of `rmtree`ing it | ✗ | ✗ |
| `pipeline` | preprocess_data | `preprocess_data.py:64` | runs an entire additional CLI (`MEDS_transform-pipeline`) before tensorization | ✗ | ✗ |
| `pipeline_overrides` | preprocess_data | argv extension | appends Hydra overrides to that CLI | ✗ | ✗ |
| `attach_labels` | predict | `predict.py:121` | emit predictions *without* `boolean_value` | ✗ | ✗ |
| `seed` | pretrain, supervised_train | `train.py:58`, `train.py:217` | `null` skips `seed_everything` entirely | ✗ | ✗ |
| `splits` | infer, predict | `_runtime.py:175-177` | accepts `null` / a bare string / a list | ✗ | list only |
| `input_*_dir` sources | supervised_train, predict | arbitration in `commands/base.py` | selects the source artifact; two is an error, wrong type is an error | ✓ | two-source ✓, wrong-type ✗ |
| `input_data_dir: null` | predict | `_runtime.py:74` | recovers the workspace from the source artifact's manifest | ✓ | ✓ (MEDS-DEV e2e) |
| `do_overwrite` | all | `manifest.py:346` | replace an existing artifact instead of refusing | ✗ | ✓ |

### Ranked by how quietly it fails

1. **`do_resume` / `work_dir` / `clean_work_dir`.** The only *silent* failure mode in the table: a resume
   that picks up an unrelated run's checkpoint yields a model that trains, predicts and looks entirely
   plausible. Everything else here fails loudly. Inspect first.
2. **Determinism settings** (`trainer.deterministic` / `trainer.benchmark`, §1, plus the hardcoded matmul
   precision below). Equally silent, and it degrades every other result: two runs at the same seed can
   disagree, so no measurement in the repository is reproducible evidence of anything.
3. **`pipeline` / `pipeline_overrides`.** Known broken until 2026-08-01; see §3.
4. **`attach_labels=false`.** Produces a file `meds_evaluation` rejects. Loud, but nothing tells a user
   the key exists or why they would want it.
5. **`seed`.** A reproducibility guarantee no test has ever checked. The default is `0`, not `null`, and
   `seed_everything` is called with `workers=True` — the seeding itself is right; it is the surrounding
   determinism that is not.
6. **`splits` as a bare string.** `resolve_splits` handles it; no test does.

### Related behaviour with no config key, and no test

Not config-driven, but in the same blind spot and worth inspecting together:

- `_persist_checkpoint`'s **`final_weights` fallback** (`train.py:300`) — when no `ModelCheckpoint`
  monitored anything, the last epoch's weights are published instead of a selected best. Recorded in the
  manifest, never asserted.
- `load_pretrained_weights`'s **zero-match `RuntimeError`** (`train.py:280`) — the load-bearing guard of
  the `finetune` profile: it is what stops a renamed encoder from becoming a silent from-scratch run.
- **`torch.set_float32_matmul_precision("medium")`** (`train.py:60`, and again at `train.py:219`) — set
  unconditionally, with no config key and no mention in any document. On Ampere and later this runs fp32
  matmuls in bfloat16. It is stable run-to-run, so it does not break same-machine reproducibility, but it
  silently diverges from any source implementation that ran at full precision — and because it lives in
  the vendored contract, a user can only change it by editing a file `copier update` overwrites. For the
  porting procedure in `PORTING-A-MODEL.md`, whose deliverable is a ledger of deviations that affect
  results, this is a deviation the template imposes invisibly.
- **`CoverageError`** (`predict.py:175`) — reachable in normal use; nothing exercises it.
- **`InferenceKind.scores` / `MaterializedPredictCommand`** — reachable only through
  `zero_shot_materialized`, which `skip_if_stub`s.

## 3. Config surfaces that are broken or dangling

**~~`use_wandb` / `use_mlflow` render no config.~~ Fixed.** Both `copier.yml` questions used to add only a
`pyproject.toml` extra. With no `configs/logger/wandb.yaml` or `mlflow.yaml` in the payload:

```
$ meds-model pretrain logger=wandb
hydra.errors.MissingConfigException: In 'pretrain': Could not find 'logger/wandb'
```

Answering "yes" bought a dependency and nothing to use it with. Both configs are now rendered under
`{% if use_wandb %}` / `{% if use_mlflow %}`, adapted from the ones MEDS-EIC-AR ships in
`configs/trainer/logger/` — with one deliberate difference: this template's group members nest under their
own name (`wandb:`), because `instantiate_group` reads the *children* of `cfg.logger`. A verbatim copy of
MEDS-EIC-AR's file would compose fine and log nothing.

Covered in both directions: `test_optional_loggers_render_with_their_extra` (config and extra appear
together, and neither appears when the answer is no), `test_logger_group_members_nest_under_their_own_name`
(the shape above), and, in a generated repo, `test_every_rendered_logger_composes` (`logger=<name>` for
every rendered name, which is the `MissingConfigException` itself).

**`do_reshard` is gone; split-sharded input is a precondition.** ~~`pipeline` + `do_reshard=true` cannot
work.~~ Inside a pipeline, MEDS-transforms resolves splits from the pipeline's *own* `input_dir` and from
`train/` shard prefixes, so it never copies `metadata/subject_splits.parquet` between stages — and its
output carries none. `MTD_preprocess`'s `reshard_to_split` needs one, so `do_reshard=true` could only ever
fail, and always *after* the pipeline had run.

Resolved by removing the key rather than guarding the combination. `preprocess_data` now refuses input
that is not sharded by split, up front and with the resharding command to run, and always calls MTD with
`do_reshard=false`. That is the one rule which holds with and without a `pipeline`, and it is already the
layout `meds-dev-dataset` and the standard MEDS ETL produce — verified against a real `meds-dev-dataset`
build, whose `data/` is `train/0.parquet`, `tuning/0.parquet`, `held_out/0.parquet`.

The premise of the old note was wrong, incidentally: `preprocess_data`'s docstring called non-split-sharded
input "the common case", and no dataset reachable through MEDS-DEV is.

**~~A filtering pipeline desynchronises the published split table.~~** `_preserve_subject_splits` copies
from `external_meds_dir` — the *original* dataset — so a pipeline that dropped subjects left `patients/`
claiming subjects its tokenization does not contain; `preprocess_data`, `pretrain` and `supervised_train`
all succeeded, then `predict` failed with `CoverageError`, an accurate guard pointing at the wrong culprit
two training runs too late.

Resolved in two places. `restrict_to_cohort` filters materialized labels down to the subjects present in
`tokenization/schemas/<split>/`, which is where meds-torch-data itself reads split membership — so
`n_expected` is already correct when `predict` counts. `_describe_cohort` now counts the tensorized output
rather than the source dataset, so the manifest records the cohort the artifact actually holds.

Measured end to end on the MIMIC-IV demo (100 subjects, 80/10/10) with
`filter_subjects(min_events_per_subject=500)`: 51 subjects survive, the manifest records 43/4/4, and real
`meds-dev-task` labels lose 11 rows with a warning naming the pipeline as the cause.

The table itself is still published and still needed — `meds-dev-task` output takes the shard branch of
`read_labels`, which partitions from it. Its remaining second job is diagnostic: a labelled subject absent
from *both* the table and the cohort means the labels are for another dataset, which is worth
distinguishing from one the pipeline merely dropped.

## 4. Full key inventory per command

For seeing the whole surface at once. `???` = required, no default.

| command | keys |
|---|---|
| `preprocess_data` | `external_meds_dir ???`, `output_data_dir`, **`pipeline`**, **`pipeline_overrides`**, `do_overwrite` |
| `pretrain` | `input_data_dir`, `output_pretrained_model_dir`, `max_seq_len`, `batch_size`, `num_workers`, **`seed`**, `do_overwrite`, **`do_resume`**, **`work_dir`**, **`clean_work_dir`** |
| `supervised_train` | the `pretrain` keys, plus `external_labels_dir ???`, `output_supervised_model_dir`, `input_pretrained_model_dir`, `input_inference_subdir` |
| `infer` | `input_data_dir`, `input_pretrained_model_dir`, `output_inference_subdir`, `external_labels_dir ???`, **`splits`**, `max_seq_len`, `batch_size`, `num_workers`, `do_overwrite` |
| `predict` | **`input_data_dir` (nullable)**, `external_labels_dir ???`, `output_predictions_dir`, **`attach_labels`**, the three `input_*` sources, **`splits`**, `max_seq_len`, `batch_size`, `num_workers`, `do_overwrite` |

Bold = live key from §2.

## 5. Where the tests would go

Not in `tests/` at the template root — those render a repo and inspect it statically; they never install
or execute one. The rendered conformance suite (`template/tests/`) is the only tier that runs real
commands, and it runs in the `rendered-smoke` CI job.

`preprocess_data` needs no torch, so `pipeline` coverage is cheap there. The `do_resume`
family needs a training run, so it belongs alongside `test_smoke_pipeline.py`.

## 6. Documentation gap, separately

Nine keys are reachable only by reading the YAML. The generated README documents the five commands and
their artifacts, then says "any extra argument is a Hydra override" — true, and the reason nobody
discovers `pipeline`. The §2 table is roughly the content a "Configuration" section of the generated
README should carry; the inert keys of §1 can be a one-line pointer at `configs/`.
