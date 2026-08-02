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
for k in <keys>; do grep -c "$k" README.md.jinja ../design-interface.md; done
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

- **`logger`** — group selection, but see §3: the only group member that exists is `csv`.
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
| `do_reshard` | preprocess_data | no template branch; selects a different **MTD** pipeline | `false` skips `reshard_to_split`, requiring split-sharded input | ✗ | `true` only, and only on pre-sharded data |
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

**`use_wandb` / `use_mlflow` render no config.** Both `copier.yml` questions only add a `pyproject.toml`
extra. There is no `configs/logger/wandb.yaml` or `mlflow.yaml` in the payload, so:

```
$ meds-model pretrain logger=wandb
hydra.errors.MissingConfigException: In 'pretrain': Could not find 'logger/wandb'
```

Answering "yes" buys a dependency and nothing to use it with. (MEDS-EIC-AR ships all three loggers under
`configs/trainer/logger/`.) Either render the configs under `{% if use_wandb %}`, or drop the questions.

**`pipeline` + `do_reshard=true` cannot work.** Inside a pipeline, MEDS-transforms resolves splits from
the pipeline's *own* `input_dir` and from `train/` shard prefixes, so it never copies
`metadata/subject_splits.parquet` between stages — and its output carries none.
`MTD_preprocess`'s `reshard_to_split` needs one, so the default `do_reshard=true` fails immediately
after the pipeline succeeds. The rule is conditional, not a global precondition:

- no `pipeline` → `do_reshard=true` is correct, and is what makes raw MEDS work;
- with `pipeline` → `do_reshard=false`, and the pipeline output must be split-sharded, either because
  its input was (pipelines preserve shard layout) or because the pipeline starts with `reshard_to_split`.

Worth a guard that raises on the impossible combination rather than failing minutes in.

**A filtering pipeline desynchronises the published split table.** `_preserve_subject_splits` copies from
`external_meds_dir` — the *original* dataset — so if the pipeline dropped subjects, `patients/` claims
subjects its tokenization does not contain. Measured on the synthetic dataset with
`filter_subjects(min_events_per_subject=8)`: 280 subjects in, 145 survive, 280 published; `preprocess_data`,
`pretrain` and `supervised_train` all succeed, then `predict` fails with `CoverageError` — an accurate
guard pointing at the wrong culprit, two training runs too late. Publishing the intersection with the
tokenized cohort (and failing on an empty split) makes the artifact self-consistent by construction.

**The default `do_reshard` path is untested on the data shape it exists for.** The rendered tests use
`SIMPLE_STATIC_SHARDED_BY_SPLIT_WITH_TASKS`, which is already sharded by split. `preprocess_data`'s own
docstring says non-split-sharded input "is the common case" and is the reason `do_reshard=true` is the
default — that case has never run in CI.

## 4. Full key inventory per command

For seeing the whole surface at once. `???` = required, no default.

| command | keys |
|---|---|
| `preprocess_data` | `external_meds_dir ???`, `output_data_dir`, **`pipeline`**, **`pipeline_overrides`**, **`do_reshard`**, `do_overwrite` |
| `pretrain` | `input_data_dir`, `output_pretrained_model_dir`, `max_seq_len`, `batch_size`, `num_workers`, **`seed`**, `do_overwrite`, **`do_resume`**, **`work_dir`**, **`clean_work_dir`** |
| `supervised_train` | the `pretrain` keys, plus `external_labels_dir ???`, `output_supervised_model_dir`, `input_pretrained_model_dir`, `input_inference_subdir` |
| `infer` | `input_data_dir`, `input_pretrained_model_dir`, `output_inference_subdir`, `external_labels_dir ???`, **`splits`**, `max_seq_len`, `batch_size`, `num_workers`, `do_overwrite` |
| `predict` | **`input_data_dir` (nullable)**, `external_labels_dir ???`, `output_predictions_dir`, **`attach_labels`**, the three `input_*` sources, **`splits`**, `max_seq_len`, `batch_size`, `num_workers`, `do_overwrite` |

Bold = live key from §2.

## 5. Where the tests would go

Not in `tests/` at the template root — those render a repo and inspect it statically; they never install
or execute one. The rendered conformance suite (`template/tests/`) is the only tier that runs real
commands, and it runs in the `rendered-smoke` CI job.

`preprocess_data` needs no torch, so `pipeline` / `do_reshard` coverage is cheap there. The `do_resume`
family needs a training run, so it belongs alongside `test_smoke_pipeline.py`.

## 6. Documentation gap, separately

Nine keys are reachable only by reading the YAML. The generated README documents the five commands and
their artifacts, then says "any extra argument is a Hydra override" — true, and the reason nobody
discovers `pipeline`. The §2 table is roughly the content a "Configuration" section of the generated
README should carry; the inert keys of §1 can be a one-line pointer at `configs/`.
