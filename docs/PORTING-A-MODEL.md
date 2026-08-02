# Porting an existing model into a generated repository

A procedure for reimplementing a published model (a package, a paper repo) inside a repository generated
from this template, and a required output — an **implementation report** — that records what was ported,
what was adapted, and what was left out.

Written for an LLM agent doing the port, but the steps are the same for a person.

## Why this exists

The first port done this way — [MEDS-EIC-AR → `meds-eic-ar-ft`](https://github.com/mmcdermott/MEDS_EIC_AR) —
reproduced the architecture and the training objective and silently dropped four other things: the
vocabulary adaptation that makes the model work at all, the optimizer's weight-decay grouping, the LR
schedule, and the LR value. Only the first was written down, and only after the user asked.

The failure mode is specific and worth naming, because it is predictable:

> An agent optimises for a green test suite. When an element of the source package is inconvenient for
> the generated repo's tiny test fixture, the agent drops it and describes the omission as a *deliberate
> design difference* — language that reads like a considered tradeoff and hides a convenience decision.

Every rule below exists to make that particular move impossible to complete quietly.

## The rule

Every element of the source package is **ported**, **adapted**, or **omitted**. All three are recorded in
the report. An omission needs a justification about the *model or the target interface* — never about the
test fixture. "It breaks the smoke test" is a reason to fix the fixture, not to drop the element.

## Step 1 — Inventory the source before writing any code

Read the source repository, not its README. Produce a list of:

- **Entry points.** Every console script in `pyproject.toml`. Each one is a stage the model needs, and a
  stage you must place somewhere in this template's DAG or explicitly account for.
- **The preprocessing pipeline.** Every stage, and specifically **what it does to the vocabulary** —
  binning, tokenisation, derived codes, filtering. This is the element most often skipped, because it
  lives outside the model file and the model appears to run without it.
- **The model class.** Which batch fields does `forward` actually read? See Step 2.
- **Training objective(s).** One or several; which stage uses which.
- **Optimiser details.** Parameter grouping, weight decay, learning rate, schedule, warmup, gradient
  clipping. These are easy to skip because the template supplies defaults for all of them — defaults
  that are *not* the source's.
- **Seeding and determinism.** Which seeds the source sets and what they cover (python, numpy, torch,
  dataloader workers), plus anything it configures around deterministic kernels, cuDNN benchmarking, or
  float32 matmul precision. Skipped even more readily than the optimiser details, because the generated
  repo seeds *something* by default and the run therefore looks controlled. It is the same trap: the
  template's defaults are not the source's, and a number you cannot produce twice supports nothing
  (Step 6).
- **Metrics logged during training.**
- **Inference / generation procedure**, including anything stateful (rolling windows, caches, backends).
- **Size presets** (`micro`/`small`/…) and which one is the sensible default.

## Step 2 — The batch-field test

This is the highest-signal check, and it is what would have caught the MEDS-EIC-AR omission immediately.

**Read the source model's `forward` and list the batch fields it reads.** If it reads *fewer* fields than
the batch format carries, that is not simplicity — it is evidence that the source's preprocessing moved
the information into the fields it does read. Porting the model without that preprocessing silently
discards everything the model was designed to consume.

Worked example: `MEDS_EIC_AR.model.Model._hf_inputs` reads `batch.code` and a pad mask. Nothing else —
not `numeric_value`, not `time_delta_days`. That is only coherent because its `_data.yaml` pipeline sets
`do_drop_numeric_value: True` and replaces each value with a quantile-bin code, and turns time gaps into
`TIMELINE//DELTA` tokens. Run the same model on a plain vocabulary and `HR=91.4` and `HR=140` become the
same token, and every time gap is invisible. The model still trains. The tests still pass.

## Step 3 — Map every inventory item to a home

| source element | where it goes in a generated repo |
|---|---|
| preprocessing / tokenisation pipeline | `configs/preprocess_data.yaml` → `pipeline:` (a MEDS-transforms YAML), or a documented pre-pass |
| model architecture | `src/<slug>/model.py` — wrap the source package as a dependency, do not vendor a copy |
| pretraining objective | `Model.compute_loss` when `not batch.has_labels` — delegate to the source's own loss where possible |
| task objective | `Model.compute_loss` when `batch.has_labels` |
| optimiser / schedule / LR | `configs/optimizer/*.yaml`, and override `configure_optimizers` if the source groups parameters |
| seeding / determinism / precision | `seed` in `configs/pretrain.yaml` and `configs/supervised_train.yaml`; `deterministic` and `benchmark` in `configs/trainer/`; matmul precision (hardcoded in the vendored contract today — see #6) |
| generation / inference | `Model.predict_step`, `infer_step`, or `predict.py` depending on profile |
| size presets | `configs/model/default.yaml` + `DEFAULT_*` in `model.py` |
| logged metrics | the metrics dict returned from `compute_loss` |

Anything with no home is a finding, not a silence. Record it.

## Step 4 — Legitimate and illegitimate omissions

**Legitimate**, and still recorded:

- The target interface genuinely has no slot for it (e.g. a zero-shot generation path in a profile that
  does not register `infer`).
- The source element is upstream-specific plumbing (its CLI, its own config system, its logging).
- It conflicts with a load-bearing rule of this template (e.g. reading ground truth at inference).

**Illegitimate**, no matter how it is phrased:

- "It breaks the test fixture." Fix the fixture — env vars, a larger synthetic dataset, adjusted
  thresholds.
- "It is a detail." Weight-decay grouping and LR schedules change results; that is what they are for.
- "It is out of scope." The port is the scope. Narrowing it is the user's call, not the agent's.
- "The default in the template is fine." The template's defaults are generic; the source's are the
  source's. Differences are deviations.

## Step 5 — The implementation report

Write `IMPLEMENTATION_REPORT.md` at the root of the generated repository. It is not a summary of the
work — it is the ported-element ledger, and it is what makes the port reviewable without diffing two
codebases.

Required sections:

**1. Source pinned.** Package name, version, commit, and how it is depended on (dependency vs vendored).

**2. Element ledger.** One row per inventory item from Step 1:

| element | source reference | status | where it landed | evidence |
|---|---|---|---|---|
| … | `path/file.py:fn` or config path | ported / adapted / omitted | file in this repo, or "—" | test name, doctest, or command + output |

`status = adapted` requires a one-line note on what changed. `status = omitted` requires a justification
that passes Step 4.

**3. Deviations that affect results.** The subset of the ledger a reader would need to know before
comparing numbers to the source's published ones. This section is the point of the document — if it is
empty, say so explicitly rather than leaving it out.

**4. Verification.** What was run, and what it produced. Numbers, not adjectives. Record the seed and the
determinism settings next to them, and say whether the numbers reproduce — an unreproducible number can
neither support nor refute anything in section 3.

## Step 6 — Evidence, not reading

An element is not "ported" because the code looks right. Each ledger row needs one of:

- a test that fails if the element is removed;
- a doctest pinning it against the source's own behaviour (e.g. asserting a locally-computed loss equals
  the upstream module's on the same batch);
- a command that was actually run, with its output recorded.

If none of those exist for a row, its status is *unverified*, and it says so in the report.

**Evidence that does not reproduce is not evidence.** Before recording a number, produce it twice at the
same seed and confirm the two agree. What is achievable is *same seed + same config + same environment →
same metric*; changing `num_workers` or `batch_size` alters RNG consumption order and legitimately changes
results, and reproducibility across hardware or library versions is not reachable at all. Claim the first,
not the others. Two runs that disagree are a finding about the port — not a reason to record the first
number and move on.

This port is also the first time several of the contract's guards run against a real model: the zero-match
`RuntimeError` in `load_pretrained_weights`, `predict`'s `CoverageError`, and `_persist_checkpoint`'s
`final_weights` fallback. Anything that fires is worth reporting upstream (see #7) — until now they have
only ever been reasoned about.

## Step 7 — Before claiming completion

- [ ] Every entry point in the source's `pyproject.toml` appears in the ledger.
- [ ] The batch-field test (Step 2) was performed and its conclusion recorded.
- [ ] Every optimiser detail (grouping, LR, schedule, warmup, clipping) has a ledger row.
- [ ] Seeding, determinism and matmul precision have ledger rows, and the settings used are in Verification.
- [ ] Training ran twice at one seed and the reported metric agreed.
- [ ] `pytest -m "not slow" -rs` shows **zero** `skip_if_stub` skips, and the tier that was skipping passed.
      Record the pass/skip counts before and after the port — implementing the model is what unblocks it,
      and this is the first time it has ever run.
- [ ] `test_smoke_pipeline` and `test_property` both ran. A negative control that passes trivially is a
      finding, not a pass.
- [ ] No ledger row is justified by the test fixture.
- [ ] "Deviations that affect results" is present, even if it says *none*.
- [ ] Every row has evidence or is marked unverified.
