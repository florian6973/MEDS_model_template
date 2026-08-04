# Predicate Featurization and the `data_backend` Option

**Status: §§1–9 implemented (2026-08-04) — steps 1–5 of §11; TECO adoption (step 6) pending.**
Companion to [design-interface.md](design-interface.md), which remains the authoritative spec for the
command graph, artifact layout and manifests. This document specifies an *addition*: a second data
backend for `preprocess_data` and the changes that make the rest of the contract survive it.

Drafted 2026-08-04 from the TECO port discussion (Florent Pollet / Claude).

---

## 1. Motivation

The template's only data path today is MEDS-transforms (optional) → meds-torch-data tensorization
("MTD"). Every downstream command consumes the tensorized layout through the MTD datamodule, and the
feature axis a model sees is the **code vocabulary** of whatever dataset it is handed.

That is the right default for foundation-style sequence models, and the wrong shape for the large class
of published clinical models built on **named clinical variables**. The concrete case is TECO
(`models/teco`): the paper uses 11 named variables on a 15-minute grid; the MEDS port substitutes the
whole code vocabulary because a MEDS model cannot resolve "heart rate" against an arbitrary dataset.

The MEDS ecosystem already has the resolution layer: per-dataset ACES predicates
(`MEDS-DEV/src/MEDS_DEV/datasets/<name>/predicates.yaml`) map abstract names (`creatinine`, `sodium`)
to dataset codes, and task configs bind to them through `???` placeholders. But predicates are only ever
consumed for **task extraction**. This proposal reuses them for **featurization**: `preprocess_data`
stamps one 0/1 column per predicate onto the MEDS data, and the model decides everything else —
whether to read `numeric_value` on active rows, how to aggregate, what temporal structure to impose.

What this deliberately is *not* (v1 non-goals):

* **No value channels, units, or transformations.** Presence only. The original rows and columns are
  untouched, so a model can read `numeric_value` wherever a predicate column is 1. A future value/
  expression layer (candidate: [dftly](https://github.com/mmcdermott/dftly)) composes on top without
  changing this contract.
* **No temporal aggregation.** Grids, windows, carry-forward, imputation are model-side.
* **No MEDS-DEV changes.** See §9.
* **No `es-aces` dependency.** See §5.

---

## 2. The `data_backend` copier option

New `copier.yml` question:

```yaml
data_backend:
  type: str
  help: "How preprocess_data materializes patient data"
  choices:
    "mtd — MEDS-transforms + meds-torch-data tensorization (the default)": "mtd"
    "custom_featurization — predicate columns on MEDS parquet; you write the datamodule": "custom_featurization"
  default: "mtd"
```

A choice rather than a boolean (`include_mtd`) so a third backend can land later without a second
vocabulary, and so the answer maps one-to-one onto the runtime `featurization` config value and the
manifest's `representation` field.

### What the option gates (render-time, following the `use_wandb` pattern)

| Rendered surface | `mtd` | `custom_featurization` |
|---|---|---|
| `meds-torch-data` in `pyproject.toml` / `requirements.txt` | present | **absent** |
| `configs/datamodule/patients.yaml`, `task.yaml` (MTD targets) | rendered | not rendered |
| `configs/datamodule/` custom stub + `src/<slug>/datamodule.py` stub | not rendered | rendered, in `_skip_if_exists` |
| `configs/preprocess_data.yaml` default | `featurization: mtd` | `featurization: predicates` |
| `predicates.yaml` starter (user-owned, `_skip_if_exists`) | rendered (the equivalence guard uses it) | rendered and wired into the test fixtures |
| `model.yaml` | as today | first command of each chain adds `external_predicates_file={predicates_path}` (§9) |
| Rendered `CLAUDE.md` / `PORTING-A-MODEL.md` backend sections | MTD text | custom text |

The dependency removal is the real payoff: MEDS-DEV builds each model's venv from `requirements.txt`,
so a featurized model's benchmark environment does not install torch-adjacent MTD machinery it never
imports.

### What the option must NOT do

**No Jinja conditionals inside `src/meds_model_base/`.** The vendored contract stays one unconditional
codebase containing both paths, copied byte-for-byte as today. Reasons, in order of weight:

1. `copier update` 3-way merges the contract into existing repos; a render-conditional contract adds a
   second variant axis to every merge.
2. The render/test matrix doubles (7 profiles × 2 backends) for every structural guarantee in
   `tests/test_render.py`, and Jinja-in-Python is the exact artefact class that produced the
   docstring-collapse bug recorded in CLAUDE.md.
3. There is no runtime win. Import-weight discipline already defers the heavy imports; with the import
   guard of §6.2, unused MTD branches cost nothing when the package is absent — the same price
   `ProbeTrainCommand` costs a supervised model today.

Invariant: **one `meds_model_base`, tolerant of either dependency set; `data_backend` decides what is
installed and which configs and stubs render.** A repo that flips its answer later runs `copier update`
and the configs backfill, exactly like `use_wandb`.

---

## 3. `preprocess_data`: the `predicates` branch

New config keys on `preprocess_data` (both backends parse them; the rendered default differs):

```yaml
featurization: mtd            # mtd | predicates
external_predicates_file: null
```

* `featurization=mtd` — byte-identical to today's behavior. `external_predicates_file` is ignored.
* `featurization=predicates` with `external_predicates_file=null` — **error**, before any work. A
  silent passthrough would publish a valid-looking artifact with zero feature columns.
* `featurization=predicates` with a file — the branch below.

`external_meds_dir` is **read-only**, as every `external_` input is: the featurizer reads input shards
and writes augmented copies into the `write_artifact` staging directory under `output_data_dir` (the
temp sibling of `patients/` that is renamed into place atomically — the same mechanism the MTD branch
uses, and where the optional pipeline's intermediate already lives). Nothing is ever written next to,
or into, the source dataset.

Branch behavior, replacing `_run_mtd` + `_validate_tensorized` after the (unchanged, still optional)
MEDS-transforms `pipeline:` stage:

1. Parse `external_predicates_file` with the reader of §4.
2. For each shard `data/<split>/<shard>.parquet` of the (possibly pipeline-transformed) input: append
   the predicate columns, write the same relative path into the staging directory. Pure augmentation —
   no rows dropped, no columns modified, no resharding. (Row filtering is a MEDS-transforms pipeline
   job that already exists; it does not get a second home here.)
3. Copy `metadata/codes.parquet` and `metadata/dataset.json` through. **Do not copy
   `metadata/subject_splits.parquet`.** Split membership travels as the shard layout and nothing else —
   the same invariant the MTD branch has, asserted by `test_workspace_is_published_with_manifests`: a
   copied splits file describes the *input* to preprocessing, and a filtering pipeline makes those two
   different things.
4. Write `features.json` at the artifact root — the authoritative, **ordered** definition of this
   artifact's feature space:

   ```json
   {
     "version": 1,
     "features": [
       {"name": "icu_admission", "column": "predicate//icu_admission"},
       {"name": "creatinine",    "column": "predicate//creatinine"}
     ]
   }
   ```

   Consumers: the datamodule (`vocab_size = len(features)`; columns selected **in this order**),
   `_validate_featurized` (every listed column present in every shard), and the tests. The file exists
   instead of "scan a shard for `predicate//` columns" for one load-bearing reason: **feature order is
   part of what a trained checkpoint means.** Feature index *i* at training time must be the same
   predicate at predict time; parquet column order is incidental, `features.json` is the contract. It
   also keeps consumers decoupled from the column-naming convention and makes the artifact
   self-describing without opening parquet. It is data-plane metadata, which is why it is a file the
   datamodule reads rather than a manifest field: the manifest is provenance and gating
   (`read_manifest`), and holds only `n_features` and the digest (§ below).
5. Validate (`_validate_featurized`, the sibling of `_validate_tensorized`): every declared predicate
   column present in every shard, shard layout split-sharded, `features.json` consistent. Per-predicate
   match counts are logged, and a predicate that matched **nothing** across all shards is a prominent
   **warning, not an error**: on real data it signals a binding mismatch worth reading, but it is also
   the expected outcome when real-vocabulary predicates run over synthetic test data (§8), and the
   machinery being exercised there does not depend on columns being non-zero.
6. Publish through the same `write_artifact` context manager, same artifact name (`patients`), same
   `ArtifactType.data` — so `predict`'s `read_manifest` check passes unchanged. Manifest `extras` gain:

```yaml
representation: predicates          # "mtd" on the other branch — new field on BOTH branches
featurization:
  predicates_file: <path>
  predicates_digest: <sha256>
  n_features: <int>
  skipped: []          # predicate names read but not featurized (§4 skip policy) — the feature-space
                       # provenance beyond the run's log; the §8 validation test asserts this would be
                       # empty for the repo's own predicates.yaml
  match_counts: {}     # events matched per predicate across all shards — makes degenerate (all-zero)
                       # features visible in the artifact itself, not only in the run's warning
```

`representation` is the dispatch key for §6.1 and the fail-fast guard replacing `_validate_tensorized`
downstream: a command whose datamodule expects the other flavor fails at the manifest, not minutes in.

`predicates_digest` exists because the manifest otherwise records only a *path* to a file that lives
outside the artifact and can change after the build: edit `predicates.yaml`, rerun without
`do_overwrite`, and the kept artifact silently describes predicates that no longer exist. The digest
makes "which predicates produced this artifact" answerable and two experiment runs distinguishable —
the same reason `train.py` already digests `external_labels_dir` into its manifests. The full column
list deliberately does **not** appear in the manifest: `features.json` is the single machine-readable
definition of the feature space (datamodules read it, `vocab_size` derives from it, validation checks
shards against it), and a manifest copy would be a second source of truth waiting to drift.

### Resulting artifact layout

```text
patients/
  manifest.yaml                     # representation: predicates
  features.json
  data/train/0.parquet              # original columns + predicate//<name> Int8 columns
  data/tuning/0.parquet
  data/held_out/0.parquet
  metadata/codes.parquet
  metadata/dataset.json
```

Still a valid MEDS dataset (the MEDS `DataSchema` is open to extra columns), so any MEDS tool —
including a later MTD run — can consume it.

---

## 4. The predicates reader

A small, hand-rolled parser in `meds_model_base` (new module, e.g. `featurize.py`). Input is a YAML
file whose `predicates:` mapping uses ACES syntax; the reader is **pure code matching plus `or()`**:

| Form | Example | Semantics |
|---|---|---|
| exact code | `code: LAB//50912//mg/dL` | `code == literal` |
| regex | `code: { regex: "^ICU_ADMISSION//.*" }` | `str.contains` — anchors live in the pattern, matching ACES's own semantics |
| any-of | `code: { any: [HR, PULSE] }` | membership |
| derived `or` | `expr: or(creatinine_1, creatinine_2)` | column-wise max of already-computed plain columns |

`or()` is in scope from v1 because multi-code concepts are the norm, not the exception — `creatinine`
is two codes and `sodium` three in the MIMIC-IV predicates file; without it the concept layer fragments
back into per-source-code columns, which is the problem this feature exists to solve.

**Value bounds (`value_min`/`value_max` + inclusivity) are deliberately NOT parsed in v1**, although
ACES counts them as plain. A bounded predicate (`abnormally_high_creatinine`) bakes a clinical
threshold into featurization, and thresholds are exactly the semantics this design declares model-side:
the model reads `numeric_value` on active rows and learns thresholds itself. Dropping bounds also keeps
the reader free of numeric edge cases (null values, inclusivity). They return with the future value-
channel layer (§10), where thresholds belong. Consequence: bounded entries fall under the
unsupported-entry policy below — skipped by default with a warning and a manifest record — so a raw
dataset file remains usable, while a **curated featurization predicates file** stays the recommended
input and the required form for the repo's own file: that file is the model's declaration of its
concepts (TECO's 11 variables), not a dump of everything the dataset can express.
Concretely it is the repo's own `predicates.yaml` (§2): rendered as a user-owned starter with working
example predicates over the synthetic test vocabulary, then replaced by the author with their real
concepts. **The tests always read this file** (§8) — there is no separate test-only predicates file,
so the file the model runs with in production is the file the tests exercise.

The unsupported-entry policy, stated precisely because it matters: the reader always parses the
**entire** file first, then applies one of two deterministic behaviors. **Default (skip)**: entries
using an unsupported form — value bounds, `and()`, sequential/temporally-scoped derived predicates,
`???` placeholders, any key the reader does not recognize — are skipped with a **warning listing them
by name**, **cascading** (an `expr` referencing a skipped predicate is itself skipped and logged), and
the skipped names are recorded in the manifest (`featurization.skipped`, §3) so the feature-space
provenance outlives the log. With `featurization_strict: true`: any unsupported entry is instead a
**hard error listing every offending predicate by name**, before any shard is touched. In **both**
modes, ending with zero featurizable predicates is a hard error — an empty feature space is never a
valid artifact. There is no third mode — entries are never dropped without having been read and named.

Skip is a safe default only because the loss is not silent — it is named in the log and in the
manifest — and because strictness is enforced where the author's intent actually lives: the rendered
predicates-file validation test (§8) asserts the repo's **own** `predicates.yaml` parses with **zero
skips**. Unsupported forms in your own declaration are a bug in your file; the same forms in a foreign
file at runtime (a raw dataset `predicates.yaml` in a MEDS-DEV run, §9) degrade gracefully, with the
presence signal flowing through the plain sibling predicates those files already contain.

**Considered and rejected: reading bounded predicates but ignoring their bounds** (treating
`abnormally_high_creatinine_1: {code: ..., value_min: 1.3}` as its code matcher). Two reasons. First,
the predicate *name* carries the threshold semantics: a `predicate//abnormally_high_creatinine` column
that actually means "any creatinine measurement" is a lie under a trustworthy name, propagated through
`features.json` to everyone who interprets the model — a feature lost visibly (skip) is strictly
better than a feature kept invisibly wrong. Second, in presence-only space it buys nothing: bounded
predicates share codes with their unbounded siblings (MIMIC-IV's `creatinine_1` vs
`abnormally_high_creatinine_1` are the same code), so ignoring bounds manufactures byte-identical
duplicate columns under different names, while skip mode already keeps the presence signal via the
plain siblings. The one losing case — a file defining only the bounded form of a concept — is fixed by
one line of curation (add the plain predicate; the skip log should say exactly that), and when value
bounds return with the value-channel layer they will apply *as stated*, with no version boundary where
a column changes meaning under an unchanged name.

Column naming: `predicate//<name>`, dtype `Int8`, dense 0/1 (not null/1 — dense columns keep model-side
logic trivial). The reader validates that no generated column collides with an existing column and that
no two predicates share a name.

**Why not depend on `es-aces`:** the template's standing rule is that it never parses ACES *task*
definitions (they need dataset predicates it does not have). That reasoning does not forbid parsing a
predicates file handed in explicitly — but the dependency does drag in its own `meds` pin (MEDS-DEV
pins `meds==0.3.3` against the template's `meds~=0.4`), and the supported subset above is ~50 lines of
polars. If the subset ever grows toward real ACES semantics, revisit; do not reimplement ACES here.

---

## 5. What breaks without MTD — the four seams

Established by tracing every consumer of the `patients` artifact. The contract layer — manifests,
`write_artifact`/`read_manifest`, `read_labels`/`split_labels`, coverage checking, `PredictionSchema`,
`resolve_workspace` — is representation-agnostic and needs **no changes**. Four seams are MTD-shaped:

1. **`tasks.tokenized_cohort()` — the one hard failure.** Reads
   `patients/tokenization/schemas/<split>/*.parquet`; raises "not a tensorized cohort" otherwise.
   Called from `materialize_labels`, i.e. from **`supervised_train`, `infer` and `predict`** — every
   task-conditioned command dies before touching the model.
2. **The datamodule.** All commands call `build_datamodule(cfg)`; the shipped datamodule configs
   target MTD's `Datamodule`; commands patch `cfg.datamodule.config.task_labels_dir` at runtime.
3. **`vocab_size`.** Both training commands' `build_module` pass
   `vocab_size=datamodule.config.vocab_size` — an MTD config attribute.
4. **Prediction key alignment.** `_runtime.run_predict_step` takes `(subject_id, prediction_time)`
   from `dataset.schema_df`, in loader order, via the `SPLIT_ATTRS` attribute names.

---

## 6. Contract changes (all in `meds_model_base`, both backends, unconditional)

### 6.1 `cohort_subjects()` — manifest-driven split membership

Replace the direct `tokenized_cohort()` call inside `materialize_labels` with a dispatcher:

```python
def cohort_subjects(patients_dir: Path) -> dict[str, set[int]]:
    match read_manifest(patients_dir).extras["representation"]:  # sketch
        case "mtd":
            return tokenized_cohort(patients_dir)  # today's reader
        case "predicates":
            return featurized_cohort(patients_dir)  # subject_id off data/<split>/*.parquet
```

`featurized_cohort` reads subject ids from the data shards themselves — the same
shard-path-is-the-authority philosophy, different files. The documented guarantee of `split_labels`
(labels partitioned by the cohort the model will actually see, so `CoverageError` cannot fire two
training runs late) carries over verbatim. A manifest without `representation` (artifact predating this
change) is treated as `mtd`.

### 6.2 Import guard in `lightning.register_structured_configs`

The dispatcher calls it before composing **every** command's config, and it imports `meds_torchdata`
unconditionally — with MTD uninstalled, every command dies at import, including `--help`. Change: if
`meds_torchdata` is not importable, skip registration and return. The `MEDSTorchDataConfig` config-store
group is referenced only by the MTD datamodule yamls, which a custom-backend repo does not render; if a
user composes one anyway, Hydra's missing-group error names the config, which is the right message.
`build_datamodule` itself is `instantiate(cfg.datamodule)` and needs nothing.

### 6.3 `vocab_size` stays, by protocol

The protocol (§7) requires `config.vocab_size`; a featurized datamodule sets it to the feature count
read from `features.json`. The name is slightly wrong for features, but renaming a contract kwarg
ripples into every generated model's `build_module` signature for zero behavior — not now. A model
wanting a different signature already has the designed escape hatch: subclass the command in user-owned
`commands.py` and override `build_module`.

### 6.4 Predict alignment stays, by protocol

`run_predict_step`, `stack_outputs`, `_check_coverage` and the whole `_PredictRunMixin` flow are
untouched provided the dataset exposes `schema_df` (§7). Two error strings that say "tensorized cohort"
get softened to name both layouts.

---

## 7. The datamodule protocol

The duck-typed surface the commands already rely on, promoted to a named, documented `Protocol` in
`meds_model_base` so a custom datamodule knows exactly what to implement. A conforming datamodule:

* is a `lightning.pytorch.LightningDataModule`;
* has a `config` object with:
  * `task_labels_dir` — **settable at runtime**; consumes the `{split}.parquet` layout that
    `materialize_labels` writes (this layout is the interface; `boolean_value` may be absent — that is
    how inference-without-ground-truth reaches the batch);
  * `vocab_size` — feature-space size (`len(features.json)` for the predicates backend);
* accepts `batch_size` and `num_workers` (the conformance harness and every command pass them);
* exposes the `SPLIT_ATTRS` surface: `train_dataset`/`train_dataloader`, `val_dataset`/`val_dataloader`
  (tuning), `test_dataset`/`test_dataloader` (held_out);
* each dataset exposes `schema_df`: a polars frame with `subject_id`, `prediction_time` rows **in
  loader iteration order** (predict loaders must not shuffle) — this is what lets `predict` align model
  outputs to timepoints without models hand-aligning anything.

The template ships the protocol and a rendered stub implementing none of it (`datamodule.py`, in
`_skip_if_exists`) — consistent with the no-model rule: the aggregation strategy over predicate columns
*is* modeling, so it lives in user-owned code.

---

## 8. Tests

Guiding principle: both backends' contract code ships in every repo, so **contract tests run
everywhere**; only dependency-bound tests condition on the environment, via
`pytest.importorskip("meds_torchdata")` — the dependency analogue of `skip_if_stub`: green, but honest.

### Tier 1 — template repo `tests/` (structural, no torch, seconds)

* Parametrize the render tests over the full **7 profiles × 2 backends** (cheap here).
* Per-backend assertions: dependency present/absent in rendered `pyproject.toml`/`requirements.txt`;
  correct datamodule configs/stubs rendered; correct `featurization` default; fixtures wired.
* `test_rendered_repo_is_clean_as_written` runs on both — the new Jinja conditionals are exactly the
  artefact class it exists to catch.

### Tier 2 — generated-repo conformance suite

The predicates path needs **no extra dependencies**, so it is testable in every repo regardless of the
copier answer; the MTD path is testable only where MTD is installed.

* **One predicates file, the model's own.** The tests read the repo's `predicates.yaml` — the same
  file production runs use — never a test-only copy; a parallel test fixture would mean the model's
  real predicates are the one thing the suite never exercises. The rendered *starter* content of that
  file doubles as the working example: every reader form over the stable `meds_testing_helpers`
  synthetic vocab — exact (`HR`), regex (`^ADMISSION//.*`), any-of (`{ any: [HR, TEMP] }`),
  `or(hr, temp)` — so a fresh repo passes out of the box, and an author who replaces the content keeps
  the same tests running against *their* concepts (all-zero columns over synthetic data are a logged
  warning, §3, and the machinery assertions do not depend on match counts). The session `data_dir`
  fixture passes `external_predicates_file=<repo>/predicates.yaml` via a new `extra_args` passthrough
  on `build_workspace` (the only harness change).
* **Reader unit tests**: each grammar form against literal YAML snippets inline in the test (grammar is
  the template's, not the model's — literals are correct here); the skip cascade and manifest
  recording; the strict-mode errors; the always-error cases (colliding names, zero remaining
  predicates, missing file with `featurization=predicates`).
* **Predicates-file validation test**: parse the repo's `predicates.yaml` and fail if **any predicate
  would be skipped**, listing them. This is where strictness lives under the skip-by-default policy
  (§4): unsupported forms in the model's own declaration are a bug in the file even though runtime
  tolerates them in foreign files. Cheap, always-run, and the first thing to break — directly — when
  an author's edit introduces an unsupported form, rather than indirectly through a warning nobody
  reads.
* **Featurizer tests**: 0/1 values against hand-computed rows; original columns untouched; shard layout
  and `metadata/` handling per §3 (including the `subject_splits.parquet` **absence**); `features.json`
  and manifest fields.
* **`test_smoke_pipeline`, test by test**:
  * `test_workspace_is_published_with_manifests` — swap `tokenized_cohort` for `cohort_subjects`;
    assertions otherwise verbatim on both backends.
  * `test_inference_never_sees_ground_truth` — split: the `materialize_labels` half (does
    `include_labels=False` omit `boolean_value` from the written parquet) is backend-agnostic and
    load-bearing — `predict` enforces the property at the command level; it runs everywhere. The
    `MEDSPytorchDataset` half is an MTD-behavior check; `importorskip`. The equivalent guarantee for a
    custom datamodule becomes a protocol obligation plus a `skip_if_stub`-style test that activates
    when `datamodule.py` is implemented.
  * `test_conflicting_sources_are_rejected` — backend-agnostic, unchanged.
  * `test_end_to_end` — **unchanged**. It skips on the stub in a fresh repo on either backend (the
    template's documented "nothing trains" gap is backend-independent). Every argument `run_chain`
    passes is backend-neutral, so when a model and datamodule exist the same chain runs without MTD.
* **The final end-to-end using features** (what `test_end_to_end` executes once a model exists):
  1. `preprocess_data external_meds_dir=<synthetic MEDS> featurization=predicates
     external_predicates_file=<repo>/predicates.yaml output_data_dir=ws` →
     `ws/patients` with `predicate//*` columns and `representation: predicates`;
  2. `supervised_train input_data_dir=ws external_labels_dir=<fixture task labels> ...` — the custom
     datamodule reads featurized shards + materialized `{split}.parquet`, `vocab_size` = count from
     `features.json`, 2 CPU epochs;
  3. `predict input_supervised_model_dir=... splits=[held_out] ...` — keys via `schema_df`, coverage
     `n_written == n_expected`;
  4. assertions verbatim from today: `PredictionSchema.align`, manifests, coverage recorded.

  The stronger variant is `test_property` (designed-signal learnability + negative control); its
  featurized form has one constraint worth stating: the planted signal must be **predicate-visible** —
  the model only sees the feature columns its datamodule selects, so the label-carrying code must be
  matched by a declared predicate, or the model is structurally blind to the label and the learnability
  assertion fails with nothing actually broken. Consistent with the one-file principle, the featurized
  variant does **not** ship a signal-specific predicates fixture. Instead the dependency inverts:
  `build_signal_dataset` gains a from-predicates mode that generates the synthetic dataset **out of the
  model's own `predicates.yaml`** — it selects the predicates with literal codes (exact and any-of; a
  regex cannot be reverse-instantiated into a code), designates one as the signal predicate and the
  rest as distractors, emits their codes with the signal's presence carrying the label and the
  distractors label-independent, and keeps the existing anti-leak construction (random insertion
  position, label-independent sequence length, null `numeric_value`). The feature space the model is
  tested on is therefore its *production* feature space — same names, same order, same
  `features.json`. Distractors matter for the same reason as before: with a single feature the column
  trivially equals the answer. Preconditions, stated rather than hidden: the file must contain **at
  least two literal-code predicates** (one signal + one distractor); otherwise the test skips with a
  message telling the author to add one or provide example codes (a per-predicate example-code sidecar
  for regex predicates is possible future work, not v1). The rendered starter content satisfies the
  precondition out of the box. The negative control is unaffected (shuffled labels defeat every
  feature equally). Where it first runs: TECO is the real-world proof, but the featurized backend is also the
  cheapest possible **reference implementation** for the template's agreed `examples/` follow-up
  (predicate counts → linear head; no MTD, no heavy datamodule) — landing that would make this e2e run
  in template CI itself, closing the "nothing trains" gap and proving the custom path in one move.
* **Equivalence guard** (MTD repos only, both workspaces built in-session): run the same labels through
  `materialize_labels` against the MTD workspace and the predicates workspace; assert **identical split
  partitions**. This is the test that stops the two cohort readers from drifting — the failure mode
  that would otherwise surface as `CoverageError` two training runs late.
* **`test_cli_smoke` as the import-guard test, for free**: in a custom-rendered repo,
  `meds_torchdata` is genuinely not installed, so `meds-model commands` / `--help` succeeding *is* the
  regression test for §6.2. No mocking; the environment is the test.

### Tier 3 — CI (`rendered-smoke`)

Structural tier already covers all 14 combos; executing them all (~1 min/profile) is waste. Execution
matrix: the existing full-profile sweep on `mtd`, plus **`supervised` on `custom_featurization`**
(TECO's profile). The custom lane's venv deliberately omits meds-torch-data — the lane itself proves
nothing imports it. Update the documented rendered-suite expectations ("~37 passed, 2 skipped") so the
new skips stay legible under `-rs`.

---

## 9. MEDS-DEV integration (via MEDS-DEV PR #325)

[MEDS-DEV#325](https://github.com/Medical-Event-Data-Standard/MEDS-DEV/pull/325) adds the missing
plumbing: an optional `predicates_path` argument to `meds-dev-model`, exposed to model commands as the
`{predicates_path}` template variable. Properties that shape the integration:

* **Pure explicit passthrough.** The caller of `meds-dev-model` supplies the path; there is no
  automatic fallback to the registered dataset's `predicates.yaml` (unlike `meds-dev-task`). Auto-
  binding is a possible MEDS-DEV follow-up, noted in the PR.
* **Backwards compatible**: models that do not reference the variable are unaffected. A model that
  references it without the argument fails with a `KeyError` at command-formatting time — blunt but
  early.

### Template wiring (gated on #325 merging)

* **`model.yaml`** under `custom_featurization`: the first command of each chain renders with
  `external_predicates_file={predicates_path}` appended to `preprocess_data`. This makes
  `predicates_path` *required* for such a model in MEDS-DEV — the honest contract, since
  `featurization=predicates` without a file is an error by design (§3).
* **`meds-model-add-to-meds-dev`**: when registering a `custom_featurization` model, copy the repo's
  `predicates.yaml` into the model's MEDS-DEV directory alongside `model.yaml` and `requirements.txt`,
  so benchmark users have a reference file to pass:
  `meds-dev-model ... predicates_path=<meds-dev>/models/<name>/predicates.yaml`.
* **`test_meds_dev_e2e`**: the rendered e2e adds `predicates_path=<repo>/predicates.yaml` to its
  `meds-dev-model mode=full dataset_type=full` invocation — the one-file principle extended to the
  outermost test. Expectation to state plainly: over the MEDS-DEV demo dataset's vocabulary the
  model's predicates may match nothing, so features can be all-zero (a §3 warning) and predictions
  near-constant — the e2e asserts *plumbing* (venv build, placeholder fill, chain execution,
  `PredictionSchema`, coverage), not learnability, and all of that holds on all-zero features.
  Until #325 merges, this lane runs only against a checkout of the PR branch (`MEDS_DEV_DIR`
  override); it must not be wired to clone `main`.

### What predicates file does a MEDS-DEV caller pass?

The contract: a file whose parseable predicates define the model's feature space *for that dataset*.
The model's shipped `predicates.yaml` is the reference (correct for the dataset it was curated
against); a caller running another dataset supplies a binding file for it. Passing a raw MEDS-DEV
dataset `predicates.yaml` **works by default**: its value-bounded task predicates (e.g. MIMIC-IV's
`abnormally_high_*`) are skipped under §4's policy — warned, cascaded, recorded in the manifest —
while the presence signal flows through the plain sibling predicates those files already define.
Callers who want a hard failure on any unsupported entry pass `featurization_strict=true`. Note the consequence of a trained model's feature space being
`features.json` (§3): predict-time featurization must use the same bindings the model was trained
with, and the digest in the manifest is what makes a mismatch detectable.

---

## 10. Future work (recorded, not scoped)

* **Value channels**: per-predicate value extraction/transformation (units, parsing, clipping) —
  dftly is the candidate expression language (row-wise, YAML-native, same author, no temporal
  semantics; the temporal grammar would be ours). Value-bounded predicates (dropped from v1, §4)
  return here, where thresholds belong.
* **Reference implementation in `examples/`** (predicate counts → linear head over the
  `custom_featurization` backend): closes the template's "nothing trains" gap and gives the features
  e2e a home in template CI (§8).
* **Temporal spec**: grid/window/aggregation/imputation declarations — at that point this stops being
  a template feature and becomes an ecosystem spec ("what ACES is to cohorts"), with MEDS-Tab's
  windowed-aggregation machinery and TECO's `IntervalGrid` as the two existing consumers to unify.
* **MEDS-DEV auto-binding**: `predicates_path` defaulting to the registered dataset's
  `predicates.yaml` when `dataset_name` is set (the follow-up MEDS-DEV#325 explicitly leaves open),
  plus per-dataset featurization compatibility metadata (which concepts a dataset can bind, validated
  at registry time — requires a `Metadata` schema change, which currently accepts exactly
  `description`/`contacts`/`links` and raises on anything else at import).
* **Benchmark fairness**: concept-mapped vs vocab-generic models are different lanes; MEDS-DEV results
  should eventually record which lane a result came from.

---

## 11. Implementation order

1. **Seam refactor** — `cohort_subjects` dispatch (+ `representation: mtd` written by the existing
   branch), import guard, error-string softening, protocol module. Template stays green; MTD behavior
   byte-identical.
2. **Featurizer** — reader + shard rewriter + `features.json` + manifest fields + `_validate_featurized`;
   unit and featurizer tests (these run in MTD repos too).
3. **`data_backend` copier option** — copier.yml question, dependency/config/stub gating, per-backend
   conftest rendering, tier-1 test parametrization.
4. **CI lane** — `supervised` × `custom_featurization` rendered-smoke job; docs expectation updates.
5. **MEDS-DEV wiring** (gated on MEDS-DEV#325 merging) — `model.yaml` renders
   `external_predicates_file={predicates_path}`, `meds-model-add-to-meds-dev` ships `predicates.yaml`,
   `test_meds_dev_e2e` passes `predicates_path` (§9).
6. **TECO adoption** — datamodule implementing the protocol over predicate columns; the first
   end-to-end run of the custom path, and the point where `IntervalGrid`'s feature axis becomes the
   predicate set instead of the code vocabulary.
