# Minimal MEDS model template

This repository is a self-contained [Copier](https://copier.readthedocs.io) template for minimal,
contract-first MEDS model repositories.

The template fixes the external contract—MEDS input, six command DAGs and standard arguments, typed
artifact manifests, final MEDS predictions, measurements, MEDS-DEV, SLURM, and GitHub result exchange—while
leaving preprocessing, featurization, intermediate formats, frameworks, batching, and model execution to
the generated repository.

## Render a repository

From the root of this checkout:

```bash
uv tool install copier
copier copy . ../my-meds-model
```

For a reproducible non-interactive render:

```bash
copier copy --defaults --trust \
  --data model_name="My MEDS Model" \
  --data model_slug=my_meds_model \
  --data profile=probe \
  --data uses_predicates=true \
  --data implementation_source=https://github.com/example/source-model \
  . ../my-meds-model
```

`--trust` is safe but currently unnecessary because the template defines no post-copy tasks. It is included
so the command remains usable if formatting/bootstrap tasks are added later.

### Copier questions

| Question | Meaning |
|---|---|
| `model_name` | Human-readable model name. |
| `model_slug` | Importable Python package and MEDS-DEV model identifier. |
| `profile` | One of the six supported command DAGs. |
| `model_description` | Short description rendered into README and `model.yaml`. |
| `author_name`, `author_email` | MEDS-DEV contact metadata. |
| `uses_predicates` | Whether generated MEDS-DEV commands pass `external_predicates_file`. The model still owns interpretation. |
| `implementation_source` | Paper path/URL, source repository path/URL, or both; prefilled into generated guidance and the report. |

Profiles are `supervised`, `finetune`, `probe`, `zero_shot_direct`, `zero_shot_materialized`, and
`packaged`. They select command topology, not an architecture or execution backend.

## Work in the generated repository

```bash
cd ../my-meds-model
uv sync --extra test
uv run meds-model commands
uv run pytest -rs
```

The generated commands are intentional stubs. Before implementation, the local, MEDS-DEV, and MIMIC
end-to-end tests report explicit `model_stub` skips. Start with these rendered files:

1. `AGENTS.md`
2. `SPEC.md`
3. `README.md`
4. `docs/IMPLEMENTATION_GUIDE.md`
5. `src/<model_slug>/commands.py`
6. `tests/e2e.py`
7. `IMPLEMENTATION_REPORT.md`

A completed implementation sets `IS_STUB = False` only after its actual DAG works.

## Copyable model-implementation prompt

Use this after rendering a repository. Replace both placeholders before sending it to a coding agent.

```text
Based on the following source:

<PAPER_PATH_OR_URL_AND_OR_REPOSITORY_PATH_OR_URL>

implement the model in the generated MEDS model repository at:

<GENERATED_REPOSITORY_PATH>

Read AGENTS.md, SPEC.md, README.md, and docs/IMPLEMENTATION_GUIDE.md in the generated repository
completely before writing code. Inspect the paper, supplementary material, source repository,
configuration files, checkpoints, preprocessing code, and evaluation code available from the supplied
source.

Preserve the selected command DAG and standardized command arguments, MEDS input boundary, artifact
manifest envelopes, MEDS-DEV model.yaml interface, and final MEDS prediction output. Intermediate payload
formats and execution mechanisms are model-owned: choose whatever preprocessing, featurization, framework,
batching or non-batching strategy, external executable, storage format, and model architecture most
faithfully reproduce the source.

Reuse MEDS-DEV dataset predicates when needed, but decide within the model how those predicates become
inputs. Do not introduce a generic predicate-derived feature representation into the contract package.
Do not omit source behavior because it is inconvenient for a fixture; adapt the implementation or fixture
instead.

Record every material source element as ported, adapted, or omitted, with evidence and justification, in
IMPLEMENTATION_REPORT.md. Implement every registered command, finalize the implementation-specific
command strings in model.yaml, and implement tests/e2e.py. Keep IS_STUB = True until the actual selected
DAG completes and produces standards-conformant predictions.

Then run:

    uv run pytest -rs
    uv run pytest -m meds_dev -rs
    uv run pytest -m real_data -rs

There must be zero model_stub skips. Report exact pass/skip counts and account for every remaining skip.
Complete IMPLEMENTATION_REPORT.md with exact source revisions, commands executed, final-output coverage,
predicate provenance, environment, wall time, peak process-tree PSS/RSS, GPU measurements when available,
scientific deviations, and known limitations.
```

## Test tiers in a generated repository

```bash
uv run pytest -rs
uv run pytest -m meds_dev -rs
uv run pytest -m real_data -rs
```

- The default suite checks the contract and model-owned local end-to-end adapter.
- `meds_dev` registers `model.yaml` and runs it in MEDS-DEV's isolated environment.
- `real_data` builds or reuses the MEDS MIMIC-IV demo, extracts MEDS-DEV task
  `mortality/in_icu/first_24h`, reuses MEDS-DEV's MIMIC predicates, and runs the full selected DAG.

Set `MEDS_DEV_DIR=/path/to/MEDS-DEV` to reuse a checkout and `MEDS_DEMO_DIR=/path/to/demo` to reuse the
dataset. A bare test run never clones MEDS-DEV or downloads MIMIC data.

## Cluster and result interfaces

```bash
$EDITOR slurm/config.sh
./slurm/submit.sh --dry-run
./slurm/submit.sh
scripts/github-sync.sh <run-id>
```

Each command manifest records wall time and sampled process-tree PSS/RSS. Structured run evidence is kept
under `runs/<run-id>/`. GitHub synchronization publishes a run branch; it does not commit datasets,
checkpoints, credentials, or raw logs.

## Develop the Copier template

From the parent repository root:

```bash
uv sync --group dev
uv run pytest tests/test_render.py -q
uv run pytest tests/ -q
uv run ruff check tests/test_render.py
uv run ruff format --check tests/test_render.py
```

`tests/test_render.py` renders all six DAGs, compiles the generated repositories, validates command
registries and operational files, tests measurement manifests and run results, and exercises MEDS-DEV
registration. To test the rendered repository itself, render it into a temporary directory and run:

```bash
uv run --project /tmp/my-render --extra test pytest /tmp/my-render/tests -q -rs
```

The authority is [`SPEC.md`](SPEC.md). Files under `template/` are the generated payload; root tests
validate that payload without imposing a model implementation.
