"""``preprocess_data`` — external MEDS → this model's patient representation.

Optionally runs a MEDS-transforms pipeline (``cfg.pipeline``, for model-specific enrichment such as
time-derived tokens or value binning), then one of two representations (``cfg.featurization``, recorded
in the manifest as ``representation``):

- ``mtd`` (default): meds-torch-data's ``MTD_preprocess`` normalizes, tokenizes and tensorizes the
  dataset into the layout the MTD training datamodule consumes.
- ``predicates``: presence featurization (:mod:`meds_model_base.featurize`) — the data keeps its MEDS
  layout, augmented with one 0/1 ``predicate//<name>`` column per predicate in
  ``cfg.external_predicates_file``, plus a ``features.json`` defining the feature space. Needs no
  meds-torch-data; the model brings its own datamodule (see ``lightning/protocol.py``).

Both stages are shelled out (they are Hydra applications with their own console scripts) and streamed live,
so long runs show progress. The result is published atomically as ``<output_data_dir>/patients``: either the
directory exists and is complete, or it does not exist at all.

**The input must already be sharded by split** (``data/train/…``, ``data/tuning/…``, ``data/held_out/…``).
That is what ``meds-dev-dataset`` and the standard MEDS ETL produce, and it is the only layout from which
meds-torch-data can recover split membership: it reads the shard path and never opens
``subject_splits.parquet``. Input sharded another way is refused up front, with the resharding command to
run, rather than failing several minutes in.
"""

from __future__ import annotations

import logging
import os
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

from ..manifest import ArtifactType, input_ref, write_artifact
from ..schemas import SPLITS, code_metadata_filepath, subject_splits_filepath
from .base import PreprocessDataCommand

if TYPE_CHECKING:  # pragma: no cover - typing only
    from omegaconf import DictConfig

logger = logging.getLogger(__name__)

PATIENTS_SUBDIR = "patients"


def run_streamed(cmd: list[str], *, env: dict[str, str] | None = None, stage: str) -> None:
    """Run ``cmd`` streaming stdout/stderr live; raise on non-zero exit."""
    logger.info("[%s] running: %s", stage, " ".join(cmd))
    result = subprocess.run(cmd, env=env, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"{stage} failed with exit code {result.returncode}. See output above.")


class DefaultPreprocessDataCommand(PreprocessDataCommand):
    """Default preprocessing: (optional) MEDS-transforms pipeline → MTD tensorization."""

    def run(self, cfg: DictConfig) -> Path:
        external_meds_dir = Path(cfg.external_meds_dir)
        data_dir = Path(cfg.output_data_dir)
        patients_dir = data_dir / PATIENTS_SUBDIR
        pipeline = cfg.get("pipeline")
        featurization = str(cfg.get("featurization") or "mtd")
        if featurization not in ("mtd", "predicates"):
            raise ValueError(f"featurization must be 'mtd' or 'predicates', got {featurization!r}.")

        parsed = predicates_file = digest = None
        if featurization == "predicates":
            from ..featurize import predicates_digest, read_predicates_file

            predicates_file = cfg.get("external_predicates_file")
            if not predicates_file:
                # An error, not a passthrough: silently publishing an artifact with zero feature
                # columns would look exactly like a finished one.
                raise ValueError(
                    "featurization=predicates requires external_predicates_file. Pass the model's "
                    "predicates file (e.g. external_predicates_file=predicates.yaml)."
                )
            # Parsed before any heavy work: a bad predicates file fails in seconds, not after a pipeline.
            parsed = read_predicates_file(
                predicates_file, strict=bool(cfg.get("featurization_strict", False))
            )
            digest = predicates_digest(predicates_file)

        # Before the artifact is staged and long before a pipeline runs: this is the one precondition
        # whose violation is otherwise reported by MTD, several minutes in and in its own vocabulary.
        _require_split_sharded(external_meds_dir, "external_meds_dir")

        with write_artifact(
            patients_dir,
            artifact_type=ArtifactType.data,
            command=self.name.value,
            name=PATIENTS_SUBDIR,
            inputs=[input_ref("external_meds_dir", external_meds_dir)],
            config=cfg,
            do_overwrite=bool(cfg.get("do_overwrite", False)),
        ) as (staging, extras):
            source_input = external_meds_dir
            if pipeline:
                intermediate = staging.parent / f"{staging.name}.transforms"
                self._run_meds_transforms(pipeline, external_meds_dir, intermediate, cfg)
                # MEDS-transforms preserves shard layout, so this holds for any pipeline that does not
                # deliberately reshard. Checking anyway costs a directory listing and turns "MTD found no
                # schema files" back into a statement about the pipeline that actually caused it.
                _require_split_sharded(intermediate, "the pipeline output")
                source_input = intermediate

            extras["source"] = {"external_meds_dir": str(external_meds_dir)}
            if featurization == "predicates":
                from ..featurize import featurize_dataset
                from ..tasks import featurized_cohort

                counts = featurize_dataset(source_input, staging, parsed)
                _validate_featurized(staging, parsed)
                extras["representation"] = "predicates"
                extras["featurization"] = {
                    "pipeline": str(pipeline) if pipeline else None,
                    "predicates_file": str(predicates_file),
                    "predicates_digest": digest,
                    "n_features": len(parsed.order),
                    "skipped": sorted(parsed.skipped),
                    "match_counts": counts,
                }
                cohort = featurized_cohort(staging)
            else:
                self._run_mtd(source_input, staging)
                _validate_tensorized(staging)
                extras["representation"] = "mtd"
                extras["tensorization"] = {"pipeline": str(pipeline) if pipeline else None}
                cohort = None
            extras.update(_describe_cohort(external_meds_dir, staging, cohort=cohort))

        logger.info("Patient data ready at %s.", patients_dir)
        return patients_dir

    @staticmethod
    def _run_meds_transforms(pipeline: str, input_dir: Path, output_dir: Path, cfg: DictConfig) -> None:
        """Run a MEDS-transforms pipeline YAML, overriding its input/output dirs.

        ``MEDS_transform-pipeline`` is argparse, not Hydra: it takes the pipeline YAML positionally and
        every Hydra-style override after a single ``--overrides`` flag. Bare ``k=v`` positionals are an
        argparse error (exit 2, before any stage runs), so the flag is load-bearing — without it
        ``preprocess_data pipeline=...`` fails for every pipeline.
        """
        extra = list(cfg.get("pipeline_overrides", []) or [])
        run_streamed(
            [
                "MEDS_transform-pipeline",
                str(pipeline),
                "--overrides",
                f"input_dir={input_dir}",
                f"output_dir={output_dir}",
                *extra,
            ],
            env=os.environ.copy(),
            stage="MEDS_transform-pipeline",
        )

    @staticmethod
    def _run_mtd(input_dir: Path, output_dir: Path) -> None:
        """Run meds-torch-data tensorization (``MTD_preprocess``) into the staging directory.

        ``do_reshard=false`` always, passed explicitly rather than left to MTD's default so the invariant
        is legible at the call site. Resharding here was the source of an impossible combination:
        ``reshard_to_split`` reads ``metadata/subject_splits.parquet`` from *its own* input, and a
        MEDS-transforms pipeline does not carry that file through — so ``pipeline`` with resharding could
        only ever fail, and always after the pipeline had already run. Requiring split-sharded input
        removes the combination rather than guarding it.
        """
        run_streamed(
            [
                "MTD_preprocess",
                f"MEDS_dataset_dir={input_dir}",
                f"output_dir={output_dir}",
                "do_reshard=false",
                "do_overwrite=true",
            ],
            stage="MTD_preprocess",
        )


def _require_split_sharded(meds_dir: Path, what: str) -> None:
    """Refuse MEDS input that is not sharded by split.

    meds-torch-data recovers split membership from the shard path and from nothing else — its dataset
    keeps a shard only when the name starts with ``f"{split}/"``. Input sharded any other way tensorizes
    into an artifact with no splits at all, which surfaces as MTD's "No schema files found" long after the
    cause.

    This is a precondition rather than something to fix by resharding here, because resharding *here* is
    what could not be made to work: it needs ``metadata/subject_splits.parquet`` in its own input, and a
    MEDS-transforms pipeline drops that file. Requiring the layout up front is the only rule that holds
    with and without a ``pipeline``, and it is already what ``meds-dev-dataset`` and the standard MEDS ETL
    produce.
    """
    data_dir = meds_dir / "data"
    shards = [
        p
        for p in sorted(data_dir.rglob("*.parquet"))
        if not any(part.startswith(".") for part in p.relative_to(data_dir).parts)
    ]
    if not shards:
        raise FileNotFoundError(f"No MEDS data shards under {data_dir}; {what} is not a MEDS dataset.")

    stray = sorted({p.relative_to(data_dir).parts[0] for p in shards} - set(SPLITS))
    if stray:
        listed = ", ".join(stray[:5]) + (" …" if len(stray) > 5 else "")
        raise ValueError(
            f"{what} at {meds_dir} is not sharded by split: data/ contains {listed}, but every shard "
            f"must sit under one of {'/'.join(SPLITS)}/. meds-torch-data reads split membership from the "
            "shard path alone, so this input cannot be tensorized into a usable artifact.\n\n"
            "Reshard it first with a one-stage MEDS-transforms pipeline:\n\n"
            "    # reshard.yaml\n"
            f"    input_dir: {meds_dir}\n"
            "    output_dir: /path/to/split_sharded\n"
            "    stages:\n"
            "      - reshard_to_split\n\n"
            "    MEDS_transform-pipeline reshard.yaml\n\n"
            "then point external_meds_dir at the output. A pipeline= that already starts with "
            "reshard_to_split does the same job in one pass."
        )


def _validate_tensorized(output_dir: Path) -> None:
    """Check the invariants a bare schema ``validate()`` misses, before the artifact is published."""
    codes_fp = output_dir / code_metadata_filepath
    if not codes_fp.is_file():
        raise FileNotFoundError(
            f"Expected tensorized code metadata at {codes_fp}; preprocessing did not complete."
        )
    schemas = list((output_dir / "tokenization" / "schemas").glob("*/*.parquet"))
    if not schemas:
        raise FileNotFoundError(
            f"No tokenization schema files under {output_dir}/tokenization/schemas. The input passed "
            "the split-sharding check, so tensorization itself produced nothing — check "
            "MTD_preprocess's output above."
        )


def _validate_featurized(output_dir: Path, parsed) -> None:
    """The featurized twin of :func:`_validate_tensorized`: re-verify the artifact before publishing.

    Every declared predicate column must be present in every shard, and ``features.json`` must agree
    with what was parsed — cheap re-reads of what :func:`~meds_model_base.featurize.featurize_dataset`
    just wrote, standing between a partial write and a published artifact.
    """
    import json

    import polars as pl

    from ..featurize import FEATURES_FILENAME

    features_fp = output_dir / FEATURES_FILENAME
    if not features_fp.is_file():
        raise FileNotFoundError(f"Expected {features_fp}; featurization did not complete.")
    declared = [f["column"] for f in json.loads(features_fp.read_text())["features"]]
    if declared != parsed.columns:
        raise RuntimeError(
            f"{features_fp} disagrees with the parsed predicates: {declared} != {parsed.columns}."
        )
    shards = sorted((output_dir / "data").rglob("*.parquet"))
    if not shards:
        raise FileNotFoundError(f"No data shards under {output_dir}/data; featurization wrote nothing.")
    for fp in shards:
        missing = set(declared) - set(pl.scan_parquet(fp).collect_schema().names())
        if missing:
            raise RuntimeError(f"Shard {fp} is missing predicate column(s): {sorted(missing)}.")


def _describe_cohort(meds_dir: Path, tensorized: Path, cohort: dict | None = None) -> dict:
    """Cohort statistics for the manifest, counted from the tensorized output.

    Counting the *artifact* rather than ``external_meds_dir`` is what makes a filtering ``pipeline``
    visible: these numbers now describe what the artifact holds, not what preprocessing was handed.

    ``dropped_by_pipeline`` is the difference, recorded here because here is the only place both sides
    exist at once. Nothing downstream needs it — labels are partitioned from the tokenized cohort, so a
    subject the pipeline removed is dropped by construction. It is recorded so that a user who later
    wonders why their label count fell can find the answer in the artifact that caused it, rather than
    inferring it from a warning several commands later.
    """
    import polars as pl

    from ..tasks import tokenized_cohort

    described: dict = {}
    if cohort is None:
        cohort = tokenized_cohort(tensorized)
    if cohort:
        described["splits"] = {split: len(subjects) for split, subjects in sorted(cohort.items())}

    splits_fp = meds_dir / subject_splits_filepath
    if splits_fp.is_file():
        n_source = pl.read_parquet(splits_fp).height
        n_tokenized = sum(len(s) for s in cohort.values())
        if n_tokenized < n_source:
            described["dropped_by_pipeline"] = n_source - n_tokenized
            logger.warning(
                "Preprocessing dropped %d of %d subjects (%d tokenized). Labels for them will be "
                "dropped downstream; this is expected when the pipeline filters subjects.",
                n_source - n_tokenized,
                n_source,
                n_tokenized,
            )

    codes_fp = tensorized / code_metadata_filepath
    if codes_fp.is_file():
        described["vocabulary_size"] = pl.read_parquet(codes_fp).height
    return described
