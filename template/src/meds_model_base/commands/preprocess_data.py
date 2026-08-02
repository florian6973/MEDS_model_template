"""``preprocess_data`` — external MEDS → this model's patient representation.

Optionally runs a MEDS-transforms pipeline (``cfg.pipeline``, for model-specific enrichment such as
time-derived tokens or value binning), then meds-torch-data's ``MTD_preprocess`` to normalize, tokenize and
tensorize the dataset into the layout the training datamodule consumes.

Both stages are shelled out (they are Hydra applications with their own console scripts) and streamed live,
so long runs show progress. The result is published atomically as ``<output_data_dir>/patients``: either the
directory exists and is complete, or it does not exist at all.

``do_reshard=True`` is required when the input is not already sharded by split, which is the common case;
without it MTD raises "No schema files found".
"""

from __future__ import annotations

import logging
import os
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

from ..manifest import ArtifactType, input_ref, write_artifact
from ..schemas import code_metadata_filepath, subject_splits_filepath
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
        do_reshard = bool(cfg.get("do_reshard", True))
        pipeline = cfg.get("pipeline")

        with write_artifact(
            patients_dir,
            artifact_type=ArtifactType.data,
            command=self.name.value,
            name=PATIENTS_SUBDIR,
            inputs=[input_ref("external_meds_dir", external_meds_dir)],
            config=cfg,
            do_overwrite=bool(cfg.get("do_overwrite", False)),
        ) as (staging, extras):
            mtd_input = external_meds_dir
            if pipeline:
                intermediate = staging.parent / f"{staging.name}.transforms"
                self._run_meds_transforms(pipeline, external_meds_dir, intermediate, cfg)
                mtd_input = intermediate

            self._run_mtd(mtd_input, staging, do_reshard=do_reshard)
            _validate_tensorized(staging)

            extras["source"] = {"external_meds_dir": str(external_meds_dir)}
            extras["tensorization"] = {
                "do_reshard": do_reshard,
                "pipeline": str(pipeline) if pipeline else None,
            }
            extras.update(_describe_cohort(external_meds_dir, staging))

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
    def _run_mtd(input_dir: Path, output_dir: Path, *, do_reshard: bool) -> None:
        """Run meds-torch-data tensorization (``MTD_preprocess``) into the staging directory."""
        run_streamed(
            [
                "MTD_preprocess",
                f"MEDS_dataset_dir={input_dir}",
                f"output_dir={output_dir}",
                f"do_reshard={do_reshard}",
                "do_overwrite=true",
            ],
            stage="MTD_preprocess",
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
            f"No tokenization schema files under {output_dir}/tokenization/schemas. If the input was not "
            "split-sharded, re-run preprocess_data with do_reshard=True."
        )


def _describe_cohort(meds_dir: Path, tensorized: Path) -> dict:
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
