"""The ``preprocess`` step (a): raw MEDS → a model-ready (tensorized) cohort.

``DefaultPreprocessStep`` optionally runs a MEDS-transforms pipeline (``cfg.pipeline``, for model-specific
enrichment like time-derived tokens or value binning) and then meds-torch-data's ``MTD_preprocess`` to
normalize, tokenize and tensorize the dataset into the layout the training datamodule consumes.

Both stages are shelled out (they are Hydra apps with their own console scripts) and streamed live so
long real-dataset runs show progress. ``do_reshard=True`` is required when the input is not already
sharded by split (the common case) — without it MTD raises "No schema files found".
"""

from __future__ import annotations

import logging
import os
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

from ..schemas import code_metadata_filepath
from .base import PreprocessStep

if TYPE_CHECKING:  # pragma: no cover - typing only
    from omegaconf import DictConfig

logger = logging.getLogger(__name__)


def _run_streamed(cmd: list[str], *, env: dict[str, str] | None = None, stage: str) -> None:
    """Run ``cmd`` streaming stdout/stderr live; raise on non-zero exit."""
    logger.info("[%s] running: %s", stage, " ".join(cmd))
    result = subprocess.run(cmd, env=env, check=False)  # noqa: S603 - trusted CLI args
    if result.returncode != 0:
        raise RuntimeError(f"{stage} failed with exit code {result.returncode}. See output above.")


class DefaultPreprocessStep(PreprocessStep):
    """Default preprocessing: (optional) MEDS-transforms pipeline → MTD tensorization."""

    def run(self, cfg: "DictConfig") -> Path:
        input_dir = Path(cfg.input_dir)
        output_dir = Path(cfg.output_dir)
        do_reshard = bool(cfg.get("do_reshard", True))
        do_overwrite = bool(cfg.get("do_overwrite", False))
        pipeline = cfg.get("pipeline")

        mtd_input = input_dir
        if pipeline:
            intermediate_dir = Path(cfg.get("intermediate_dir") or (output_dir.parent / "_intermediate"))
            self._run_meds_transforms(pipeline, input_dir, intermediate_dir, cfg)
            mtd_input = intermediate_dir

        self._run_mtd(mtd_input, output_dir, do_reshard=do_reshard, do_overwrite=do_overwrite)
        _validate_tensorized(output_dir)
        logger.info("Preprocessing complete; tensorized cohort at %s", output_dir)
        return output_dir

    @staticmethod
    def _run_meds_transforms(pipeline: str, input_dir: Path, output_dir: Path, cfg: "DictConfig") -> None:
        """Run a MEDS-transforms pipeline YAML, overriding its input/output dirs."""
        extra = list(cfg.get("pipeline_overrides", []) or [])
        _run_streamed(
            [
                "MEDS_transform-pipeline",
                str(pipeline),
                f"input_dir={input_dir}",
                f"output_dir={output_dir}",
                *extra,
            ],
            env=os.environ.copy(),
            stage="MEDS_transform-pipeline",
        )

    @staticmethod
    def _run_mtd(input_dir: Path, output_dir: Path, *, do_reshard: bool, do_overwrite: bool) -> None:
        """Run meds-torch-data tensorization (``MTD_preprocess``)."""
        _run_streamed(
            [
                "MTD_preprocess",
                f"MEDS_dataset_dir={input_dir}",
                f"output_dir={output_dir}",
                f"do_reshard={do_reshard}",
                f"do_overwrite={do_overwrite}",
            ],
            stage="MTD_preprocess",
        )


def _validate_tensorized(output_dir: Path) -> None:
    """Sanity-check the tensorized output (the invariants a bare schema ``validate()`` misses)."""
    codes_fp = output_dir / code_metadata_filepath
    if not codes_fp.is_file():
        raise FileNotFoundError(
            f"Expected tensorized code metadata at {codes_fp}; preprocessing did not complete."
        )
    schemas = list((output_dir / "tokenization" / "schemas").glob("*/*.parquet"))
    if not schemas:
        raise FileNotFoundError(
            f"No tokenization schema files under {output_dir}/tokenization/schemas. If the input was not "
            "split-sharded, re-run preprocess with do_reshard=True."
        )
