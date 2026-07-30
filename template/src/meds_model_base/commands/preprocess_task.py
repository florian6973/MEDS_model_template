"""``preprocess_task`` — materialize or validate a task against already-preprocessed patient data.

Appends ``<data_dir>/<output_task_subdir>/{train,tuning,held_out}.parquet`` and never touches
``patients/``. The subdirectory is published atomically, so concurrent jobs materializing different tasks
against the same ``data_dir`` cannot interfere: each writes its own directory once, and there is no shared
index file to update.

The ACES path needs the *raw* MEDS dataset, not the tensorized cohort (tensorization replaces codes with
vocabulary indices, so the predicates can no longer be evaluated). Rather than make the caller repeat an
argument they already gave ``preprocess_data``, the raw location is recovered from the ``patients/``
manifest; ``cfg.external_meds_dir`` overrides it when the dataset has since moved.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from ..manifest import ArtifactType, input_ref, read_manifest, write_artifact
from ..tasks import (
    TaskMaterializationError,
    extract_with_aces,
    is_aces_task_file,
    load_subject_splits,
    read_materialized_labels,
    split_labels,
    summarize_labels,
    write_task_splits,
)
from ..utils import resolve_subdir
from .base import PreprocessTaskCommand
from .preprocess_data import PATIENTS_SUBDIR

if TYPE_CHECKING:  # pragma: no cover - typing only
    from omegaconf import DictConfig

logger = logging.getLogger(__name__)


class DefaultPreprocessTaskCommand(PreprocessTaskCommand):
    """Default task materialization: ACES extraction or validation of pre-extracted labels."""

    def run(self, cfg: DictConfig) -> Path:
        data_dir = Path(cfg.input_data_dir)
        patients_dir = data_dir / PATIENTS_SUBDIR
        task_file = Path(cfg.external_task_file)
        dest = resolve_subdir(data_dir, cfg.output_task_subdir)

        # Validate the workspace before doing any work: a data_dir without a patients manifest was not
        # produced by preprocess_data, and extraction against it would be meaningless.
        patients_manifest = read_manifest(patients_dir, require_type=ArtifactType.data)

        if is_aces_task_file(task_file):
            meds_dir = self._resolve_meds_dir(cfg, patients_manifest)
            logger.info("Extracting ACES task %s against %s.", task_file, meds_dir)
            labels = extract_with_aces(task_file, meds_dir, cfg.get("predicates_file"))
            by_split = split_labels(labels, load_subject_splits(meds_dir))
            materialization = "aces_extracted"
        else:
            logger.info("Reading pre-materialized labels from %s.", task_file)
            loaded = read_materialized_labels(task_file)
            if isinstance(loaded, dict):
                by_split = loaded
            else:
                meds_dir = self._resolve_meds_dir(cfg, patients_manifest)
                by_split = split_labels(loaded, load_subject_splits(meds_dir))
            materialization = "passed_through"

        with write_artifact(
            dest,
            artifact_type=ArtifactType.task,
            command=self.name.value,
            name=Path(cfg.output_task_subdir).name,
            inputs=[
                input_ref("input_data_dir", patients_dir),
                input_ref("external_task_file", task_file),
            ],
            config=cfg,
            do_overwrite=bool(cfg.get("do_overwrite", False)),
        ) as (staging, extras):
            counts = write_task_splits(by_split, staging)
            extras["materialization"] = materialization
            extras["source"] = {"external_task_file": str(task_file)}
            extras["labels"] = summarize_labels(by_split)
            logger.info("Materialized %s label rows across %d splits.", sum(counts.values()), len(counts))

        return dest

    @staticmethod
    def _resolve_meds_dir(cfg: DictConfig, patients_manifest: dict) -> Path:
        """Locate the raw MEDS dataset: an explicit override, else the one ``preprocess_data`` recorded."""
        if override := cfg.get("external_meds_dir"):
            return Path(override)
        recorded = (patients_manifest.get("source") or {}).get("external_meds_dir")
        if not recorded:
            raise TaskMaterializationError(
                "This task needs the raw MEDS dataset, but the patients manifest does not record one. "
                "Pass external_meds_dir=... explicitly."
            )
        meds_dir = Path(recorded)
        if not meds_dir.exists():
            raise TaskMaterializationError(
                f"The raw MEDS dataset recorded in the patients manifest ({meds_dir}) no longer exists. "
                "Pass external_meds_dir=... to point at its current location, or supply pre-materialized "
                "labels as external_task_file."
            )
        return meds_dir
