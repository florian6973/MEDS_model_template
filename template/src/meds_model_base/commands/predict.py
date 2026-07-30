"""``predict`` — task index (+ exactly one source) → ``predictions.parquet``.

The shared ``run`` is fixed for every implementation:

1. validate the workspace and the arbitrated source;
2. load the index of timepoints, **dropping any ground-truth labels**;
3. call the implementation's :meth:`predict` hook;
4. check coverage, validate against ``meds_evaluation.PredictionSchema``, and publish.

Steps 1, 2 and 4 are not overridable. They are what makes every model's output comparable, and they are
where the two failure modes that produce plausible-but-wrong results are caught:

- **Ground truth never enters.** The index carries only ``(subject_id, prediction_time)``; a task's
  ``boolean_value`` is discarded on load. Scoring is a separate tool.
- **Coverage is checked.** A model that scores only part of the index fails here rather than emitting a
  short file that looks like a complete run. ``n_expected`` and ``n_written`` land in the manifest per
  split so the invariant stays checkable afterwards.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

import polars as pl
import pyarrow.parquet as pq

from ..manifest import ArtifactType, InferenceKind, input_ref, read_manifest, write_artifact
from ..schemas import validate_predictions
from ..utils import resolve_subdir
from ._runtime import KEYS, load_index, load_trained_module, resolve_splits, run_predict_step, stack_outputs
from .base import PredictCommand
from .preprocess_data import PATIENTS_SUBDIR

if TYPE_CHECKING:  # pragma: no cover - typing only
    from omegaconf import DictConfig

logger = logging.getLogger(__name__)

PREDICTIONS_FILENAME = "predictions.parquet"

PROBABILITY_COLUMN = "predicted_boolean_probability"

#: Which artifact type each prediction source must point at.
_SOURCE_ARTIFACT_TYPES = {
    "input_supervised_model_dir": ArtifactType.supervised_model,
    "input_pretrained_model_dir": ArtifactType.pretrained_model,
    "input_inference_subdir": ArtifactType.inference,
}


class CoverageError(RuntimeError):
    """Raised when a model did not score every timepoint in the requested index."""


class _PredictRunMixin:
    """The fixed ``run`` contract shared by every prediction implementation."""

    def run(self, cfg: DictConfig) -> Path:
        data_dir = Path(cfg.input_data_dir)
        read_manifest(data_dir / PATIENTS_SUBDIR, require_type=ArtifactType.data)

        task_dir = resolve_subdir(data_dir, cfg.input_task_subdir)
        read_manifest(task_dir, require_type=ArtifactType.task)

        role, value = self.source
        source_path = self._validate_source(cfg, role, value)

        splits = resolve_splits(cfg)
        index = load_index(task_dir, splits)
        logger.info("Predicting %d timepoints across %d split(s).", len(index), index["split"].n_unique())

        predictions = self.predict(cfg, (role, value), index)
        coverage = _check_coverage(index, predictions)

        output_dir = Path(cfg.output_predictions_dir)
        with write_artifact(
            output_dir,
            artifact_type=ArtifactType.predictions,
            command=self.name.value,
            inputs=[
                input_ref("input_data_dir", data_dir / PATIENTS_SUBDIR),
                input_ref("input_task_subdir", task_dir),
                input_ref(role, source_path) if role else None,
            ],
            config=cfg,
            do_overwrite=bool(cfg.get("do_overwrite", False)),
        ) as (staging, extras):
            table = validate_predictions(predictions.drop("split", strict=False).to_arrow())
            pq.write_table(table, staging / PREDICTIONS_FILENAME)
            extras["source"] = {"role": role or "packaged_model", "path": str(source_path or "")}
            extras["coverage"] = coverage
            extras["splits"] = sorted(index["split"].unique().to_list())

        logger.info("Wrote %d predictions to %s.", len(predictions), output_dir / PREDICTIONS_FILENAME)
        return output_dir

    def _validate_source(self, cfg: DictConfig, role: str | None, value) -> Path | None:
        """Resolve the arbitrated source to a path and check its manifest declares the right artifact type."""
        if role is None:
            return None
        path = resolve_subdir(cfg.input_data_dir, value) if role.endswith("_subdir") else Path(value)
        require_kind = InferenceKind.embeddings if role == "input_inference_subdir" else None
        read_manifest(path, require_type=_SOURCE_ARTIFACT_TYPES[role], require_kind=require_kind)
        return path


def _check_coverage(index: pl.DataFrame, predictions: pl.DataFrame) -> dict:
    """Verify one prediction per index row, per split; return the manifest's ``coverage`` block."""
    expected = index.group_by("split").agg(pl.len().alias("n_expected"))
    if "split" in predictions.columns:
        written = predictions.group_by("split").agg(pl.len().alias("n_written"))
    else:
        joined = index.join(predictions.select(KEYS).unique(), on=KEYS, how="semi")
        written = joined.group_by("split").agg(pl.len().alias("n_written"))

    report = (
        expected.join(written, on="split", how="left")
        .with_columns(pl.col("n_written").fill_null(0))
        .sort("split")
    )
    short = report.filter(pl.col("n_written") < pl.col("n_expected"))
    if len(short):
        detail = "; ".join(
            f"{r['split']}: {r['n_written']}/{r['n_expected']}" for r in short.to_dicts()
        )
        raise CoverageError(
            f"The model scored fewer timepoints than the task defines ({detail}). Predictions must cover "
            "the whole index: a short file is indistinguishable from a complete one downstream. Either fix "
            "the model's coverage, or restrict the request with splits=[...]."
        )
    return {
        r["split"]: {"n_expected": r["n_expected"], "n_written": r["n_written"]}
        for r in report.to_dicts()
    }


class SupervisedPredictCommand(_PredictRunMixin, PredictCommand):
    """Score a task with a trained supervised model (the ordinary supervised and fine-tuned case)."""

    supported_sources: ClassVar[frozenset[str]] = frozenset({"input_supervised_model_dir"})

    def predict(self, cfg: DictConfig, source, index: pl.DataFrame) -> pl.DataFrame:
        _, value = source
        module = load_trained_module(Path(value))
        frames = []
        for split in index["split"].unique(maintain_order=True).to_list():
            keys, outputs = run_predict_step(cfg, module, split)
            if not len(keys):
                continue
            scored = keys.with_columns(stack_outputs(outputs))
            frames.append(scored.with_columns(pl.lit(split).alias("split")))
        if not frames:
            raise CoverageError("The datamodule produced no rows for any requested split.")
        return _select_prediction_columns(pl.concat(frames, how="vertical_relaxed"))


class ProbePredictCommand(_PredictRunMixin, PredictCommand):
    """Score a task with a probe trained on frozen embeddings.

    Takes only the probe (``input_supervised_model_dir``) — a probe is a supervised model, so this is the
    supervised case as far as the interface is concerned. **Which embeddings to score is read from the
    probe's own manifest**, since ``supervised_train`` recorded the inference artifacts it was trained on.
    Requiring the caller to re-supply them would invite the mismatch that a manifest exists to prevent.
    """

    supported_sources: ClassVar[frozenset[str]] = frozenset({"input_supervised_model_dir"})

    def predict(self, cfg: DictConfig, source, index: pl.DataFrame) -> pl.DataFrame:
        from ..lightning.probe import probe_predict_frame

        _, value = source
        probe_dir = Path(value)
        manifest = read_manifest(probe_dir, require_type=ArtifactType.supervised_model)
        initialization = manifest.get("initialization") or {}
        inference_path = initialization.get("path")
        if initialization.get("from") != "inference_artifacts" or not inference_path:
            raise ValueError(
                f"{probe_dir} was not trained on inference artifacts (initialization: "
                f"{initialization.get('from', 'unknown')!r}), so it cannot be scored as a probe. Use "
                "SupervisedPredictCommand for a model trained on the tokenized cohort."
            )
        read_manifest(
            inference_path,
            require_type=ArtifactType.inference,
            require_kind=InferenceKind.embeddings,
        )
        module = load_trained_module(probe_dir)
        return probe_predict_frame(Path(inference_path), index.select(KEYS), module)


class MaterializedPredictCommand(_PredictRunMixin, PredictCommand):
    """Score a task from inference artifacts that already contain probabilities.

    This is the materialized zero-shot chain: ``infer`` wrote native scores for each timepoint, and this
    command only has to align them to the index. The artifacts must declare kind ``scores`` and carry a
    ``predicted_boolean_probability`` column.
    """

    supported_sources: ClassVar[frozenset[str]] = frozenset({"input_inference_subdir"})

    def predict(self, cfg: DictConfig, source, index: pl.DataFrame) -> pl.DataFrame:
        from ..lightning.probe import ARTIFACTS_FILENAME

        _, value = source
        inference_dir = resolve_subdir(cfg.input_data_dir, value)
        read_manifest(inference_dir, require_type=ArtifactType.inference, require_kind=InferenceKind.scores)

        artifacts = pl.read_parquet(inference_dir / ARTIFACTS_FILENAME)
        if PROBABILITY_COLUMN not in artifacts.columns:
            raise ValueError(
                f"{inference_dir / ARTIFACTS_FILENAME} has no {PROBABILITY_COLUMN!r} column; scores "
                "materialized for prediction must carry one."
            )
        return index.select(KEYS).join(
            artifacts.select([*KEYS, PROBABILITY_COLUMN]), on=KEYS, how="left"
        )


class ZeroShotPredictCommand(_PredictRunMixin, PredictCommand):
    """Score a task directly from a pretrained model, without task-specific training.

    Resolving a task definition over a model's own outputs — running ACES over generated trajectories, or
    translating the task into a native query — is model-specific, so :meth:`resolve` is abstract. There is
    deliberately no default: a placeholder that returned a constant would produce a schema-valid
    predictions file of pure noise, and every check downstream of here would pass.
    """

    supported_sources: ClassVar[frozenset[str]] = frozenset({"input_pretrained_model_dir"})

    def predict(self, cfg: DictConfig, source, index: pl.DataFrame) -> pl.DataFrame:
        _, value = source
        module = load_trained_module(Path(value))
        return self.resolve(cfg, module, index)

    def resolve(self, cfg: DictConfig, module, index: pl.DataFrame) -> pl.DataFrame:
        """Turn model outputs at the index timepoints into probabilities for ``cfg.external_task_file``.

        Implement this in your model. Parse the task with
        :func:`meds_model_base.tasks.load_task_config` and resolve it over the model's outputs — over
        generated futures for an autoregressive model, or by translating it into a native query.

        Must return ``subject_id``, ``prediction_time`` and ``predicted_boolean_probability`` for **every**
        row of ``index``.
        """
        raise NotImplementedError(
            f"{type(self).__name__}.resolve is not implemented. A zero-shot model must define how a task "
            "is resolved over its own outputs; there is no meaningful default."
        )


def task_definition_path(task_dir: Path | str) -> Path | None:
    """Recover the *definition* (e.g. the ACES YAML) a materialized task came from, via its manifest.

    Zero-shot and query models need to know what to predict, not just where. Rather than add a second task
    argument to ``predict`` — and with it the chance of passing a definition that disagrees with the
    materialized labels — the definition is read from the task artifact that ``preprocess_task`` recorded.

    Returns None when the task was materialized from pre-extracted labels, which carry no definition.
    """
    manifest = read_manifest(task_dir, require_type=ArtifactType.task)
    source = (manifest.get("source") or {}).get("external_task_file")
    if not source or manifest.get("materialization") != "aces_extracted":
        return None
    return Path(source)


def _select_prediction_columns(frame: pl.DataFrame) -> pl.DataFrame:
    """Keep the keys, the split, and the predicted columns emitted by ``predict_step``."""
    if PROBABILITY_COLUMN not in frame.columns:
        raise ValueError(
            f"The model's predict_step did not emit {PROBABILITY_COLUMN!r} (got: "
            f"{', '.join(c for c in frame.columns if c not in KEYS)}). Prediction requires a probability."
        )
    keep = [*KEYS, "split", PROBABILITY_COLUMN]
    if "predicted_boolean_value" in frame.columns:
        keep.append("predicted_boolean_value")
    return frame.select(keep)


__all__ = [
    "PREDICTIONS_FILENAME",
    "CoverageError",
    "task_definition_path",
    "MaterializedPredictCommand",
    "ProbePredictCommand",
    "SupervisedPredictCommand",
    "ZeroShotPredictCommand",
]
