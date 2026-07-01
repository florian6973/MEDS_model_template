"""The ``prediction`` step (e): index timepoints (+ optional task) → ``predictions.parquet``.

The concrete ``run`` (shared by every prediction step) is fixed:

1. call the model's :meth:`~meds_model_base.steps.base.PredictionStep.predict` hook;
2. validate the result against ``meds_evaluation.PredictionSchema`` and write a single
   ``predictions.parquet``.

The repo ends there — scoring is a separate, shared tool, and no ground-truth label is ever read.

The *index* of timepoints reaches the model through the datamodule: the ``_prediction`` config sets
``datamodule.config.task_labels_dir = ${index}`` and ``TO_END`` sampling, so **meds-torch-data handles the
timestep alignment** and its dataset exposes the ordered ``(subject_id, prediction_time)`` keys via
``schema_df``. We do not hand-build datasets or re-align outputs.

``SupervisedPredictionStep`` is the default for supervised / fine-tuned models (it ignores ``cfg.task``):
it runs the trained ``LightningModule``'s ``predict_step`` over that datamodule and reads the keys off the
dataset's ``schema_df``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl
import pyarrow.parquet as pq

from ..schemas import held_out_split, validate_predictions
from ..utils import find_checkpoint
from .base import PredictionStep

if TYPE_CHECKING:  # pragma: no cover - typing only
    import lightning.pytorch as pl_light
    from omegaconf import DictConfig

logger = logging.getLogger(__name__)

PREDICTIONS_FILENAME = "predictions.parquet"

#: MEDS split name → the MTD ``Datamodule`` dataset/dataloader attribute pair (non-shuffling ones).
_SPLIT_TO_DM = {
    "train": ("train_dataset", "train_dataloader"),
    "tuning": ("val_dataset", "val_dataloader"),
    held_out_split: ("test_dataset", "test_dataloader"),
}


class _PredictionRunMixin:
    """The fixed ``run`` contract: ``predict`` → validate → write ``predictions.parquet``."""

    def run(self, cfg: "DictConfig") -> Path:
        predictions = self.predict(cfg)  # type: ignore[attr-defined]
        output_dir = Path(cfg.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        table = validate_predictions(predictions.to_arrow())
        out_fp = output_dir / PREDICTIONS_FILENAME
        pq.write_table(table, out_fp)
        logger.info("Wrote %d predictions to %s.", table.num_rows, out_fp)
        return output_dir


class SupervisedPredictionStep(_PredictionRunMixin, PredictionStep):
    """Default prediction for supervised / fine-tuned models (ignores ``cfg.task``)."""

    def predict(self, cfg: "DictConfig") -> pl.DataFrame:
        module = load_trained_module(cfg, Path(cfg.model_initialization_dir))
        module.eval()
        split = cfg.get("split") or held_out_split
        keys, outputs = run_predict_step(cfg, module, split)
        return attach_outputs(keys, outputs, "predicted_boolean_probability")


class ZeroShotPredictionStep(_PredictionRunMixin, PredictionStep):
    """Prediction for zero-shot / query models: resolve the ACES ``task`` over the model's output.

    **Resolving a task over a model's own outputs (e.g. running ACES over generated trajectories, or
    translating the task into a native query) is model-specific and typically delegated to a separate,
    shared tool — it is intentionally not implemented here.** Override :meth:`resolve` to plug in your
    resolver. The default uses whatever ``predicted_boolean_probability`` the model's ``predict_step``
    emits (so a query model that answers directly just works), falling back to a constant ``0.5`` so the
    pipeline still runs and produces a ``PredictionSchema``-valid file.
    """

    def predict(self, cfg: "DictConfig") -> pl.DataFrame:
        module = load_trained_module(cfg, Path(cfg.model_initialization_dir))
        module.eval()
        split = cfg.get("split") or held_out_split
        keys, outputs = run_predict_step(cfg, module, split)
        probs = self.resolve(cfg, keys, outputs)
        return keys.with_columns(pl.Series("predicted_boolean_probability", probs, dtype=pl.Float32))

    def resolve(self, cfg: "DictConfig", keys: pl.DataFrame, outputs: list[dict]) -> list[float]:
        """Turn model outputs at the index timepoints into probabilities for the ``cfg.task``.

        Default: pass through the model's ``predicted_boolean_probability`` if present, else ``0.5``.
        Override to run your task resolver (parse ``cfg.task`` via ``meds_model_base.tasks.load_task_config``
        and resolve it over generated trajectories / a native query — usually a separate tool).
        """
        if outputs and "predicted_boolean_probability" in outputs[0]:
            import torch

            return torch.cat(
                [o["predicted_boolean_probability"].flatten() for o in outputs]
            ).float().tolist()
        logger.warning(
            "ZeroShotPredictionStep.resolve is using the placeholder constant 0.5. Override `resolve` "
            "(or have the model's predict_step emit predicted_boolean_probability) to make real predictions."
        )
        return [0.5] * len(keys)


# ------------------------------------------------------------------------------------------------------
# Shared helpers (reused by profile-specific prediction / inference steps)
# ------------------------------------------------------------------------------------------------------


def load_trained_module(cfg: "DictConfig", model_dir: Path) -> "pl_light.LightningModule":
    """Reconstruct the trained ``LightningModule`` from a run dir's checkpoint.

    Uses ``cfg.model._target_`` to find the class and ``LightningModule.load_from_checkpoint`` (which reads
    the saved hyperparameters), so no data or re-instantiation kwargs are needed.
    """
    from hydra.utils import get_class

    ckpt_fp = find_checkpoint(model_dir)
    if ckpt_fp is None:
        raise FileNotFoundError(f"No checkpoint found in model_initialization_dir {model_dir}.")
    model_cls = get_class(cfg.model._target_)
    logger.info("Loading %s from %s.", model_cls.__name__, ckpt_fp)
    return model_cls.load_from_checkpoint(ckpt_fp, map_location="cpu")


def split_dataset_and_loader(cfg: "DictConfig", split: str):
    """Return ``(dataset, dataloader)`` for ``split`` from MTD's datamodule (built from ``cfg.datamodule``).

    The datamodule config points ``task_labels_dir`` at the index, so the dataset's ``schema_df`` carries
    the ordered ``(subject_id, prediction_time)`` keys. The non-shuffling ``{split}_dataloader`` yields
    batches in that same order.
    """
    from hydra.utils import instantiate

    datamodule = instantiate(cfg.datamodule)
    dataset_attr, loader_attr = _SPLIT_TO_DM[split]
    dataset = getattr(datamodule, dataset_attr)
    dataloader = getattr(datamodule, loader_attr)()
    return dataset, dataloader


def run_predict_step(cfg: "DictConfig", module: "pl_light.LightningModule", split: str):
    """Run ``trainer.predict`` over ``split`` and return ``(keys_df, per_batch_outputs)``.

    ``keys_df`` is ``schema_df[[subject_id, prediction_time]]`` (the alignment MTD already computed).
    """
    import lightning.pytorch as L

    dataset, dataloader = split_dataset_and_loader(cfg, split)
    keys = dataset.schema_df.select(["subject_id", "prediction_time"])
    if len(dataset) == 0:
        logger.warning("No index rows for split %r after joining to the tensorized cohort.", split)
        return keys, []

    trainer = L.Trainer(
        accelerator="auto", devices=1, logger=False, enable_checkpointing=False, enable_progress_bar=False
    )
    outputs = trainer.predict(module, dataloaders=dataloader)
    return keys, outputs


def attach_outputs(keys: pl.DataFrame, outputs: list[dict], probability_key: str) -> pl.DataFrame:
    """Attach concatenated per-batch ``predict_step`` outputs to the ordered ``keys`` dataframe."""
    import torch

    if not outputs:
        return keys.with_columns(pl.Series(probability_key, [], dtype=pl.Float32))
    probs = torch.cat([o[probability_key].flatten() for o in outputs]).float().tolist()
    result = keys.with_columns(pl.Series(probability_key, probs, dtype=pl.Float32))
    if "predicted_boolean_value" in outputs[0]:
        bvals = torch.cat([o["predicted_boolean_value"].flatten() for o in outputs]).bool().tolist()
        result = result.with_columns(pl.Series("predicted_boolean_value", bvals))
    return result
