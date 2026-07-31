"""Shared runtime for the two inference-time commands (``infer`` and ``predict``).

Loading a trained module, resolving the index of timepoints, and running a batched forward pass are the
same job in both commands; only what is done with the outputs differs.

The module class comes from the **model artifact's manifest**, not from the config. A checkpoint is only
loadable by the class that wrote it, and that class is a property of the artifact — reading it from the
current config would happily load a supervised classifier's weights into whatever ``cfg.model`` happens to
name today, which fails confusingly at best and silently mis-scores at worst.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl

from ..manifest import read_manifest
from ..utils import require_checkpoint

if TYPE_CHECKING:  # pragma: no cover - typing only
    import lightning.pytorch as pl_light
    from omegaconf import DictConfig

logger = logging.getLogger(__name__)

KEYS = ["subject_id", "prediction_time"]

#: MEDS split name → the meds-torch-data ``Datamodule`` (dataset, dataloader) attribute pair. Only the
#: non-shuffling loaders are listed: prediction order must match ``schema_df`` order.
SPLIT_ATTRS = {
    "train": ("train_dataset", "train_dataloader"),
    "tuning": ("val_dataset", "val_dataloader"),
    "held_out": ("test_dataset", "test_dataloader"),
}


def load_trained_module(model_dir: Path | str) -> pl_light.LightningModule:
    """Reconstruct a trained ``LightningModule`` from a published model artifact.

    Reads ``module_class`` from the artifact's manifest and uses ``load_from_checkpoint`` (which restores
    the saved hyperparameters), so no configuration needs to agree with how the model was built.
    """
    from hydra.utils import get_class

    model_dir = Path(model_dir)
    manifest = read_manifest(model_dir)
    module_class = manifest.get("module_class")
    if not module_class:
        raise ValueError(
            f"The manifest in {model_dir} does not record a `module_class`, so the checkpoint cannot be "
            "loaded. It was probably written by an older version of this template; re-run training."
        )
    ckpt_fp = require_checkpoint(model_dir)
    cls = get_class(module_class)
    logger.info("Loading %s from %s.", cls.__name__, ckpt_fp)
    module = cls.load_from_checkpoint(ckpt_fp, map_location="cpu")
    module.eval()
    return module


def resolve_workspace(cfg: DictConfig, source_path: Path | str | None) -> Path:
    """Determine the ``data_dir`` to run against, preferring an explicit argument.

    When ``input_data_dir`` is not given, it is recovered from the source artifact's manifest. That is what
    makes ``predict`` usable under MEDS-DEV, where ``model_initialization_dir`` rolls forward to the most
    recent *training* output while the tensorized workspace may live under an earlier one — the training
    artifact recorded which workspace it used, so nothing has to be re-derived or re-tensorized.
    """
    from ..manifest import recover_input

    if cfg.get("input_data_dir"):
        return Path(cfg.input_data_dir)
    if source_path is None:
        raise ValueError(
            "input_data_dir was not given and there is no source artifact to recover it from. A model that "
            "ships its own weights must be told where the preprocessed data is."
        )
    recovered = recover_input(source_path, "input_data_dir")
    if recovered is None:
        raise ValueError(
            f"input_data_dir was not given, and {source_path} does not record one in its manifest. Pass "
            "input_data_dir=... explicitly."
        )
    if not recovered.is_dir():
        raise FileNotFoundError(
            f"{source_path} was built against {recovered}, which no longer exists. Pass input_data_dir=... "
            "to point at the workspace's current location."
        )
    logger.info("Recovered input_data_dir=%s from %s.", recovered, source_path)
    return recovered


def load_index(task_dir: Path | str, splits: list[str] | None = None) -> pl.DataFrame:
    """Load the timepoints to score from materialized labels, **dropping any ground-truth labels**.

    This command graph never reads ground truth at inference time: it needs only *where* to predict.
    ``boolean_value`` is discarded here so that no downstream code can accidentally consume it.

    Returns a frame of ``subject_id, prediction_time, split``.
    """
    task_dir = Path(task_dir)
    wanted = splits or [s for s in SPLIT_ATTRS if (task_dir / f"{s}.parquet").is_file()]
    frames = []
    for split in wanted:
        fp = task_dir / f"{split}.parquet"
        if not fp.is_file():
            raise FileNotFoundError(
                f"No {split}.parquet under {task_dir} (requested splits: {wanted}). The labels supplied "
                "as external_labels_dir contain no rows for that split."
            )
        frames.append(pl.read_parquet(fp).select(KEYS).with_columns(pl.lit(split).alias("split")))
    if not frames:
        raise FileNotFoundError(f"No split parquets found in {task_dir}.")
    return pl.concat(frames, how="vertical_relaxed")


def split_dataset_and_loader(cfg: DictConfig, split: str):
    """Return ``(dataset, dataloader)`` for ``split`` from the meds-torch-data datamodule."""
    from ..lightning import build_datamodule

    datamodule = build_datamodule(cfg)
    dataset_attr, loader_attr = SPLIT_ATTRS[split]
    return getattr(datamodule, dataset_attr), getattr(datamodule, loader_attr)()


def run_predict_step(cfg: DictConfig, module: pl_light.LightningModule, split: str):
    """Run ``trainer.predict`` over one split; return ``(keys_df, per_batch_outputs)``.

    ``keys_df`` comes from the dataset's ``schema_df``, which is the alignment meds-torch-data already
    computed between the index and the tensorized cohort — models never hand-align their outputs.
    """
    import lightning.pytorch as L

    dataset, dataloader = split_dataset_and_loader(cfg, split)
    keys = dataset.schema_df.select(KEYS)
    if len(dataset) == 0:
        return keys, []

    trainer = L.Trainer(
        accelerator="auto",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
    )
    return keys, trainer.predict(module, dataloaders=dataloader)


def stack_outputs(outputs: list[dict]) -> list[pl.Series]:
    """Turn per-batch ``predict_step`` dicts into aligned polars columns.

    Scalar tensors (``[B]``) become scalar columns; vector tensors (``[B, D]``) become list columns.
    """
    import torch

    if not outputs:
        return []
    columns: list[pl.Series] = []
    for key in outputs[0]:
        parts = [o[key] for o in outputs]
        if hasattr(parts[0], "dim") and parts[0].dim() >= 2:
            values = [row.tolist() for part in parts for row in part.detach().cpu()]
        else:
            values = torch.cat([p.flatten() for p in parts]).detach().cpu().tolist()
        columns.append(pl.Series(key, values))
    return columns


def resolve_splits(cfg: DictConfig) -> list[str] | None:
    """Normalize the optional ``splits`` argument to a list, or None for "every split present"."""
    raw = cfg.get("splits")
    if raw is None:
        return None
    if isinstance(raw, str):
        return [raw]
    return list(raw)


__all__ = [
    "KEYS",
    "SPLIT_ATTRS",
    "load_index",
    "resolve_workspace",
    "load_trained_module",
    "resolve_splits",
    "run_predict_step",
    "split_dataset_and_loader",
    "stack_outputs",
]
