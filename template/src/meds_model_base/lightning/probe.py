"""Inference-artifact plumbing for the representation-probe chain.

The chain (``pretrain`` → ``infer`` embeddings → ``supervised_train`` → ``predict``) never re-runs the
foundation model downstream. Once embeddings are materialized, training a task head is a dense-feature
problem: no tokenization, no meds-torch-data, just a join on ``(subject_id, prediction_time)``.

What lives here is knowledge of the *artifact layout* — where the feature column is, how it joins to a
task, and whether the join covered the index. The probe head itself is a model and lives in the generated
repository, built through ``cfg.model`` like every other trainable module.

That join is the load-bearing part: an embedding table and a task are both keyed on prediction timepoints,
but nothing guarantees they were built from the same index, so coverage is measured rather than assumed.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl
import torch

from ..schemas import SPLITS, train_split, tuning_split

if TYPE_CHECKING:  # pragma: no cover - typing only
    import lightning.pytorch as L
    from omegaconf import DictConfig

logger = logging.getLogger(__name__)

ARTIFACTS_FILENAME = "artifacts.parquet"

KEYS = ["subject_id", "prediction_time"]


def _feature_column(df: pl.DataFrame) -> str:
    """Pick the embedding column: the single list-typed column that is not a key."""
    candidates = [
        name
        for name, dtype in zip(df.columns, df.dtypes, strict=True)
        if name not in KEYS and isinstance(dtype, pl.List)
    ]
    if not candidates:
        raise ValueError(
            f"No list-valued feature column in {ARTIFACTS_FILENAME} (columns: {', '.join(df.columns)}). "
            "A probe needs a vector-valued column such as `embedding`."
        )
    if len(candidates) > 1:
        raise ValueError(
            f"Ambiguous feature columns in {ARTIFACTS_FILENAME}: {', '.join(candidates)}. A probe needs "
            "exactly one vector-valued column."
        )
    return candidates[0]


def load_probe_frames(
    inference_dir: Path, task_dir: Path
) -> tuple[dict[str, pl.DataFrame], str, dict[str, object]]:
    """Join inference artifacts to task labels per split.

    Returns ``(frames_by_split, feature_column, coverage)``. ``coverage`` reports how many label rows found
    a matching embedding, so a probe trained on a fraction of the task is visible in the manifest instead of
    looking like a complete run.
    """
    artifacts = pl.read_parquet(Path(inference_dir) / ARTIFACTS_FILENAME)
    feature_column = _feature_column(artifacts)
    artifacts = artifacts.select([*KEYS, feature_column])

    frames: dict[str, pl.DataFrame] = {}
    coverage: dict[str, object] = {}
    for split in SPLITS:
        fp = Path(task_dir) / f"{split}.parquet"
        if not fp.is_file():
            continue
        labels = pl.read_parquet(fp).select([*KEYS, "boolean_value"])
        joined = labels.join(artifacts, on=KEYS, how="inner")
        coverage[split] = {"labels": len(labels), "matched": len(joined)}
        if len(joined) < len(labels):
            logger.warning(
                "Split %s: only %d of %d label rows have embeddings in %s.",
                split,
                len(joined),
                len(labels),
                inference_dir,
            )
        if len(joined):
            frames[split] = joined

    if train_split not in frames:
        raise ValueError(
            f"No training rows after joining {task_dir} to the embeddings in {inference_dir}. The inference "
            "artifacts were most likely materialized for a different index than this task."
        )
    return frames, feature_column, {"coverage": coverage}


def _to_tensors(df: pl.DataFrame, feature_column: str) -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.tensor(df[feature_column].to_list(), dtype=torch.float32)
    y = torch.tensor(df["boolean_value"].to_list(), dtype=torch.float32)
    return x, y


def build_probe_dataloaders(
    frames: dict[str, pl.DataFrame], feature_column: str, cfg: DictConfig
) -> tuple[dict[str, object], int]:
    """Build ``{train_dataloaders, val_dataloaders}`` from the joined frames; return them with the dim."""
    from torch.utils.data import DataLoader, TensorDataset

    batch_size = int(cfg.get("batch_size", 32))
    num_workers = int(cfg.get("num_workers", 0))

    x_train, y_train = _to_tensors(frames[train_split], feature_column)
    loaders: dict[str, object] = {
        "train_dataloaders": DataLoader(
            TensorDataset(x_train, y_train),
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
        )
    }
    if tuning_split in frames:
        x_val, y_val = _to_tensors(frames[tuning_split], feature_column)
        loaders["val_dataloaders"] = DataLoader(
            TensorDataset(x_val, y_val), batch_size=batch_size, num_workers=num_workers
        )
    return loaders, int(x_train.shape[1])


def probe_predict_frame(
    inference_dir: Path, index: pl.DataFrame, module: L.LightningModule, batch_size: int = 256
) -> pl.DataFrame:
    """Score an index of timepoints from materialized embeddings (used by ``predict``).

    Every index row must have an embedding: a probe that cannot score part of the index would otherwise
    emit a short predictions file, which the coverage check in ``predict`` treats as a failure.
    """
    artifacts = pl.read_parquet(Path(inference_dir) / ARTIFACTS_FILENAME)
    feature_column = _feature_column(artifacts)
    joined = index.join(artifacts.select([*KEYS, feature_column]), on=KEYS, how="left")

    missing = joined.filter(pl.col(feature_column).is_null())
    if len(missing):
        raise ValueError(
            f"{len(missing)} of {len(index)} index rows have no embedding in {inference_dir}. Re-run `infer` "
            "with input_task_subdir pointing at this task so every prediction timepoint is covered."
        )

    x = torch.tensor(joined[feature_column].to_list(), dtype=torch.float32)
    module.eval()
    probs: list[float] = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            probs.extend(torch.sigmoid(module(x[start : start + batch_size])).tolist())
    return joined.select(KEYS).with_columns(
        pl.Series("predicted_boolean_probability", probs, dtype=pl.Float32)
    )


__all__ = [
    "ARTIFACTS_FILENAME",
    "build_probe_dataloaders",
    "load_probe_frames",
    "probe_predict_frame",
]
