"""Small example MEDS cohort for a model-owned primary-behavior test."""

from __future__ import annotations

import json
import random
from datetime import datetime, timedelta
from pathlib import Path

import polars as pl

SIGNAL_CODE = "SIGNAL//POS"
BACKGROUND_CODES = [f"BG//{index}" for index in range(6)]


def build_signal_dataset(
    root: Path,
    *,
    n_train: int = 200,
    n_tuning: int = 40,
    n_held_out: int = 40,
    seed: int = 0,
    shuffle_labels: bool = False,
    signal_code: str = SIGNAL_CODE,
    background_codes: list[str] | None = None,
) -> Path:
    """Write a tiny MEDS dataset where ``signal_code`` alone determines the binary label.

    Positive and negative subjects have equal-length histories. The signal position and history length
    are randomized independently of the label, so a model must represent the code rather than exploit a
    length or position leak. Shuffling labels supplies the paired negative control.

    This is an example fixture, not an input representation imposed on the model. Adapt or replace it
    when the implementation needs codes from a fixed vocabulary, predicates, numeric values, or another
    model-specific binding.
    """
    root = Path(root)
    rng = random.Random(seed)
    background = list(background_codes or BACKGROUND_CODES)
    if not background:
        raise ValueError("background_codes must contain at least one code")

    split_sizes = {"train": n_train, "tuning": n_tuning, "held_out": n_held_out}
    subject_splits: list[dict] = []
    subject_id = 1
    base_time = datetime(2020, 1, 1)

    for split, size in split_sizes.items():
        events: list[dict] = []
        labels: list[dict] = []
        for _ in range(size):
            positive = rng.random() < 0.5
            start = base_time + timedelta(days=rng.randrange(100))
            sequence = [rng.choice(background) for _ in range(rng.randint(4, 9))]
            marker_position = rng.randrange(len(sequence) + 1)
            marker = signal_code if positive else rng.choice(background)
            sequence.insert(marker_position, marker)

            for offset, code in enumerate(sequence):
                events.append(
                    {
                        "subject_id": subject_id,
                        "time": start + timedelta(hours=offset),
                        "code": code,
                        "numeric_value": None,
                    }
                )
            labels.append(
                {
                    "subject_id": subject_id,
                    "prediction_time": start + timedelta(hours=len(sequence) + 1),
                    "boolean_value": positive,
                }
            )
            subject_splits.append({"subject_id": subject_id, "split": split})
            subject_id += 1

        data_dir = root / "data" / split
        labels_dir = root / "task_labels" / "signal_task"
        data_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)
        pl.DataFrame(events).with_columns(
            pl.col("subject_id").cast(pl.Int64),
            pl.col("time").cast(pl.Datetime("us")),
            pl.col("code").cast(pl.String),
            pl.col("numeric_value").cast(pl.Float32),
        ).write_parquet(data_dir / "0.parquet")
        label_frame = pl.DataFrame(labels).with_columns(
            pl.col("subject_id").cast(pl.Int64),
            pl.col("prediction_time").cast(pl.Datetime("us")),
            pl.col("boolean_value").cast(pl.Boolean),
        )
        if shuffle_labels:
            label_frame = label_frame.with_columns(
                pl.col("boolean_value").shuffle(seed=seed + 1).alias("boolean_value")
            )
        label_frame.write_parquet(labels_dir / f"{split}.parquet")

    metadata = root / "metadata"
    metadata.mkdir(parents=True, exist_ok=True)
    pl.DataFrame({"code": [*background, signal_code]}).write_parquet(metadata / "codes.parquet")
    pl.DataFrame(subject_splits).with_columns(pl.col("subject_id").cast(pl.Int64)).write_parquet(
        metadata / "subject_splits.parquet"
    )
    (metadata / "dataset.json").write_text(
        json.dumps({"dataset_name": "synthetic_signal", "meds_version": "0.4"})
    )
    return root
