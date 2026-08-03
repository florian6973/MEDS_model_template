"""Builders for synthetic MEDS datasets with a *designed* signal (for property tests).

Everything is written directly in the canonical MEDS on-disk layout (``data/<split>/0.parquet`` +
``metadata/*`` + ``task_labels/<task>/<split>.parquet``) so the model's real ``preprocess`` step can
consume it unchanged.
"""

from __future__ import annotations

import json
import random
from datetime import datetime, timedelta
from pathlib import Path

import polars as pl

from ..schemas import (
    code_metadata_filepath,
    data_subdirectory,
    dataset_metadata_filepath,
    held_out_split,
    subject_splits_filepath,
    train_split,
    tuning_split,
)

#: The marker code whose presence in a subject's history determines the label in a signal dataset.
SIGNAL_CODE = "SIGNAL//POS"
_BACKGROUND = [f"BG//{i}" for i in range(6)]
_BASE_TIME = datetime(2020, 1, 1)

#: Fixed programs for a generative "pattern" dataset (a tiny grammar; cf. MEDS-EIC-AR).
PATTERN_PROGRAMS = {"A": ("P//A0", "P//A1", "P//A2"), "B": ("P//B0", "P//B1")}

#: Static (baseline) measurements every synthetic subject gets: one value-less code drawn from
#: :data:`STATIC_GROUPS` and one numeric :data:`STATIC_NUMERIC`. Both branches of a model's static handling
#: are therefore covered, and a cohort built here is a cohort meds-torch-data can collate under
#: ``static_inclusion_mode: INCLUDE``.
STATIC_GROUPS = ("BASELINE//GROUP_A", "BASELINE//GROUP_B")
STATIC_NUMERIC = "BASELINE//AGE"
STATIC_CODES = (*STATIC_GROUPS, STATIC_NUMERIC)


def _static_rows(subject_id: int, rng: random.Random) -> list[dict]:
    """Static measurements for one subject: MEDS marks them with a null ``time``.

    ``rng`` must be independent of the generator that decides the labels. A static that correlated with the
    outcome would be a second signal, which would make the designed-signal test pass for the wrong reason
    and give the negative control something real to learn.
    """
    return [
        {"subject_id": subject_id, "time": None, "code": rng.choice(STATIC_GROUPS), "numeric_value": None},
        {
            "subject_id": subject_id,
            "time": None,
            "code": STATIC_NUMERIC,
            "numeric_value": rng.uniform(20.0, 90.0),
        },
    ]


def _write_meds(
    root: Path,
    per_split: dict[str, list[dict]],
    codes: list[str],
    task_labels: dict[str, pl.DataFrame] | None = None,
    task_name: str = "signal_task",
) -> Path:
    """Write MEDS ``data/``, ``metadata/`` (+ optional ``task_labels/``) from per-split event-row lists."""
    root = Path(root)
    splits = []
    for split, rows in per_split.items():
        split_dir = root / data_subdirectory / split
        split_dir.mkdir(parents=True, exist_ok=True)
        df = pl.DataFrame(rows).with_columns(
            pl.col("subject_id").cast(pl.Int64),
            pl.col("time").cast(pl.Datetime("us")),
            pl.col("code").cast(pl.Utf8),
            pl.col("numeric_value").cast(pl.Float32),
        )
        df.write_parquet(split_dir / "0.parquet")
        splits.extend({"subject_id": s, "split": split} for s in df["subject_id"].unique().to_list())

    meta_dir = root / "metadata"
    meta_dir.mkdir(parents=True, exist_ok=True)
    pl.DataFrame({"code": codes, "description": codes}).write_parquet(root / code_metadata_filepath)
    pl.DataFrame(splits).with_columns(pl.col("subject_id").cast(pl.Int64)).write_parquet(
        root / subject_splits_filepath
    )
    (root / dataset_metadata_filepath).write_text(
        json.dumps({"dataset_name": "synthetic_signal", "meds_version": "0.4"})
    )

    if task_labels:
        task_dir = root / "task_labels" / task_name
        task_dir.mkdir(parents=True, exist_ok=True)
        for split, labels in task_labels.items():
            labels.write_parquet(task_dir / f"{split}.parquet")
    return root


def build_signal_dataset(
    root: Path,
    *,
    n_train: int = 200,
    n_tuning: int = 40,
    n_held_out: int = 40,
    signal_rate: float = 0.5,
    seed: int = 0,
    shuffle_labels: bool = False,
    with_statics: bool = True,
) -> Path:
    """Write a classifier-signal MEDS dataset: the presence of :data:`SIGNAL_CODE` determines the label.

    Each subject gets a short run of background codes; with probability ``signal_rate`` the ``SIGNAL_CODE``
    is inserted. The boolean label is exactly whether the subject has the signal (so a model that reads the
    sequence can reach AUROC ≈ 1.0). ``shuffle_labels=True`` breaks the signal↔label link (negative
    control: nothing to learn, AUROC ≈ 0.5).

    **Only the signal code is informative.** Sequence length and the signal's position within the sequence
    are both label-independent by construction; see the comment in the loop for why that matters. The
    static measurements are drawn from a separate generator for the same reason.

    ``with_statics`` (default on) gives every subject the baseline measurements described by
    :data:`STATIC_CODES`. It defaults on because a cohort *without* them is not merely a simpler cohort: a
    model whose datamodule sets ``static_inclusion_mode: INCLUDE`` cannot collate it at all —
    meds-torch-data raises ``ValueError: Cannot infer dtype from empty values`` when the static tensors are
    empty for a whole batch, before the model is ever called. A learnability suite that cannot exercise a
    model's static path is not testing that model. Turn it off only to reproduce that case.

    Returns ``root`` (a MEDS dataset with a ``signal_task`` task-labels directory).
    """
    rng = random.Random(seed)
    # Independent of `rng`, so nothing about a subject's baseline variables can carry information about
    # its label — the signal code stays the only thing there is to learn.
    static_rng = random.Random(seed + 9973)
    counts = {train_split: n_train, tuning_split: n_tuning, held_out_split: n_held_out}
    codes = [*_BACKGROUND, SIGNAL_CODE, *(STATIC_CODES if with_statics else ())]

    per_split: dict[str, list[dict]] = {}
    task_labels: dict[str, pl.DataFrame] = {}
    subject_id = 1
    for split, n in counts.items():
        rows: list[dict] = []
        labels: list[dict] = []
        for _ in range(n):
            if with_statics:
                # MEDS puts a subject's static measurements first, marked by a null time.
                rows.extend(_static_rows(subject_id, static_rng))
            has_signal = rng.random() < signal_rate
            t = _BASE_TIME + timedelta(days=int(rng.random() * 100))
            # The signal is *inserted at a random position* and a negative subject gets a filler event, so
            # that neither the signal's position nor the sequence length carries any information about the
            # label. Prepending the signal instead (and leaving negatives one event shorter) leaks it twice
            # over: the code would always sit at position 0, a positive subject would always have exactly
            # one more event than a negative one — a length-only predictor measured AUROC 0.59 — and since
            # prediction_time is derived from the event count, it would differ by class too. A model could
            # then score well here without ever reading SIGNAL_CODE, which is the one thing this dataset
            # exists to check.
            events = [rng.choice(_BACKGROUND) for _ in range(rng.randint(4, 9))]
            if has_signal:
                events.insert(rng.randrange(len(events) + 1), SIGNAL_CODE)
            else:
                events.append(rng.choice(_BACKGROUND))
            for i, code in enumerate(events):
                rows.append(
                    {
                        "subject_id": subject_id,
                        "time": t + timedelta(hours=i),
                        "code": code,
                        "numeric_value": None,
                    }
                )
            prediction_time = t + timedelta(hours=len(events) + 1)
            labels.append(
                {"subject_id": subject_id, "prediction_time": prediction_time, "boolean_value": has_signal}
            )
            subject_id += 1
        per_split[split] = rows
        label_df = pl.DataFrame(labels).with_columns(
            pl.col("subject_id").cast(pl.Int64),
            pl.col("prediction_time").cast(pl.Datetime("us")),
            pl.col("boolean_value").cast(pl.Boolean),
        )
        if shuffle_labels:
            shuffled = label_df["boolean_value"].shuffle(seed=seed + 1)
            label_df = label_df.with_columns(shuffled.alias("boolean_value"))
        task_labels[split] = label_df

    return _write_meds(root, per_split, codes, task_labels)


def build_pattern_dataset(
    root: Path,
    *,
    n_train: int = 200,
    n_tuning: int = 40,
    n_held_out: int = 40,
    seed: int = 0,
    with_statics: bool = True,
) -> Path:
    """Write a generative-signal MEDS dataset: each subject is a run of fixed code *programs*.

    Used to test autoregressive generation: a model that learns the grammar should generate valid programs.
    Also emits a ``pattern_task`` index (one prediction_time per subject) for the generation entry point.

    ``with_statics`` carries the same meaning and the same default as in :func:`build_signal_dataset`: the
    baseline measurements are what makes the cohort collatable under ``static_inclusion_mode: INCLUDE``.
    They are not part of the grammar and are never emitted into the dynamic sequence, so a model generating
    programs does not have to generate them.
    """
    rng = random.Random(seed)
    static_rng = random.Random(seed + 9973)
    counts = {train_split: n_train, tuning_split: n_tuning, held_out_split: n_held_out}
    codes = sorted({c for prog in PATTERN_PROGRAMS.values() for c in prog})
    if with_statics:
        codes += list(STATIC_CODES)

    per_split: dict[str, list[dict]] = {}
    task_labels: dict[str, pl.DataFrame] = {}
    subject_id = 1
    for split, n in counts.items():
        rows: list[dict] = []
        labels: list[dict] = []
        for _ in range(n):
            if with_statics:
                rows.extend(_static_rows(subject_id, static_rng))
            t = _BASE_TIME + timedelta(days=int(rng.random() * 100))
            seq: list[str] = []
            for _ in range(rng.randint(3, 6)):
                seq += list(PATTERN_PROGRAMS[rng.choice(list(PATTERN_PROGRAMS))])
            for i, code in enumerate(seq):
                rows.append(
                    {
                        "subject_id": subject_id,
                        "time": t + timedelta(hours=i),
                        "code": code,
                        "numeric_value": None,
                    }
                )
            labels.append({"subject_id": subject_id, "prediction_time": t + timedelta(hours=len(seq) // 2)})
            subject_id += 1
        per_split[split] = rows
        task_labels[split] = pl.DataFrame(labels).with_columns(
            pl.col("subject_id").cast(pl.Int64), pl.col("prediction_time").cast(pl.Datetime("us"))
        )

    return _write_meds(root, per_split, codes, task_labels, task_name="pattern_task")
