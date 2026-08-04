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
    signal_code: str = SIGNAL_CODE,
    background: list[str] | None = None,
) -> Path:
    """Write a classifier-signal MEDS dataset: the presence of ``signal_code`` determines the label.

    Each subject gets a short run of background codes; with probability ``signal_rate`` the ``signal_code``
    is inserted. The boolean label is exactly whether the subject has the signal (so a model that reads the
    sequence can reach AUROC ≈ 1.0). ``shuffle_labels=True`` breaks the signal↔label link (negative
    control: nothing to learn, AUROC ≈ 0.5).

    **Only the signal code is informative.** Sequence length and the signal's position within the sequence
    are both label-independent by construction; see the comment in the loop for why that matters.

    ``signal_code`` / ``background`` default to this module's fixed vocabulary; a featurized repo passes
    codes drawn from its own predicates file instead (:func:`signal_dataset_from_predicates`), so the
    model is tested on its *production* feature space.

    Returns ``root`` (a MEDS dataset with a ``signal_task`` task-labels directory).
    """
    rng = random.Random(seed)
    background = list(background) if background else list(_BACKGROUND)
    counts = {train_split: n_train, tuning_split: n_tuning, held_out_split: n_held_out}
    codes = [*background, signal_code]

    per_split: dict[str, list[dict]] = {}
    task_labels: dict[str, pl.DataFrame] = {}
    subject_id = 1
    for split, n in counts.items():
        rows: list[dict] = []
        labels: list[dict] = []
        for _ in range(n):
            has_signal = rng.random() < signal_rate
            t = _BASE_TIME + timedelta(days=int(rng.random() * 100))
            # The signal is *inserted at a random position* and a negative subject gets a filler event, so
            # that neither the signal's position nor the sequence length carries any information about the
            # label. Prepending the signal instead (and leaving negatives one event shorter) leaks it twice
            # over: the code would always sit at position 0, a positive subject would always have exactly
            # one more event than a negative one — a length-only predictor measured AUROC 0.59 — and since
            # prediction_time is derived from the event count, it would differ by class too. A model could
            # then score well here without ever reading the signal code, which is the one thing this
            # dataset exists to check.
            events = [rng.choice(background) for _ in range(rng.randint(4, 9))]
            if has_signal:
                events.insert(rng.randrange(len(events) + 1), signal_code)
            else:
                events.append(rng.choice(background))
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


def signal_dataset_from_predicates(
    root: Path,
    predicates_file: Path | str,
    *,
    seed: int = 0,
    shuffle_labels: bool = False,
) -> Path:
    """A signal dataset whose codes come from **the model's own predicates file** (the one-file principle).

    The dependency inverts: instead of a predicates fixture matched to a fixed synthetic vocabulary, the
    dataset adapts to the predicates. The predicates with literal codes (exact / any-of — a regex cannot
    be reverse-instantiated into a code) are collected; the first becomes the signal predicate (its first
    code planted with the label), the rest are distractors (their codes emitted label-independently). The
    feature space the model is then tested on is its production one: same names, same order, same
    ``features.json``.

    Distractors are why at least **two** literal-code predicates are required: with a single feature the
    column trivially equals the answer and the test stops checking that the model weights the right
    feature. Callers should ``pytest.skip`` on the ValueError this raises.

    Raises:
        ValueError: if fewer than two predicates with literal codes remain after parsing.
    """
    import yaml

    from ..featurize import literal_code_predicates, parse_predicates

    raw = yaml.safe_load(Path(predicates_file).read_text())["predicates"]
    literal = literal_code_predicates(parse_predicates(raw), raw)
    if len(literal) < 2:
        raise ValueError(
            f"The designed-signal test needs at least two predicates with literal codes (exact or "
            f"any-of) in {predicates_file}; found {len(literal)} ({', '.join(literal) or 'none'}). "
            "Add one, or provide example codes for your regex predicates."
        )
    names = list(literal)
    signal_code = literal[names[0]][0]
    background = sorted({c for name in names[1:] for c in literal[name]} - {signal_code})
    return build_signal_dataset(
        root, seed=seed, shuffle_labels=shuffle_labels, signal_code=signal_code, background=background
    )


def build_pattern_dataset(
    root: Path, *, n_train: int = 200, n_tuning: int = 40, n_held_out: int = 40, seed: int = 0
) -> Path:
    """Write a generative-signal MEDS dataset: each subject is a run of fixed code *programs*.

    Used to test autoregressive generation: a model that learns the grammar should generate valid programs.
    Also emits a ``pattern_task`` index (one prediction_time per subject) for the generation entry point.
    """
    rng = random.Random(seed)
    counts = {train_split: n_train, tuning_split: n_tuning, held_out_split: n_held_out}
    codes = sorted({c for prog in PATTERN_PROGRAMS.values() for c in prog})

    per_split: dict[str, list[dict]] = {}
    task_labels: dict[str, pl.DataFrame] = {}
    subject_id = 1
    for split, n in counts.items():
        rows: list[dict] = []
        labels: list[dict] = []
        for _ in range(n):
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
