"""Turning an external labels directory into the split layout meds-torch-data expects.

A task reaches this model as ``external_labels_dir``: a directory of parquet files in the MEDS label
format (``subject_id``, ``prediction_time``, ``boolean_value``), exactly what MEDS-DEV's ``meds-dev-task``
produces and passes to models. The template does **not** extract tasks from ACES definitions — that is
``meds-dev-task``'s job upstream, and duplicating it here would mean reimplementing dataset-specific
predicate resolution.

What remains is genuinely model-side: labels have to be partitioned into ``{train,tuning,held_out}.parquet``
before meds-torch-data can use them as a ``task_labels_dir``. That is a per-command implementation detail
rather than a published artifact — :func:`materialize_labels` writes it into a command's work directory.

Consequently a *task definition* never enters this package. A zero-shot model that needs to know **what**
it is predicting, not merely where, must obtain that itself (weights, a shipped mapping, an extra
override); see the rendered ``predict.py``.
"""

from __future__ import annotations

import logging
from pathlib import Path

import polars as pl

from .schemas import SPLITS, LabelSchema, subject_splits_filepath

logger = logging.getLogger(__name__)

#: Column aliases accepted from a hand-built or upstream label table, in priority order.
_COLUMN_ALIASES = {
    "subject_id": ("subject_id", "patient_id"),
    "prediction_time": ("prediction_time", "index_timestamp", "trigger"),
    "boolean_value": ("boolean_value", "label"),
}


class TaskMaterializationError(RuntimeError):
    """Raised when a labels directory cannot be turned into split label parquets."""


def normalize_label_columns(df: pl.DataFrame, *, source: str) -> pl.DataFrame:
    """Rename a label table's columns to the canonical MEDS names and drop the rest.

    Upstream label files vary in what they call the trigger time and the label, and the spellings have
    changed across tool versions. Rather than pin one, accept the known aliases and fail with the observed
    columns listed when none matches.

    Examples:
        >>> import polars as pl
        >>> from datetime import datetime
        >>> df = pl.DataFrame({"subject_id": [1], "trigger": [datetime(2020, 1, 1)], "label": [True]})
        >>> normalize_label_columns(df, source="test").columns
        ['subject_id', 'prediction_time', 'boolean_value']
    """
    rename: dict[str, str] = {}
    for canonical, aliases in _COLUMN_ALIASES.items():
        for alias in aliases:
            if alias in df.columns:
                rename[alias] = canonical
                break
        else:
            raise TaskMaterializationError(
                f"{source} has no column for {canonical!r} (tried {', '.join(aliases)}). "
                f"Columns present: {', '.join(df.columns)}."
            )
    return df.rename(rename).select(["subject_id", "prediction_time", "boolean_value"])


def read_labels(external_labels_dir: Path | str) -> pl.DataFrame | dict[str, pl.DataFrame]:
    """Read a labels directory (or single parquet) into canonical form.

    A directory may be organized either way, and both occur:

    - **already split** — ``{train,tuning,held_out}.parquet``. Returned keyed by split; no subject-split
      table is needed.
    - **arbitrary shards** — any other ``*.parquet`` layout, which is what MEDS-DEV and the
      ``meds_testing_helpers`` fixtures produce (``labels_A.parquet``, ``labels_B.parquet``, …). Returned
      as one frame, to be partitioned by :func:`split_labels`.

    Shard names carry no meaning, so they are concatenated rather than interpreted — guessing a split from
    a filename would silently mis-assign subjects.
    """
    p = Path(external_labels_dir)
    if p.is_dir():
        by_split = {
            split: pl.read_parquet(p / f"{split}.parquet")
            for split in SPLITS
            if (p / f"{split}.parquet").is_file()
        }
        if by_split:
            return {
                split: normalize_label_columns(df, source=f"{p / f'{split}.parquet'}")
                for split, df in by_split.items()
            }

        shards = sorted(p.rglob("*.parquet"))
        if not shards:
            raise TaskMaterializationError(f"{p} is a directory but contains no parquet files.")
        logger.info("Reading %d label shard(s) from %s.", len(shards), p)
        return pl.concat(
            [normalize_label_columns(pl.read_parquet(fp), source=str(fp)) for fp in shards],
            how="vertical_relaxed",
        ).unique(maintain_order=True)

    if not p.is_file():
        raise TaskMaterializationError(f"external_labels_dir {p} does not exist.")
    return normalize_label_columns(pl.read_parquet(p), source=str(p))


def load_subject_splits(patients_dir: Path | str) -> pl.DataFrame:
    """Load the subject-split table from a published ``patients/`` artifact.

    ``preprocess_data`` copies this out of the source dataset precisely so that later commands do not
    depend on the raw MEDS directory still existing — meds-torch-data does not preserve it itself.
    """
    fp = Path(patients_dir) / subject_splits_filepath
    if not fp.is_file():
        raise TaskMaterializationError(
            f"No subject splits at {fp}. The patients artifact was built by an older version of this "
            "template; re-run `preprocess_data` to record them."
        )
    return pl.read_parquet(fp)


def split_labels(labels: pl.DataFrame, splits: pl.DataFrame) -> dict[str, pl.DataFrame]:
    """Partition a label table by MEDS subject split.

    Subjects absent from the split table are dropped with a warning: they cannot be assigned to train,
    tuning or held_out, so silently keeping them would put unassigned subjects into an arbitrary split.
    """
    joined = labels.join(splits.select(["subject_id", "split"]), on="subject_id", how="left")
    unassigned = joined.filter(pl.col("split").is_null())
    if len(unassigned):
        logger.warning(
            "Dropping %d label rows for %d subjects absent from subject_splits.parquet.",
            len(unassigned),
            unassigned["subject_id"].n_unique(),
        )
    out: dict[str, pl.DataFrame] = {}
    for split in SPLITS:
        part = joined.filter(pl.col("split") == split).drop("split")
        if len(part):
            out[split] = part
    if not out:
        raise TaskMaterializationError(
            "No label rows fell into any MEDS split; check that the labels and the dataset refer to the "
            "same subject ids."
        )
    return out


def materialize_labels(
    external_labels_dir: Path | str,
    patients_dir: Path | str,
    dest: Path | str,
    *,
    include_labels: bool = True,
) -> tuple[Path, dict[str, dict]]:
    """Write ``{split}.parquet`` under ``dest`` for meds-torch-data; return ``(dest, summary)``.

    This is the whole of what the removed ``preprocess_task`` command used to do, now run inline by
    whichever command needs a task. It is cheap — parquet in, parquet out, no model and no tensorization —
    so repeating it per command costs little and removes a shared artifact that commands would otherwise
    have to agree on.

    ``include_labels=False`` writes only ``(subject_id, prediction_time)``. meds-torch-data distinguishes a
    task *index* from task *labels* — ``boolean_value`` is optional — and without it ``batch.boolean_value``
    is simply absent. That is what makes "prediction never reads ground truth" true of the batch a model
    receives, not merely of the file it writes: at inference the labels are never put in front of the model
    at all. Only training passes them.
    """
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)

    loaded = read_labels(external_labels_dir)
    by_split = (
        loaded
        if isinstance(loaded, dict)
        else split_labels(loaded, load_subject_splits(patients_dir))
    )

    for split, df in by_split.items():
        if not include_labels:
            df = df.drop("boolean_value")
        table = LabelSchema.align(df.to_arrow())
        pl.from_arrow(table).write_parquet(dest / f"{split}.parquet")

    summary = summarize_labels(by_split, include_positive_rate=include_labels)
    logger.info(
        "Materialized %d label rows across %d split(s) into %s.",
        sum(s["n"] for s in summary.values()),
        len(summary),
        dest,
    )
    return dest, summary


def summarize_labels(
    by_split: dict[str, pl.DataFrame], *, include_positive_rate: bool = True
) -> dict[str, dict]:
    """Per-split label statistics recorded in the consuming command's manifest.

    The positive rate is omitted wherever the labels themselves are — an inference artifact should not
    carry an aggregate of the ground truth it is about to be scored against.
    """
    summary: dict[str, dict] = {}
    for split, df in by_split.items():
        times = df["prediction_time"]
        stats: dict = {
            "n": len(df),
            "prediction_time": {
                "min": str(times.min()) if len(df) else None,
                "max": str(times.max()) if len(df) else None,
            },
        }
        if include_positive_rate:
            stats["positive_rate"] = round(float(df["boolean_value"].mean()), 6) if len(df) else None
        summary[split] = stats
    return summary


__all__ = [
    "TaskMaterializationError",
    "load_subject_splits",
    "materialize_labels",
    "normalize_label_columns",
    "read_labels",
    "split_labels",
    "summarize_labels",
]
