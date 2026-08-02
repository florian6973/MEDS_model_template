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

from .schemas import LabelSchema

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


def read_labels(external_labels_dir: Path | str) -> pl.DataFrame:
    """Read a labels directory (or single parquet) into one canonical frame.

    **Accepted layouts.** Any of these, because all of them occur:

    - a single ``*.parquet`` file;
    - a directory of parquet files at any depth, in any arrangement — ``labels.parquet``,
      ``train.parquet``/``tuning.parquet``/``held_out.parquet``, ``train/0.parquet``, ``shard_A.parquet``.

    Every ``*.parquet`` found is a shard, whatever it is named, and they are concatenated without
    interpretation. Directories whose name begins with ``.`` are skipped: ``meds-dev-task`` always writes
    a ``.logs/`` beside its output, and a run's logs are not label data.

    Each shard must carry a subject id, a prediction time and a boolean label under one of the spellings
    :func:`normalize_label_columns` accepts; anything else fails there, naming the offending file and the
    columns it does have.

    **The layout carries no split information this package acts on.** Which split a labelled subject
    belongs to is decided by :func:`tokenized_cohort`, from the artifact the labels will be used against.

    That matters most for the layout which looks like it says otherwise. ``meds-dev-task`` runs ACES per
    input shard into ``{output_dir}/${data._prefix}.parquet``, so against a split-sharded dataset (what
    ``meds-dev-dataset`` produces) it emits ``train/0.parquet``, ``tuning/0.parquet``,
    ``held_out/0.parquet``. Those names are the *input shard* names; they coincide with split names only
    because that dataset happened to be sharded by split, and nothing ties the labels directory to the
    dataset this model preprocessed. Reading a split out of a path would be a second source of truth about
    something the cohort already answers.
    """
    p = Path(external_labels_dir)
    if p.is_file():
        return normalize_label_columns(pl.read_parquet(p), source=str(p))
    if not p.is_dir():
        raise TaskMaterializationError(f"external_labels_dir {p} does not exist.")

    shards = [
        fp
        for fp in sorted(p.rglob("*.parquet"))
        if not any(part.startswith(".") for part in fp.relative_to(p).parts)
    ]
    if not shards:
        raise TaskMaterializationError(
            f"{p} is a directory but contains no label parquet files. Expected a MEDS labels directory: "
            "one or more *.parquet with (subject_id, prediction_time, boolean_value), such as "
            "meds-dev-task's train/0.parquet, tuning/0.parquet, held_out/0.parquet."
        )
    logger.info("Reading %d label shard(s) from %s.", len(shards), p)
    return pl.concat(
        [normalize_label_columns(pl.read_parquet(fp), source=str(fp)) for fp in shards],
        how="vertical_relaxed",
    ).unique(maintain_order=True)


def tokenized_cohort(patients_dir: Path | str) -> dict[str, set[int]]:
    """Subjects actually present in a tensorized cohort, keyed by split.

    Read from ``tokenization/schemas/<split>/*.parquet``, which is where meds-torch-data itself resolves
    split membership from — ``MEDSPytorchDataset`` keeps a shard only when its name starts with
    ``f"{split}/"`` and never opens ``subject_splits.parquet``. Reading the same place is what makes the
    two impossible to disagree.

    A schema file with no leading split component belongs to no split, exactly as meds-torch-data treats
    it; such shards are skipped rather than guessed at.
    """
    schema_dir = Path(patients_dir) / "tokenization" / "schemas"
    if not schema_dir.is_dir():
        raise TaskMaterializationError(
            f"No tokenization schemas under {schema_dir}; {patients_dir} is not a tensorized cohort."
        )
    cohort: dict[str, set[int]] = {}
    for fp in sorted(schema_dir.rglob("*.parquet")):
        shard = fp.relative_to(schema_dir).with_suffix("")
        if len(shard.parts) < 2:
            continue
        subjects = pl.read_parquet(fp, columns=["subject_id"])["subject_id"].to_list()
        cohort.setdefault(shard.parts[0], set()).update(subjects)
    return cohort


def split_labels(labels: pl.DataFrame, cohort: dict[str, set[int]]) -> dict[str, pl.DataFrame]:
    """Partition labels by the split each subject was actually tokenized into.

    The cohort is the authority, not the dataset's ``subject_splits.parquet``, because the cohort is what
    the labels will be used against. A ``preprocess_data`` pipeline that filters subjects —
    ``filter_subjects``, or a ``filter_measurements`` aggressive enough to empty someone's timeline —
    tokenizes them away while their labels survive untouched. Partitioning from the source table would
    keep those labels, and nothing would notice: ``pretrain`` and ``supervised_train`` read training data
    through the schema directories, see only the survivors, and succeed. ``predict`` is the first command
    to compare the two, and it fails with ``CoverageError`` pointing at prediction rather than at the
    preprocessing that caused it, two training runs too late.

    Partitioning from the cohort makes ``n_expected`` correct at the point the labels are written, so the
    coverage guard stops firing because the expectation became true — not because the guard was weakened.
    A subject in no split simply lands in no partition, so the drop needs no separate pass.

    The two cannot disagree about a subject that *is* present: a split-sharded dataset's layout, and
    anything ``reshard_to_split`` produces, are both derived from that same table.
    """
    out: dict[str, pl.DataFrame] = {}
    kept = 0
    for split, subjects in sorted(cohort.items()):
        part = labels.filter(pl.col("subject_id").is_in(sorted(subjects)))
        kept += part.height
        if part.height:
            out[split] = part

    if not out:
        raise TaskMaterializationError(
            "No label row matched any subject in the tensorized cohort. Either these labels are for a "
            "different dataset, or the preprocess_data pipeline filtered the whole cohort away."
        )
    if kept < labels.height:
        dropped = labels.filter(
            ~pl.col("subject_id").is_in(sorted(set().union(*cohort.values())))
        )
        logger.warning(
            "Dropping %d label row(s) for %d subject(s) absent from the tensorized cohort: either "
            "filtered out by the preprocess_data pipeline (its manifest records how many), or not part "
            "of this dataset.",
            labels.height - kept,
            dropped["subject_id"].n_unique(),
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

    by_split = split_labels(read_labels(external_labels_dir), tokenized_cohort(patients_dir))

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
    "materialize_labels",
    "normalize_label_columns",
    "read_labels",
    "split_labels",
    "summarize_labels",
    "tokenized_cohort",
]
