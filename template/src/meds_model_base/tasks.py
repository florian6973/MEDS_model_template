"""Task materialization: turn an ``external_task_file`` into split label parquets.

``preprocess_task`` accepts either form of external task file:

- an **ACES task YAML**, which is extracted against the raw MEDS dataset that ``preprocess_data`` consumed
  (its location is recovered from the ``patients/`` manifest, so the caller need not repeat it); or
- an **already-materialized** parquet (or directory of ``{split}.parquet``) carrying at least
  ``subject_id``, ``prediction_time`` and ``boolean_value``.

Either way the result is written as one parquet per split, which is the layout meds-torch-data expects for
``task_labels_dir`` and therefore what both ``supervised_train`` and ``predict`` consume. Which of the two
paths was taken is recorded in the task manifest as ``materialization``.

Dependency-light: polars + pyarrow. ``es-aces`` is imported lazily, so models that only ever pass
pre-extracted labels never need it installed.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl

from .schemas import SPLITS, LabelSchema, subject_splits_filepath

if TYPE_CHECKING:  # pragma: no cover - typing only
    from aces.config import TaskExtractorConfig

logger = logging.getLogger(__name__)

#: Column aliases accepted from an ACES result or a hand-built label table, in priority order.
_COLUMN_ALIASES = {
    "subject_id": ("subject_id", "patient_id"),
    "prediction_time": ("prediction_time", "index_timestamp", "trigger"),
    "boolean_value": ("boolean_value", "label"),
}


class TaskMaterializationError(RuntimeError):
    """Raised when a task file cannot be turned into split label parquets."""


def load_task_config(task_path: str | Path, predicates_path: str | Path | None = None) -> TaskExtractorConfig:
    """Parse an ACES task YAML into a ``TaskExtractorConfig`` (predicates / trigger / windows / label).

    Reads only the task *definition* — no data is touched. Zero-shot and query models use this to learn what
    to predict; ``es-aces`` is imported lazily so models that never parse a task avoid the dependency.
    """
    from aces.config import TaskExtractorConfig

    return TaskExtractorConfig.load(
        config_path=Path(task_path),
        predicates_path=Path(predicates_path) if predicates_path else None,
    )


def is_aces_task_file(task_file: Path | str) -> bool:
    """Whether ``task_file`` is an ACES YAML (as opposed to materialized labels).

    Examples:
        >>> is_aces_task_file("tasks/mortality.yaml")
        True
        >>> is_aces_task_file("labels/mortality.parquet")
        False
    """
    return Path(task_file).suffix.lower() in {".yaml", ".yml"}


def normalize_label_columns(df: pl.DataFrame, *, source: str) -> pl.DataFrame:
    """Rename a label table's columns to the canonical MEDS names and drop the rest.

    ACES output and hand-built label files vary in what they call the trigger time and the label, and the
    exact names have changed across ``es-aces`` releases. Rather than pin one spelling, accept the known
    aliases and fail with the observed columns listed when none matches.

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


def extract_with_aces(
    task_yaml: Path | str,
    meds_dir: Path | str,
    predicates_path: Path | str | None = None,
) -> pl.DataFrame:
    """Run ACES over the raw MEDS dataset to materialize ``subject_id, prediction_time, boolean_value``.

    .. note::
       This calls the documented ``es-aces`` programmatic API (``get_predicates_df`` → ``query``). That API
       has moved across releases, so the result is normalized through :func:`normalize_label_columns`
       rather than assuming column names, and any failure is re-raised with the ACES error attached. If
       your ``es-aces`` version is incompatible, materialize the labels yourself and pass the parquet as
       ``external_task_file`` — that path has no ACES dependency at all.
    """
    from omegaconf import DictConfig

    try:
        from aces import predicates as aces_predicates
        from aces import query as aces_query
    except ImportError as e:  # pragma: no cover - depends on the install
        raise TaskMaterializationError(
            "Extracting an ACES task YAML requires `es-aces` (pip install 'es-aces>=0.7.3'), or supply "
            "pre-materialized labels as external_task_file instead."
        ) from e

    task_cfg = load_task_config(task_yaml, predicates_path)
    data_config = DictConfig({"path": str(Path(meds_dir) / "data" / "**/*.parquet"), "standard": "meds"})

    try:
        predicates_df = aces_predicates.get_predicates_df(task_cfg, data_config)
        result = aces_query.query(task_cfg, predicates_df)
    except Exception as e:
        raise TaskMaterializationError(
            f"ACES extraction of {task_yaml} against {meds_dir} failed: {e}"
        ) from e

    df = result if isinstance(result, pl.DataFrame) else pl.DataFrame(result)
    return normalize_label_columns(df, source=f"ACES output for {Path(task_yaml).name}")


def load_subject_splits(meds_dir: Path | str) -> pl.DataFrame:
    """Load ``metadata/subject_splits.parquet`` from a raw MEDS root."""
    fp = Path(meds_dir) / subject_splits_filepath
    if not fp.is_file():
        raise TaskMaterializationError(
            f"No subject splits at {fp}. A canonical MEDS dataset must define train/tuning/held_out splits "
            "before a task can be materialized against it."
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
            "No label rows fell into any MEDS split; check that the task and the dataset refer to the same "
            "subject ids."
        )
    return out


def read_materialized_labels(task_file: Path | str) -> pl.DataFrame | dict[str, pl.DataFrame]:
    """Read pre-materialized labels: a single parquet, or a directory of ``{split}.parquet`` files.

    Returns a dict keyed by split when the input is already split by file (in which case no split table is
    needed), else a single frame to be partitioned by :func:`split_labels`.
    """
    p = Path(task_file)
    if p.is_dir():
        by_split = {
            split: pl.read_parquet(p / f"{split}.parquet")
            for split in SPLITS
            if (p / f"{split}.parquet").is_file()
        }
        if not by_split:
            raise TaskMaterializationError(
                f"{p} is a directory but contains no {{{','.join(SPLITS)}}}.parquet files."
            )
        return {
            split: normalize_label_columns(df, source=f"{p / f'{split}.parquet'}")
            for split, df in by_split.items()
        }
    if not p.is_file():
        raise TaskMaterializationError(f"external_task_file {p} does not exist.")
    return normalize_label_columns(pl.read_parquet(p), source=str(p))


def write_task_splits(by_split: dict[str, pl.DataFrame], dest: Path) -> dict[str, int]:
    """Write ``{split}.parquet`` files, validating each against ``meds.LabelSchema``.

    Returns row counts per split (recorded in the task manifest).
    """
    counts: dict[str, int] = {}
    for split, df in by_split.items():
        table = LabelSchema.align(df.to_arrow())
        pl.from_arrow(table).write_parquet(dest / f"{split}.parquet")
        counts[split] = table.num_rows
    return counts


def summarize_labels(by_split: dict[str, pl.DataFrame]) -> dict[str, dict]:
    """Per-split label statistics for the task manifest (count, positive rate, prediction_time range)."""
    summary: dict[str, dict] = {}
    for split, df in by_split.items():
        times = df["prediction_time"]
        summary[split] = {
            "n": len(df),
            "positive_rate": round(float(df["boolean_value"].mean()), 6) if len(df) else None,
            "prediction_time": {
                "min": str(times.min()) if len(df) else None,
                "max": str(times.max()) if len(df) else None,
            },
        }
    return summary
