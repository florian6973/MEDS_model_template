"""Canonical MEDS schemas + the two template-specific schemas, in one place.

This module is the **single source of truth** for every column name the template touches. It re-exports
the canonical ``meds`` / ``meds_evaluation`` schema classes and path constants (so downstream code never
hardcodes a column name), and defines the two schemas the mandated CLI needs that the standard does not
provide:

- :class:`IndexSchema` — the "index dataframe" consumed by ``task_agnostic_inference`` and (optionally)
  ``prediction``: the two label *keys* ``(subject_id, prediction_time)`` with **no** value column and
  arbitrary extra columns allowed. (``meds.LabelSchema`` is *closed*, so it cannot carry the passthrough
  columns a caller may want on an index.)
- :class:`TaskAgnosticOutputSchema` — the *open* output of ``task_agnostic_inference``: the two keys plus
  whatever model columns the step emits (e.g. an ``embedding`` list column, or zero-shot scores). It is
  deliberately **not** ``LabelSchema`` (which is closed and would reject those columns) and **not**
  ``PredictionSchema`` (a task-agnostic output has no ground-truth ``boolean_value`` to score against).

The prediction output contract is :class:`PredictionSchema` from ``meds_evaluation`` — validate/coerce a
predictions table with :func:`validate_predictions`.
"""
# NOTE: intentionally NO ``from __future__ import annotations`` — flexible_schema reads the live
# ``Required(...)`` descriptor objects off the class annotations at class-definition time; PEP 563
# would stringize them and break schema construction (this mirrors ``meds.schema``).

from typing import ClassVar

import pyarrow as pa

# --- Canonical MEDS schemas (re-exported; do not redefine) --------------------------------------------
from meds import (
    CodeMetadataSchema,
    DataSchema,
    DatasetMetadataSchema,
    LabelSchema,
    SubjectSplitSchema,
    birth_code,
    code_metadata_filepath,
    data_subdirectory,
    dataset_metadata_filepath,
    death_code,
    held_out_split,
    subject_splits_filepath,
    train_split,
    tuning_split,
)

# --- Canonical prediction schema (re-exported) --------------------------------------------------------
from meds_evaluation.schema import PredictionSchema
from flexible_schema import PyArrowSchema, Required

__all__ = [
    # canonical MEDS
    "DataSchema",
    "LabelSchema",
    "SubjectSplitSchema",
    "CodeMetadataSchema",
    "DatasetMetadataSchema",
    "PredictionSchema",
    # constants
    "data_subdirectory",
    "subject_splits_filepath",
    "code_metadata_filepath",
    "dataset_metadata_filepath",
    "train_split",
    "tuning_split",
    "held_out_split",
    "birth_code",
    "death_code",
    "SPLITS",
    # template schemas
    "IndexSchema",
    "TaskAgnosticOutputSchema",
    # helpers
    "validate_predictions",
    "validate_index",
    "make_index",
    "load_index",
]

#: The three canonical MEDS split names, in pipeline order (train → tuning → held_out).
SPLITS: tuple[str, str, str] = (train_split, tuning_split, held_out_split)


class IndexSchema(PyArrowSchema):
    """An "index dataframe": the two MEDS label keys, with extra passthrough columns allowed.

    This is what ``task_agnostic_inference`` (and, before ACES extraction, ``prediction``) consumes to say
    *at which patient timepoints* to make inference. The ``prediction_time`` is an inclusive as-of cutoff:
    a model may use each subject's data up to and including that time.

    Unlike :class:`meds.LabelSchema`, this schema is **open** — a caller may attach arbitrary extra
    columns (e.g. a cohort id) and they will be preserved.

    Examples:
        >>> from datetime import datetime
        >>> tbl = pa.Table.from_pylist([
        ...     {"subject_id": 1, "prediction_time": datetime(2020, 1, 1)},
        ...     {"subject_id": 2, "prediction_time": datetime(2020, 6, 1)},
        ... ])
        >>> IndexSchema.validate(IndexSchema.align(tbl))   # no error
        >>> # int32 subject ids are coerced up to int64 by align():
        >>> import pyarrow as pa
        >>> tbl2 = tbl.set_column(0, "subject_id", tbl["subject_id"].cast(pa.int32()))
        >>> IndexSchema.align(tbl2).schema.field("subject_id").type
        DataType(int64)
    """

    allow_extra_columns: ClassVar[bool] = True

    subject_id: Required(pa.int64(), nullable=False)  # noqa: F821 - flexible_schema annotation
    prediction_time: Required(pa.timestamp("us"), nullable=False)  # noqa: F821


class TaskAgnosticOutputSchema(PyArrowSchema):
    """The output schema of ``task_agnostic_inference``: the two keys + arbitrary model columns.

    An **open** schema keyed on ``(subject_id, prediction_time)``. A model attaches whatever it produces
    at each timepoint — e.g. an ``embedding: list<float32>`` column, generated-trajectory tokens, or
    zero-shot scores. It is intentionally not scorable by ``meds-evaluation`` (there is no ground-truth
    ``boolean_value``); scoring is the job of the ``prediction`` step.

    Examples:
        >>> from datetime import datetime
        >>> tbl = pa.Table.from_pylist([
        ...     {"subject_id": 1, "prediction_time": datetime(2020, 1, 1),
        ...      "embedding": [0.1, 0.2, 0.3]},
        ... ])
        >>> out = TaskAgnosticOutputSchema.align(tbl)
        >>> out.column_names
        ['subject_id', 'prediction_time', 'embedding']
    """

    allow_extra_columns: ClassVar[bool] = True

    subject_id: Required(pa.int64(), nullable=False)  # noqa: F821
    prediction_time: Required(pa.timestamp("us"), nullable=False)  # noqa: F821


def validate_predictions(table: pa.Table) -> pa.Table:
    """Validate + coerce a predictions table to :class:`PredictionSchema`.

    This is the load-bearing output check for the ``prediction`` step: it guarantees the emitted parquet
    is exactly what ``meds-evaluation`` expects (correct key columns, at least one non-all-null predicted
    column, ``float32`` probabilities, canonical column order). Raises
    ``flexible_schema.SchemaValidationError`` / ``TableValidationError`` on violation.

    Examples:
        >>> from datetime import datetime
        >>> tbl = pa.Table.from_pylist([
        ...     {"subject_id": 1, "prediction_time": datetime(2020, 1, 1),
        ...      "boolean_value": True, "predicted_boolean_probability": 0.9},
        ...     {"subject_id": 2, "prediction_time": datetime(2020, 1, 1),
        ...      "boolean_value": False, "predicted_boolean_probability": 0.1},
        ... ])
        >>> out = validate_predictions(tbl)
        >>> out.schema.field("predicted_boolean_probability").type
        DataType(float)
    """
    return PredictionSchema.align(table)


def validate_index(table: pa.Table) -> pa.Table:
    """Validate + coerce an index table to :class:`IndexSchema` (keys required, extras preserved).

    Examples:
        >>> from datetime import datetime
        >>> tbl = pa.Table.from_pylist([{"subject_id": 1, "prediction_time": datetime(2020, 1, 1)}])
        >>> validate_index(tbl).column_names
        ['subject_id', 'prediction_time']
    """
    return IndexSchema.align(table)


def make_index(subject_ids: list[int], prediction_times: list) -> pa.Table:
    """Build a minimal :class:`IndexSchema` table from parallel lists.

    Examples:
        >>> from datetime import datetime
        >>> t = make_index([1, 2], [datetime(2020, 1, 1), datetime(2020, 2, 1)])
        >>> t.column_names
        ['subject_id', 'prediction_time']
        >>> t.num_rows
        2
    """
    tbl = pa.Table.from_pydict({"subject_id": subject_ids, "prediction_time": prediction_times})
    return IndexSchema.align(tbl)


def load_index(index_path, split: str | None = None) -> pa.Table:
    """Load an index of prediction timepoints, reading **only** ``subject_id`` + ``prediction_time``.

    Accepts a single parquet file, a directory of ``{split}.parquet`` files, or a directory of shards. If
    the source is a ``meds.LabelSchema`` file (e.g. a MEDS-DEV ``labels_dir``) that also carries a
    ``boolean_value``, that column is **dropped here** — this system never reads ground-truth labels at
    prediction time; it only needs *where* to predict.

    Args:
        index_path: file or directory of index/label parquets.
        split: if the directory holds ``{split}.parquet`` files, restrict to this split.

    Returns:
        An :class:`IndexSchema`-aligned table with exactly ``subject_id, prediction_time``.
    """
    import polars as pl

    from pathlib import Path

    p = Path(index_path)
    if p.is_file():
        fps = [p]
    elif split is not None and (p / f"{split}.parquet").exists():
        fps = [p / f"{split}.parquet"]
    else:
        fps = sorted(p.rglob("*.parquet"))
    if not fps:
        raise FileNotFoundError(f"No index parquet files found under {index_path!r} (split={split!r}).")

    df = pl.concat(
        [pl.read_parquet(fp).select(["subject_id", "prediction_time"]) for fp in fps],
        how="vertical_relaxed",
    ).unique(maintain_order=True)
    return IndexSchema.align(df.to_arrow())
