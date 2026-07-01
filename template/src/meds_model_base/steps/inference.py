"""The ``task_agnostic_inference`` step (d): index timepoints → an open per-timepoint output.

Runs the trained model over an index dataframe (via MTD's datamodule, ``TO_END`` sampling) and writes a
``schemas.TaskAgnosticOutputSchema`` parquet keyed on ``(subject_id, prediction_time)`` plus whatever the
model's ``predict_step`` returns (e.g. an ``embedding: list<float32>`` column, or generated tokens). MTD's
dataset ``schema_df`` supplies the aligned keys.

This is task-agnostic: no task definition, no scoring. It is the intermediate a zero-shot ``prediction``
step consumes (generate here; resolve a task over the generations elsewhere).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl
import pyarrow.parquet as pq

from ..schemas import TaskAgnosticOutputSchema, held_out_split
from .base import TaskAgnosticInferenceStep
from .predict import load_trained_module, run_predict_step

if TYPE_CHECKING:  # pragma: no cover - typing only
    from omegaconf import DictConfig

logger = logging.getLogger(__name__)

TASK_AGNOSTIC_OUTPUT_FILENAME = "task_agnostic_output.parquet"


class DefaultTaskAgnosticInferenceStep(TaskAgnosticInferenceStep):
    """Default: write whatever the model's ``predict_step`` emits at each index timepoint.

    A model exposes its outputs by returning a dict of ``[B]`` (scalar column) or ``[B, D]`` (list column)
    tensors from ``predict_step``. Override ``predict_step`` in ``model.py`` to emit, e.g., an ``embedding``
    or generated tokens.
    """

    def run(self, cfg: "DictConfig") -> Path:
        module = load_trained_module(cfg, Path(cfg.model_initialization_dir))
        module.eval()
        split = cfg.get("split") or held_out_split

        keys, outputs = run_predict_step(cfg, module, split)
        result = keys.with_columns(_output_columns(outputs))

        output_dir = Path(cfg.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        table = TaskAgnosticOutputSchema.align(result.to_arrow())
        out_fp = output_dir / TASK_AGNOSTIC_OUTPUT_FILENAME
        pq.write_table(table, out_fp)
        logger.info("Wrote task-agnostic output (%d rows) to %s.", table.num_rows, out_fp)
        return output_dir


def _output_columns(outputs: list[dict]) -> list[pl.Series]:
    """Turn per-batch ``predict_step`` dicts into aligned polars columns (scalars scalar, vectors → lists)."""
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
