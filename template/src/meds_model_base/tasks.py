"""Task-definition plumbing: make an ACES task spec available to the ``prediction`` step.

An ACES configuration is *how a task is specified*. It is a required input to any zero-shot ``prediction``
run: without it a task-agnostic model has no definition of *what* to predict on a new task. This module's
sole job is to make that definition available to a model — it does **not** run ACES.

What a model *does* with the task definition is model-specific and out of scope for this template:

- a **zero-shot generative** model generates trajectories and resolves the ACES definition over its *own
  generated futures* (typically via a separate, shared tool — not implemented here);
- **EveryQuery** translates the ACES definition into a native EQ query;
- a **supervised** model was trained for the task and may ignore the definition at predict time.

The template only guarantees the plumbing: ``prediction`` receives ``cfg.task`` (a path to the ACES YAML),
and :func:`load_task_config` is a convenience for models that want the parsed definition.

Dependency-light (ACES config parsing only; no torch, no data access).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    from aces.config import TaskExtractorConfig


def load_task_config(task_path: str | Path, predicates_path: str | Path | None = None) -> TaskExtractorConfig:
    """Parse an ACES task YAML into a ``TaskExtractorConfig`` (predicates / trigger / windows / label).

    This reads only the task *definition* — no data is touched. Zero-shot resolvers and query models use
    it to learn and translate what to predict. ``es-aces`` is imported lazily so models that pass the task
    path straight through (or don't need the parsed form) incur no ACES dependency at import time.
    """
    from aces.config import TaskExtractorConfig

    return TaskExtractorConfig.load(
        config_path=Path(task_path),
        predicates_path=Path(predicates_path) if predicates_path else None,
    )
