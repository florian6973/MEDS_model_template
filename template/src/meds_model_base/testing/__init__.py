"""Reusable test helpers: synthetic designed-signal datasets + learnability assertions.

These back the *model-specific synthetic-data property test* (tier 3) — the flagship correctness check
generalizing MEDS-EIC-AR's grammar test. The contributor builds a synthetic MEDS dataset with a *designed*
signal and asserts the model learns it, always paired with a negative control so the test can't pass
vacuously.

- :func:`build_signal_dataset` — a classifier signal: a marker code deterministically predicts the label.
- :func:`build_pattern_dataset` — a generative signal: a fixed repeating code pattern.

Both give every subject the static (baseline) measurements in :data:`STATIC_CODES`, drawn independently of
the labels. That is not decoration: a cohort with no static measurements at all cannot be collated by
meds-torch-data under ``static_inclusion_mode: INCLUDE`` — it raises ``Cannot infer dtype from empty
values`` before the model is called — so without them this suite could not exercise any model that reads
baseline data.
- :func:`binary_auroc`, :func:`assert_learns_signal` — learnability assertions (+ negative control).
- :func:`skip_if_stub` — skip a conformance test while ``model.py`` is still the generated stub.
- :func:`run_chain`, :func:`build_workspace` — drive a model's DAG from its ``COMMANDS`` registry.
"""

from .harness import build_workspace, run_chain, run_cli, supported_sources
from .property import assert_learns_signal, binary_auroc
from .stub import is_stub, skip_if_stub
from .synthetic import (
    SIGNAL_CODE,
    STATIC_CODES,
    STATIC_GROUPS,
    STATIC_NUMERIC,
    build_pattern_dataset,
    build_signal_dataset,
)

__all__ = [
    "SIGNAL_CODE",
    "STATIC_CODES",
    "STATIC_GROUPS",
    "STATIC_NUMERIC",
    "assert_learns_signal",
    "binary_auroc",
    "build_workspace",
    "run_chain",
    "run_cli",
    "supported_sources",
    "is_stub",
    "skip_if_stub",
    "build_pattern_dataset",
    "build_signal_dataset",
]
