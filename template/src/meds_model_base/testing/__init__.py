"""Reusable test helpers: synthetic designed-signal datasets + learnability assertions.

These back the *model-specific synthetic-data property test* (tier 3) — the flagship correctness check
generalizing MEDS-EIC-AR's grammar test. The contributor builds a synthetic MEDS dataset with a *designed*
signal and asserts the model learns it, always paired with a negative control so the test can't pass
vacuously.

- :func:`build_signal_dataset` — a classifier signal: a marker code deterministically predicts the label.
- :func:`build_pattern_dataset` — a generative signal: a fixed repeating code pattern.
- :func:`binary_auroc`, :func:`assert_learns_signal` — learnability assertions (+ negative control).
- :func:`skip_if_stub` — skip a conformance test while ``model.py`` is still the generated stub.
"""

from .property import assert_learns_signal, binary_auroc
from .stub import is_stub, skip_if_stub
from .synthetic import SIGNAL_CODE, build_pattern_dataset, build_signal_dataset

__all__ = [
    "SIGNAL_CODE",
    "assert_learns_signal",
    "binary_auroc",
    "is_stub",
    "skip_if_stub",
    "build_pattern_dataset",
    "build_signal_dataset",
]
