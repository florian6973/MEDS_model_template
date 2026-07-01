"""The step contract (``StepName``, ABCs) and the default step implementations.

Importing the default step classes pulls in torch / lightning / meds-torch-data; import
:mod:`meds_model_base.steps.base` directly if you only need the (dependency-light) contract.
"""

from __future__ import annotations

from .base import (
    MEDSModelStep,
    PredictionStep,
    PreprocessStep,
    StepName,
    SupervisedTrainStep,
    TaskAgnosticInferenceStep,
    TrainStep,
    UnsupervisedTrainStep,
)

__all__ = [
    # contract
    "StepName",
    "MEDSModelStep",
    "PreprocessStep",
    "TrainStep",
    "UnsupervisedTrainStep",
    "SupervisedTrainStep",
    "TaskAgnosticInferenceStep",
    "PredictionStep",
    # default implementations
    "DefaultPreprocessStep",
    "DefaultUnsupervisedTrainStep",
    "DefaultSupervisedTrainStep",
    "DefaultTaskAgnosticInferenceStep",
    "SupervisedPredictionStep",
    "ZeroShotPredictionStep",
]


def __getattr__(name: str):
    """Lazily import the (torch-heavy) default step classes so ``.base`` stays importable alone."""
    _defaults = {
        "DefaultPreprocessStep": "preprocess",
        "DefaultUnsupervisedTrainStep": "train",
        "DefaultSupervisedTrainStep": "train",
        "DefaultTaskAgnosticInferenceStep": "inference",
        "SupervisedPredictionStep": "predict",
        "ZeroShotPredictionStep": "predict",
    }
    if name in _defaults:
        import importlib

        module = importlib.import_module(f".{_defaults[name]}", __name__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
