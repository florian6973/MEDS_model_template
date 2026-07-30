"""The command contract (``CommandName``, ABCs, arbitration) and the default implementations.

Importing a default implementation pulls in torch / lightning / meds-torch-data. Import
:mod:`meds_model_base.commands.base` directly when you only need the contract — that is what keeps
``meds-model commands`` and ``--help`` fast, and what the CLI smoke tests exercise.
"""

from __future__ import annotations

from .base import (
    CommandName,
    InferCommand,
    MEDSModelCommand,
    PredictCommand,
    PreprocessDataCommand,
    PreprocessTaskCommand,
    PretrainCommand,
    SourceArbitrationError,
    SupervisedTrainCommand,
    arbitrate_sources,
)

__all__ = [
    # contract
    "CommandName",
    "MEDSModelCommand",
    "PreprocessDataCommand",
    "PreprocessTaskCommand",
    "PretrainCommand",
    "InferCommand",
    "SupervisedTrainCommand",
    "PredictCommand",
    "SourceArbitrationError",
    "arbitrate_sources",
    # default implementations
    "DefaultPreprocessDataCommand",
    "DefaultPreprocessTaskCommand",
    "DefaultPretrainCommand",
    "DefaultInferCommand",
    "DefaultSupervisedTrainCommand",
    "ProbeTrainCommand",
    "SupervisedPredictCommand",
    "ProbePredictCommand",
    "MaterializedPredictCommand",
    "PackagedPredictCommand",
    "ZeroShotPredictCommand",
]

#: Default implementation → the module it lives in (imported lazily; see ``__getattr__``).
_DEFAULTS = {
    "DefaultPreprocessDataCommand": "preprocess_data",
    "DefaultPreprocessTaskCommand": "preprocess_task",
    "DefaultPretrainCommand": "train",
    "DefaultSupervisedTrainCommand": "train",
    "ProbeTrainCommand": "train",
    "DefaultInferCommand": "infer",
    "SupervisedPredictCommand": "predict",
    "ProbePredictCommand": "predict",
    "MaterializedPredictCommand": "predict",
    "PackagedPredictCommand": "predict",
    "ZeroShotPredictCommand": "predict",
}


def __getattr__(name: str):
    """Lazily import the (torch-heavy) default command classes so ``.base`` stays importable alone."""
    if name in _DEFAULTS:
        import importlib

        module = importlib.import_module(f".{_DEFAULTS[name]}", __name__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
