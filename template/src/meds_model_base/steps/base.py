"""The mandated step contract: ``StepName`` and the ``MEDSModelStep`` ABC hierarchy.

Every model built from this template implements a subset of five steps. Each step is a *config-driven,
disk-in / disk-out* unit with a single public method, :meth:`MEDSModelStep.run`, that takes a resolved
Hydra config and returns its primary output directory. Per-step ABCs pin the IO contract and declare the
**override hooks** (``build_module``, ``predict``, ...) that concrete implementations fill.

This module is deliberately dependency-light (no torch / lightning import) so that ``meds-model steps``,
``--help``, and the dispatcher can introspect the contract cheaply. Concrete default implementations that
need the heavy stack live in the sibling modules (``preprocess``, ``train``, ``inference``, ``predict``).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import StrEnum
from typing import TYPE_CHECKING, ClassVar

if TYPE_CHECKING:  # pragma: no cover - typing only
    from pathlib import Path

    import lightning.pytorch as pl
    import polars as pl_df
    from omegaconf import DictConfig


class StepName(StrEnum):
    """The five mandated CLI steps, in canonical pipeline order.

    Examples:
        >>> StepName.prediction
        <StepName.prediction: 'prediction'>
        >>> StepName("preprocess") is StepName.preprocess
        True
        >>> [s.value for s in StepName]
        ['preprocess', 'unsupervised_train', 'supervised_train', 'task_agnostic_inference', 'prediction']
    """

    preprocess = "preprocess"
    unsupervised_train = "unsupervised_train"
    supervised_train = "supervised_train"
    task_agnostic_inference = "task_agnostic_inference"
    prediction = "prediction"


class MEDSModelStep(ABC):
    """Abstract base for every step: a config-driven unit that reads and writes disk.

    Subclasses set two class variables and implement :meth:`run`:

    - ``name`` — the :class:`StepName` this step provides.
    - ``config_name`` — the Hydra *root* config name (without extension) the dispatcher loads for it,
      e.g. ``"_prediction"``. Roots live in ``src/<model_slug>/configs/``.
    """

    name: ClassVar[StepName]
    config_name: ClassVar[str]

    @abstractmethod
    def run(self, cfg: DictConfig) -> Path:
        """Execute the step against a resolved Hydra ``cfg``; return the primary output directory."""
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name.value!r}, config_name={self.config_name!r})"


class PreprocessStep(MEDSModelStep):
    """(a) Raw MEDS → model-ready artifacts.

    - **input**  ``cfg.input_dir`` = ``$MEDS_ROOT`` (``meds.DataSchema`` shards + ``metadata/*``).
    - **output** ``cfg.output_dir`` = a model-ready directory (default: a meds-torch-data tensorized
      cohort). Consumed only by this model's later steps, so its internal layout is model-defined.
    """

    name: ClassVar[StepName] = StepName.preprocess
    config_name: ClassVar[str] = "_preprocess"


class TrainStep(MEDSModelStep):
    """Shared base for the two training steps.

    The override hook is :meth:`build_module`, which constructs the ``LightningModule`` to fit. The default
    implementations (:mod:`meds_model_base.steps.train`) handle datamodule construction, the Trainer, the
    checkpoint/resume conventions, and writing ``best_model.ckpt`` + ``resolved_config.yaml``.
    """

    @abstractmethod
    def build_module(
        self,
        cfg: DictConfig,
        datamodule: pl.LightningDataModule,
        pretrained_dir: Path | None = None,
    ) -> pl.LightningModule:
        """Build the ``LightningModule`` to train.

        Args:
            cfg: the resolved step config.
            datamodule: the constructed MTD-backed datamodule (exposes ``.config.vocab_size`` etc.).
            pretrained_dir: for fine-tuning, the ``model_initialization_dir`` of a prior run (else None).
        """
        raise NotImplementedError


class UnsupervisedTrainStep(TrainStep):
    """(b) Self-supervised pretraining.

    - **input**  a preprocessed dir; splits resolved via ``meds.SubjectSplitSchema`` (train/tuning).
    - **output** a ``model_initialization_dir`` holding ``best_model.ckpt``, ``config.yaml``,
      ``resolved_config.yaml``, ``environment.txt``.
    """

    name: ClassVar[StepName] = StepName.unsupervised_train
    config_name: ClassVar[str] = "_unsupervised_train"


class SupervisedTrainStep(TrainStep):
    """(c) Supervised (fine-)tuning.

    - **input**  preprocessed dir + ``cfg.labels_dir`` (``meds.LabelSchema``) + optional
      ``cfg.model_initialization_dir`` (a pretrained encoder to fine-tune).
    - **output** a fine-tuned ``model_initialization_dir``.
    """

    name: ClassVar[StepName] = StepName.supervised_train
    config_name: ClassVar[str] = "_supervised_train"


class TaskAgnosticInferenceStep(MEDSModelStep):
    """(d) Inference at caller-specified timepoints.

    - **input**  preprocessed dir + ``cfg.model_initialization_dir`` + an **index dataframe**
      (``schemas.IndexSchema``: ``subject_id, prediction_time``).
    - **output** a ``schemas.TaskAgnosticOutputSchema`` parquet keyed on ``(subject_id, prediction_time)``
      (e.g. an ``embedding`` column, or zero-shot scores).

    The per-timepoint computation is delegated to the trained ``LightningModule``'s ``predict_step``; the
    default step (:mod:`meds_model_base.steps.inference`) orchestrates the streamed, DDP-safe write.
    """

    name: ClassVar[StepName] = StepName.task_agnostic_inference
    config_name: ClassVar[str] = "_task_agnostic_inference"


class PredictionStep(MEDSModelStep):
    """(e) Task-specific predicted probabilities.

    - **input**  preprocessed dir + ``cfg.model_initialization_dir`` + an **index dataframe**
      (``cfg.index``: ``subject_id, prediction_time`` — *only these two columns are read*; a
      ``meds.LabelSchema`` file such as MEDS-DEV's ``labels_dir`` is accepted, but any
      ``boolean_value`` it carries is ignored) + an **optional ACES task YAML** (``cfg.task``, default
      ``null``) that *specifies the task*.
    - **output** a single ``predictions.parquet`` conforming to ``meds_evaluation.PredictionSchema``:
      ``subject_id, prediction_time, predicted_boolean_probability`` (no ``boolean_value``).

    **This step never obtains ground-truth / test-set labels.** The model repo ends at producing predicted
    probabilities; scoring is a separate, shared tool. The ``task`` definition is a required input only for
    zero-shot / query models (which resolve it over the *model's own outputs* — e.g. generated
    trajectories — typically via a separate tool, not implemented here, or by translating it into a native
    query). **Supervised / fine-tuned models ignore ``cfg.task`` entirely**, which is the common case.

    The override hook is :meth:`predict`. The concrete ``run`` (in :mod:`meds_model_base.steps.predict`)
    calls ``predict``, validates with ``PredictionSchema.align`` and writes the parquet — that contract is
    fixed so every model's predictions are compatible with the shared evaluation tool. The *index* of
    timepoints is supplied to the model through the datamodule (``cfg.datamodule.config.task_labels_dir`` =
    ``cfg.index``); meds-torch-data handles the timestep alignment and exposes the ordered keys via its
    dataset ``schema_df``, so models don't hand-align outputs.
    """

    name: ClassVar[StepName] = StepName.prediction
    config_name: ClassVar[str] = "_prediction"

    @abstractmethod
    def predict(self, cfg: DictConfig) -> pl_df.DataFrame:
        """Produce predicted probabilities at the index timepoints (from ``cfg.index``, via the datamodule).

        Returns:
            A dataframe with ``subject_id``, ``prediction_time`` and ``predicted_boolean_probability``
            (optionally ``predicted_boolean_value``). Validated by the caller against ``PredictionSchema``.
        """
        raise NotImplementedError
