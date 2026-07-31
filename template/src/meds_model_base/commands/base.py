"""The command contract: ``CommandName``, the ``MEDSModelCommand`` ABC hierarchy, and source arbitration.

Five commands form a DAG, not a fixed pipeline. Each is a config-driven, disk-in / disk-out unit with a
single public entry point, :meth:`MEDSModelCommand.__call__`, which validates its arguments and then
delegates to :meth:`~MEDSModelCommand.run`.

Naming follows the interface spec:

- ``external_*`` — an artifact supplied from outside the model pipeline;
- ``*_dir`` — an independent artifact root;
- ``*_subdir`` — a component inside the shared ``data_dir``.

Several commands accept more than one *alternative* source (a pretrained model, a supervised model, a set
of inference artifacts). Exactly one is valid: there is no precedence order, because a caller supplying two
sources has made a mistake and silently picking one produces a plausible, wrong answer. Arbitration runs in
:meth:`MEDSModelCommand.validate` — before dispatch — so no implementation can bypass it.

This module is deliberately dependency-light (no torch, no lightning, no omegaconf at runtime) so that
``meds-model commands``, ``--help``, and the dispatcher can introspect the contract cheaply.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import StrEnum
from typing import TYPE_CHECKING, Any, ClassVar

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Mapping, Sequence
    from pathlib import Path

    import lightning.pytorch as pl
    import polars as pl_df
    from omegaconf import DictConfig


class CommandName(StrEnum):
    """The five standard commands, in rough dependency order.

    Examples:
        >>> CommandName.predict
        <CommandName.predict: 'predict'>
        >>> CommandName("pretrain") is CommandName.pretrain
        True
        >>> [c.value for c in CommandName]
        ['preprocess_data', 'pretrain', 'infer', 'supervised_train', 'predict']
    """

    preprocess_data = "preprocess_data"
    pretrain = "pretrain"
    infer = "infer"
    supervised_train = "supervised_train"
    predict = "predict"


class SourceArbitrationError(ValueError):
    """Raised when a command receives an ambiguous, unsupported, or missing set of input sources."""


def arbitrate_sources(
    cfg: Mapping[str, Any],
    sources: Sequence[str],
    *,
    supported: frozenset[str],
    command: str,
    packaged_model: str | None = None,
    require_source: bool = False,
) -> tuple[str | None, Any]:
    """Pick the single input source a command should use, or fail with an actionable message.

    Args:
        cfg: the resolved config (anything supporting ``.get``).
        sources: every alternative source parameter this command's interface defines.
        supported: the subset this implementation actually handles.
        command: command name, for error messages.
        packaged_model: identifier of a model shipped with the repository, if any. Only when this is set may
            a command that ``require_source`` run with no source at all.
        require_source: whether omitting all sources is an error (true for ``predict``, false for
            ``supervised_train``, where no source means "train from scratch").

    Returns:
        ``(role, value)`` for the chosen source, or ``(None, None)`` when none was supplied.

    Raises:
        SourceArbitrationError: on more than one source, an unsupported source, or a missing required one.

    Examples:
        >>> supported = frozenset({"input_supervised_model_dir"})
        >>> arbitrate_sources(
        ...     {"input_supervised_model_dir": "/runs/sup", "input_inference_subdir": None},
        ...     ["input_supervised_model_dir", "input_inference_subdir"],
        ...     supported=supported, command="predict", require_source=True,
        ... )
        ('input_supervised_model_dir', '/runs/sup')

        Two sources is always an error, never a precedence decision:

        >>> arbitrate_sources(
        ...     {"input_supervised_model_dir": "/runs/sup", "input_inference_subdir": "inference/emb"},
        ...     ["input_supervised_model_dir", "input_inference_subdir"],
        ...     supported=supported, command="predict", require_source=True,
        ... )
        Traceback (most recent call last):
            ...
        meds_model_base.commands.base.SourceArbitrationError: predict received more than one input source...

        A source the implementation does not handle is rejected rather than ignored:

        >>> arbitrate_sources(
        ...     {"input_inference_subdir": "inference/emb"},
        ...     ["input_supervised_model_dir", "input_inference_subdir"],
        ...     supported=supported, command="predict", require_source=True,
        ... )
        Traceback (most recent call last):
            ...
        meds_model_base.commands.base.SourceArbitrationError: predict does not support...

        With nothing supplied and no packaged model, a command that requires a source fails:

        >>> arbitrate_sources({}, ["input_supervised_model_dir"], supported=supported,
        ...                   command="predict", require_source=True)
        Traceback (most recent call last):
            ...
        meds_model_base.commands.base.SourceArbitrationError: predict requires one of...

        For ``supervised_train``, no source simply means training from scratch:

        >>> arbitrate_sources({}, ["input_pretrained_model_dir"],
        ...                   supported=frozenset({"input_pretrained_model_dir"}),
        ...                   command="supervised_train")
        (None, None)
    """
    provided = [(role, cfg.get(role)) for role in sources]
    provided = [(role, value) for role, value in provided if value is not None]

    if len(provided) > 1:
        names = ", ".join(role for role, _ in provided)
        raise SourceArbitrationError(
            f"{command} received more than one input source ({names}). These are alternatives, not layers; "
            "supply exactly one."
        )

    if provided:
        role, value = provided[0]
        if role not in supported:
            allowed = ", ".join(sorted(supported)) or "(none)"
            raise SourceArbitrationError(
                f"{command} does not support {role!r} for this model. Supported sources: {allowed}."
            )
        return role, value

    if require_source and packaged_model is None:
        allowed = ", ".join(sorted(supported)) or "(none)"
        raise SourceArbitrationError(
            f"{command} requires one of: {allowed}. (A model that ships its own weights may run with no "
            "source, but this implementation does not declare a packaged model.)"
        )
    return None, None


class MEDSModelCommand(ABC):
    """Abstract base for every command: validate arguments, then read and write disk.

    Subclasses set the class variables below and implement :meth:`run`:

    - ``name`` — the :class:`CommandName` this command provides.
    - ``config_name`` — the Hydra root config it loads, e.g. ``"predict"``.
    - ``sources`` — the alternative input-source parameters the interface defines for it.
    - ``supported_sources`` — the subset this implementation handles. Narrow it in a subclass to make an
      unsupported source a loud error instead of a silently ignored argument.
    - ``packaged_model`` — set when the repository ships its own weights (PFN-style), which is the only way
      a source-requiring command may run with no source.

    Examples:
        >>> class Dummy(MEDSModelCommand):
        ...     name = CommandName.pretrain
        ...     config_name = "pretrain"
        ...     def run(self, cfg): return None
        >>> Dummy()
        Dummy(name='pretrain', config_name='pretrain')
    """

    name: ClassVar[CommandName]
    config_name: ClassVar[str]

    sources: ClassVar[tuple[str, ...]] = ()
    supported_sources: ClassVar[frozenset[str] | None] = None
    packaged_model: ClassVar[str | None] = None
    require_source: ClassVar[bool] = False

    def __call__(self, cfg: DictConfig) -> Path:
        """Validate ``cfg`` and execute the command; return the primary output directory.

        The dispatcher always invokes a command this way, so arbitration cannot be skipped. The arbitrated
        result is cached on :attr:`source` for ``run`` to consume and record.
        """
        self._source = self.validate(cfg)
        return self.run(cfg)

    @property
    def source(self) -> tuple[str | None, Any]:
        """The ``(role, value)`` chosen by :meth:`validate`; ``(None, None)`` when there is no source."""
        return getattr(self, "_source", (None, None))

    def validate(self, cfg: DictConfig) -> tuple[str | None, Any]:
        """Arbitrate the alternative input sources. Override to add command-specific checks.

        Returns the ``(role, value)`` chosen, so ``run`` can record it in the output manifest.
        """
        if not self.sources:
            return None, None
        supported = self.supported_sources
        if supported is None:
            supported = frozenset(self.sources)
        return arbitrate_sources(
            cfg,
            self.sources,
            supported=supported,
            command=self.name.value,
            packaged_model=self.packaged_model,
            require_source=self.require_source,
        )

    @abstractmethod
    def run(self, cfg: DictConfig) -> Path:
        """Execute the command against a resolved Hydra ``cfg``; return the primary output directory."""
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name.value!r}, config_name={self.config_name!r})"


class PreprocessDataCommand(MEDSModelCommand):
    """Convert an external MEDS dataset into this model's patient representation.

    - **input**  ``cfg.external_meds_dir`` — canonical MEDS with existing subject splits.
    - **output** ``cfg.output_data_dir/patients/`` plus its manifest. Creates the ``data_dir`` workspace.

    ``patients/`` is immutable afterwards; only ``infer`` appends a sibling subdirectory. The subject
    split table is copied in here too, so later commands never need the raw dataset again.
    """

    name: ClassVar[CommandName] = CommandName.preprocess_data
    config_name: ClassVar[str] = "preprocess_data"


class PretrainCommand(MEDSModelCommand):
    """Train a foundation model from patient data.

    - **input**  ``cfg.input_data_dir`` only; the interface passes no task to pretraining.
    - **output** ``cfg.output_pretrained_model_dir``.

    Model-specific target construction — EveryQuery's query generation, MOTOR's time-to-event bins — happens
    inside the implementation and does not expand the shared interface.
    """

    name: ClassVar[CommandName] = CommandName.pretrain
    config_name: ClassVar[str] = "pretrain"

    @abstractmethod
    def build_module(self, cfg: DictConfig, datamodule: pl.LightningDataModule) -> pl.LightningModule:
        """Build the ``LightningModule`` to pretrain."""
        raise NotImplementedError


class InferCommand(MEDSModelCommand):
    """Materialize reusable outputs from a pretrained model.

    - **input**  ``cfg.input_data_dir`` + ``cfg.input_pretrained_model_dir`` +
      ``cfg.external_labels_dir``, which fixes the timepoints to infer at.
    - **output** ``cfg.input_data_dir/<cfg.output_inference_subdir>/artifacts.parquet``.

    What is produced is model-defined — embeddings, generated trajectories, hazards, native scores — so
    there is no ``inference.kind`` parameter. The kind is *recorded* in the artifact's manifest, and
    downstream consumers validate against it.
    """

    name: ClassVar[CommandName] = CommandName.infer
    config_name: ClassVar[str] = "infer"


class SupervisedTrainCommand(MEDSModelCommand):
    """Train a supervised model, from scratch or on top of one prior artifact.

    - **input**  ``cfg.input_data_dir`` + ``cfg.external_labels_dir``, plus at most one of
      ``cfg.input_pretrained_model_dir`` (fine-tune) or ``cfg.input_inference_subdir`` (probe).
    - **output** ``cfg.output_supervised_model_dir``.
    """

    name: ClassVar[CommandName] = CommandName.supervised_train
    config_name: ClassVar[str] = "supervised_train"

    sources: ClassVar[tuple[str, ...]] = ("input_pretrained_model_dir", "input_inference_subdir")
    require_source: ClassVar[bool] = False

    @abstractmethod
    def build_module(
        self,
        cfg: DictConfig,
        datamodule: pl.LightningDataModule,
        source: tuple[str | None, Any],
    ) -> pl.LightningModule:
        """Build the ``LightningModule`` to train, given the arbitrated ``(role, value)`` source."""
        raise NotImplementedError


class PredictCommand(MEDSModelCommand):
    """Produce standardized predictions for a task.

    - **input**  ``cfg.external_labels_dir`` plus exactly one of ``cfg.input_supervised_model_dir``,
      ``cfg.input_pretrained_model_dir`` or ``cfg.input_inference_subdir`` — unless the implementation
      declares a ``packaged_model``. ``cfg.input_data_dir`` is optional: when omitted it is recovered from
      the source artifact's manifest, which is what lets MEDS-DEV's rolling ``model_initialization_dir``
      point at a training output whose workspace lives elsewhere.
    - **output** ``cfg.output_predictions_dir/predictions.parquet``
      (``meds_evaluation.PredictionSchema``).

    **Coverage is part of the contract.** The output has one row per row of the selected splits of the task.
    A model that cannot score some index rows must fail rather than emit a short file: silently dropping
    rows turns a partial run into a plausible-looking complete one. ``run`` enforces this and records
    ``n_expected`` / ``n_written`` per split in the manifest.

    This command never reads ground truth. ``boolean_value`` is dropped when the index is loaded from
    ``external_labels_dir``; scoring is a separate, shared tool.
    """

    name: ClassVar[CommandName] = CommandName.predict
    config_name: ClassVar[str] = "predict"

    sources: ClassVar[tuple[str, ...]] = (
        "input_supervised_model_dir",
        "input_pretrained_model_dir",
        "input_inference_subdir",
    )
    require_source: ClassVar[bool] = True

    @abstractmethod
    def predict(
        self,
        cfg: DictConfig,
        source: tuple[str | None, Any],
        index: pl_df.DataFrame,
    ) -> pl_df.DataFrame:
        """Score every row of ``index``, returning ``subject_id, prediction_time, predicted_*`` columns."""
        raise NotImplementedError
