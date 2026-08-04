"""The datamodule surface the commands rely on, promoted from duck typing to a named contract.

Every command constructs the datamodule through ``build_datamodule(cfg)`` (``hydra.utils.instantiate``
of ``cfg.datamodule``) and then interacts with it **only** through the surface below. The shipped MTD
configs satisfy it via meds-torch-data; a repo generated with ``data_backend=custom_featurization``
implements it in its own ``datamodule.py``. Conformance is structural — nothing checks
``isinstance``; these classes exist so the contract is written down in one place instead of scattered
across the commands that consume it.

Beyond this surface, a conforming datamodule must be a ``lightning.pytorch.LightningDataModule`` (it is
handed to ``trainer.fit``) and must accept ``batch_size`` and ``num_workers`` constructor arguments —
every command config and the conformance harness pass them.

Deliberately torch-free: importing this module must stay cheap (see the import-weight discipline note
in ``dispatch.py``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:  # pragma: no cover - typing only
    import polars as pl


class DataModuleConfig(Protocol):
    """The ``config`` object commands read from and write to.

    ``task_labels_dir`` is **set at runtime**: ``supervised_train`` / ``infer`` / ``predict``
    materialize ``external_labels_dir`` into a work directory and point the datamodule at it via
    ``cfg.datamodule.config.task_labels_dir = str(labels_dir)`` *before* instantiation. The layout at
    that path is ``{split}.parquet`` files written by :func:`meds_model_base.tasks.materialize_labels`;
    ``boolean_value`` may be absent (inference), which is how "prediction never reads ground truth"
    reaches the batch a model sees.

    ``vocab_size`` is the feature-space size handed to ``build_module`` as the model's ``vocab_size``
    kwarg. For the MTD backend it is the code-vocabulary size; for a featurized backend it is the
    feature count of the patients artifact (``len(features.json)``,
    :func:`meds_model_base.featurize.load_features`).
    """

    task_labels_dir: str | None
    vocab_size: int


class SplitDataset(Protocol):
    """One split's dataset, as ``predict``/``infer`` consume it.

    ``schema_df`` is the alignment contract: a polars frame with ``subject_id`` and ``prediction_time``
    rows in **loader iteration order** (prediction loaders must not shuffle). ``run_predict_step``
    zips it against the model's per-batch outputs, which is what lets models never hand-align their
    predictions to timepoints.
    """

    schema_df: pl.DataFrame

    def __len__(self) -> int: ...  # pragma: no cover - protocol


class MEDSModelDataModule(Protocol):
    """The datamodule attribute surface, keyed by MEDS split name via ``SPLIT_ATTRS``:

    ========== ================= ====================
    split      dataset attribute dataloader attribute
    ========== ================= ====================
    train      ``train_dataset`` ``train_dataloader``
    tuning     ``val_dataset``   ``val_dataloader``
    held_out   ``test_dataset``  ``test_dataloader``
    ========== ================= ====================
    """

    config: DataModuleConfig
    train_dataset: SplitDataset
    val_dataset: SplitDataset
    test_dataset: SplitDataset


__all__ = ["DataModuleConfig", "MEDSModelDataModule", "SplitDataset"]
