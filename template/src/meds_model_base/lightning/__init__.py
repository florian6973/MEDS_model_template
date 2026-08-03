"""Lightning + meds-torch-data glue: datamodule and trainer construction, reusable modules.

- :func:`register_structured_configs` registers ``MEDSTorchDataConfig`` with Hydra's ConfigStore so a
  ``datamodule.config`` group is a type-checked structured config (enums are UPPERCASE on the CLI).
- :func:`build_datamodule` builds the meds-torch-data ``Datamodule`` from a resolved ``cfg.datamodule``,
  after :func:`require_statics_if_requested` checks the cohort can satisfy the static data it asks for.
- :func:`build_trainer` builds the ``Trainer``, directing checkpoints at a training run's work directory.
- :mod:`meds_model_base.lightning.modules` holds reusable ``nn.Module`` blocks + ``BaseLightningModule``.
- :mod:`meds_model_base.lightning.probe` holds the frozen-embedding probe (dataset assembly + head).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    import lightning.pytorch as pl
    from omegaconf import DictConfig

_STRUCTURED_CONFIGS_REGISTERED = False


def register_structured_configs(group: str = "datamodule/config") -> None:
    """Register ``MEDSTorchDataConfig`` as a Hydra structured config (idempotent).

    Called by the dispatcher immediately before ``hydra.main`` composes a command's config, so
    ``datamodule.config`` resolves to a typed node with a ``_target_`` that ``instantiate`` can build.
    """
    global _STRUCTURED_CONFIGS_REGISTERED
    if _STRUCTURED_CONFIGS_REGISTERED:
        return
    from meds_torchdata import MEDSTorchDataConfig

    MEDSTorchDataConfig.add_to_config_store(group)
    _STRUCTURED_CONFIGS_REGISTERED = True


#: Static-inclusion modes that make the batch carry ``static_code`` / ``static_numeric_value``.
_STATIC_MODES = frozenset({"include", "prepend"})


def require_statics_if_requested(cfg: DictConfig) -> None:
    """Refuse a cohort with no static measurements when the datamodule asks for them.

    meds-torch-data builds the static tensors from whatever the cohort holds. When that is *nothing at
    all* — no subject has a single null-time measurement — ``JointNestedRaggedTensorDict`` has no values
    to infer a dtype from, and collation dies inside the dataloader with::

        ValueError: Cannot infer dtype from empty values; provide an explicit `schema=`.

    That error names neither the config key that asked for static data nor the cohort that lacks it, and it
    arrives once training has already started. This is the same precondition check as
    ``_require_split_sharded``: knowable in a directory listing, so it belongs before the work rather than
    several minutes into it.

    A cohort with no static measurements is perfectly legitimate MEDS — the mismatch is between it and the
    request, so either side is a valid fix, and the message says so.

    Raises:
        ValueError: if static data is requested and the tensorized cohort has none.
    """
    node = cfg.get("datamodule")
    config = node.get("config") if node else None
    if not config:
        return
    # `StaticInclusionMode` is a StrEnum, but the Hydra structured config spells it UPPERCASE.
    if str(config.get("static_inclusion_mode", "")).lower() not in _STATIC_MODES:
        return

    cohort = config.get("tensorized_cohort_dir")
    if not cohort:
        return
    schemas = sorted(Path(cohort).glob("tokenization/schemas/*/*.parquet"))
    if not schemas:
        return  # not a tensorized cohort at all; meds-torch-data's own error is the clearer one here

    import polars as pl

    frame = pl.scan_parquet(schemas)
    if "static_code" not in frame.collect_schema().names():
        return
    if frame.select(pl.col("static_code").list.len().sum()).collect().item():
        return

    raise ValueError(
        f"datamodule.config.static_inclusion_mode is "
        f"{config.get('static_inclusion_mode')!s}, but no subject in {cohort} has any static "
        "measurement, so meds-torch-data would fail to collate a batch at all ('Cannot infer dtype from "
        "empty values').\n\n"
        "Static measurements are MEDS rows with a null `time` — baseline variables such as age, sex or "
        "ethnicity. Either:\n"
        "  * the source dataset genuinely has none, in which case set "
        "`datamodule.config.static_inclusion_mode=OMIT` and drop them from the model; or\n"
        "  * they were lost in preprocessing — check that `external_meds_dir` carries null-time rows and "
        "that any `pipeline=` you passed preserves them."
    )


def build_datamodule(cfg: DictConfig) -> pl.LightningDataModule:
    """Instantiate the meds-torch-data ``Datamodule`` from a resolved ``cfg.datamodule`` node.

    Thin wrapper around ``hydra.utils.instantiate`` kept as a single choke-point so every command
    constructs the datamodule the same way (and so tests can monkeypatch it). Being the one place every
    command builds a datamodule is also why :func:`require_statics_if_requested` runs here.
    """
    from hydra.utils import instantiate

    require_statics_if_requested(cfg)
    return instantiate(cfg.datamodule)


def instantiate_group(node) -> list:
    """Instantiate every ``_target_`` child of a Hydra mapping ``node`` into a list (order-preserving).

    Used for ``callbacks`` / ``logger`` groups: a mapping of names → ``_target_`` configs becomes a list of
    built objects. Missing/empty nodes yield ``[]``.
    """
    from hydra.utils import instantiate
    from omegaconf import DictConfig

    if not node:
        return []
    items = node.values() if isinstance(node, DictConfig) else node
    return [instantiate(child) for child in items if child and "_target_" in child]


def build_trainer(cfg: DictConfig, checkpoint_dir=None) -> pl.Trainer:
    """Build the Lightning ``Trainer`` from ``cfg.trainer`` with callbacks + loggers wired in.

    Callbacks (``cfg.callbacks``) and loggers (``cfg.logger``) are instantiated from their groups and passed
    explicitly, which is more robust than nested-list interpolation inside the Trainer config.

    ``checkpoint_dir`` points every ``ModelCheckpoint`` at a training run's *work* directory. In-progress
    checkpoints must not be written where the published artifact will go: artifacts are renamed into place
    atomically and only exist once complete, so a crashed run leaves scratch behind rather than a directory
    that looks like a finished model.
    """
    from hydra.utils import instantiate

    callbacks = instantiate_group(cfg.get("callbacks"))
    if checkpoint_dir is not None:
        from lightning.pytorch.callbacks import ModelCheckpoint

        for cb in callbacks:
            if isinstance(cb, ModelCheckpoint):
                cb.dirpath = str(checkpoint_dir)
    loggers = instantiate_group(cfg.get("logger"))
    trainer_kwargs = {"callbacks": callbacks, "logger": loggers or False}
    if checkpoint_dir is not None:
        trainer_kwargs["default_root_dir"] = str(checkpoint_dir)
    return instantiate(cfg.trainer, **trainer_kwargs)
