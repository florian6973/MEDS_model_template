"""Lightning + meds-torch-data glue: datamodule and trainer construction, reusable modules.

- :func:`register_structured_configs` registers ``MEDSTorchDataConfig`` with Hydra's ConfigStore so a
  ``datamodule.config`` group is a type-checked structured config (enums are UPPERCASE on the CLI).
- :func:`build_datamodule` builds the meds-torch-data ``Datamodule`` from a resolved ``cfg.datamodule``.
- :func:`build_trainer` builds the ``Trainer``, directing checkpoints at a training run's work directory.
- :mod:`meds_model_base.lightning.modules` holds reusable ``nn.Module`` blocks + ``BaseLightningModule``.
- :mod:`meds_model_base.lightning.probe` holds the frozen-embedding probe (dataset assembly + head).
"""

from __future__ import annotations

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


def build_datamodule(cfg: DictConfig) -> pl.LightningDataModule:
    """Instantiate the meds-torch-data ``Datamodule`` from a resolved ``cfg.datamodule`` node.

    Thin wrapper around ``hydra.utils.instantiate`` kept as a single choke-point so every command
    constructs the datamodule the same way (and so tests can monkeypatch it).
    """
    from hydra.utils import instantiate

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
