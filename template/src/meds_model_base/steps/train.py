"""Default training steps: ``unsupervised_train`` (b) and ``supervised_train`` (c).

Both share one Lightning flow — build the MTD datamodule, build the ``LightningModule`` (from ``cfg.model``
via ``instantiate``, injecting ``vocab_size``), fit, and persist ``best_model.ckpt`` + run metadata. The
only difference is the datamodule config group (``unsupervised_train`` uses a no-label datamodule; a
``supervised_train`` datamodule sets ``task_labels_dir`` + ``TO_END`` sampling) and that
``supervised_train`` may warm-start from a pretrained ``model_initialization_dir``.

Models are constructed from ``cfg.model`` (Hydra ``instantiate``), so a profile customizes the model by
swapping the ``model/`` config group — no Python override needed. Override ``build_module`` for programmatic
construction, or ``run`` for a fully custom loop.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import TYPE_CHECKING

from ..utils import BEST_CKPT_FILENAME, find_checkpoint, prepare_output_dir, write_run_metadata
from .base import SupervisedTrainStep, UnsupervisedTrainStep

if TYPE_CHECKING:  # pragma: no cover - typing only
    import lightning.pytorch as pl
    from omegaconf import DictConfig

logger = logging.getLogger(__name__)


class _DefaultTrainFlow:
    """Shared ``build_module`` + ``run`` for the two default training steps."""

    def build_module(
        self,
        cfg: "DictConfig",
        datamodule: "pl.LightningDataModule",
        pretrained_dir: Path | None = None,
    ) -> "pl.LightningModule":
        from hydra.utils import instantiate

        model = instantiate(cfg.model, vocab_size=datamodule.config.vocab_size)
        if pretrained_dir is not None:
            _load_pretrained_weights(model, pretrained_dir)
        return model

    def run(self, cfg: "DictConfig") -> Path:
        import torch
        from lightning.pytorch import seed_everything

        from ..lightning import build_datamodule, build_trainer

        output_dir, resume_ckpt = prepare_output_dir(cfg)
        if resume_ckpt is None:
            write_run_metadata(cfg, output_dir)

        if cfg.get("seed") is not None:
            seed_everything(cfg.seed, workers=True)
        torch.set_float32_matmul_precision("medium")

        datamodule = build_datamodule(cfg)
        pretrained = cfg.get("model_initialization_dir")
        model = self.build_module(cfg, datamodule, Path(pretrained) if pretrained else None)

        trainer = build_trainer(cfg)
        trainer.fit(model, datamodule, ckpt_path=str(resume_ckpt) if resume_ckpt else None)

        _persist_best_checkpoint(trainer, output_dir)
        logger.info("Training complete; artifacts in %s", output_dir)
        return output_dir


class DefaultUnsupervisedTrainStep(_DefaultTrainFlow, UnsupervisedTrainStep):
    """Self-supervised pretraining with the default Lightning flow."""


class DefaultSupervisedTrainStep(_DefaultTrainFlow, SupervisedTrainStep):
    """Supervised (fine-)tuning with the default Lightning flow.

    If ``cfg.model_initialization_dir`` is set, the checkpoint there is loaded (``strict=False``) to
    warm-start overlapping parameters (e.g. a pretrained encoder feeding a fresh task head).
    """


def _load_pretrained_weights(model: "pl.LightningModule", pretrained_dir: Path) -> None:
    """Warm-start ``model`` from a checkpoint in ``pretrained_dir`` (non-strict; overlapping params only)."""
    import torch

    ckpt_fp = find_checkpoint(pretrained_dir)
    if ckpt_fp is None:
        logger.warning("No checkpoint found in %s; training from scratch.", pretrained_dir)
        return
    state = torch.load(ckpt_fp, map_location="cpu", weights_only=False)
    state_dict = state.get("state_dict", state)
    result = model.load_state_dict(state_dict, strict=False)
    logger.info(
        "Warm-started from %s (missing=%d, unexpected=%d).",
        ckpt_fp,
        len(result.missing_keys),
        len(result.unexpected_keys),
    )


def _persist_best_checkpoint(trainer: "pl.Trainer", output_dir: Path) -> None:
    """Copy the monitored best checkpoint to ``output_dir/best_model.ckpt`` (or snapshot current weights)."""
    dst = output_dir / BEST_CKPT_FILENAME
    cb = getattr(trainer, "checkpoint_callback", None)
    best_path = getattr(cb, "best_model_path", "") if cb is not None else ""
    if best_path and Path(best_path).is_file():
        shutil.copyfile(best_path, dst)
        score = getattr(cb, "best_model_score", None)
        logger.info("Best checkpoint (score=%s) copied to %s.", score, dst)
    else:
        # No ModelCheckpoint / no validation: snapshot the final weights so downstream steps have a ckpt.
        trainer.save_checkpoint(dst)
        logger.info("No monitored checkpoint; saved final weights to %s.", dst)
