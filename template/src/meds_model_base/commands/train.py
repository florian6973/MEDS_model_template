"""``pretrain`` and ``supervised_train`` — the two training commands.

Both share one Lightning flow: build the datamodule, build the ``LightningModule``, fit, then publish the
result atomically as a model artifact. They differ in what data they see and what they may start from:

- ``pretrain`` sees patient data only. There is no ``input_task_subdir`` in the shared interface; a model
  that needs targets (EveryQuery's queries, MOTOR's time-to-event bins) derives them internally.
- ``supervised_train`` sees a task, and may start from **at most one** prior artifact — a pretrained model
  to fine-tune, or inference artifacts to probe. Which one is decided by
  :meth:`~meds_model_base.commands.base.MEDSModelCommand.validate` before any work happens.

**In-progress state lives in a work directory**, not in the artifact. A crashed run therefore leaves no
artifact at all — only scratch that ``do_resume`` can pick up — so a model directory that exists is always
a directory that finished.

**A warm start that matches no parameters is an error.** Loading a checkpoint non-strictly is what makes
fine-tuning a fresh head possible, but it also means a renamed encoder silently yields a randomly
initialized model that trains and predicts perfectly plausibly. The matched-parameter count is checked here
and recorded in the manifest.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

from ..manifest import ArtifactType, InferenceKind, input_ref, read_manifest, write_artifact
from ..utils import BEST_CKPT_FILENAME, prepare_work_dir, require_checkpoint, resolve_subdir
from .base import PretrainCommand, SupervisedTrainCommand

if TYPE_CHECKING:  # pragma: no cover - typing only
    import lightning.pytorch as pl
    from omegaconf import DictConfig

logger = logging.getLogger(__name__)


class _TrainFlow:
    """Shared fit-then-publish flow for both training commands."""

    artifact_type: ClassVar[ArtifactType]
    output_key: ClassVar[str]

    def run(self, cfg: DictConfig) -> Path:
        import torch
        from lightning.pytorch import seed_everything

        from ..lightning import build_datamodule, build_trainer

        source = self.source
        output_dir = Path(cfg[self.output_key])
        work_dir, resume_ckpt = prepare_work_dir(output_dir, cfg)

        if cfg.get("seed") is not None:
            seed_everything(cfg.seed, workers=True)
        torch.set_float32_matmul_precision("medium")

        datamodule = build_datamodule(cfg)
        model, model_extras = self.build_and_prepare(cfg, datamodule, source)

        trainer = build_trainer(cfg, checkpoint_dir=work_dir)
        trainer.fit(model, datamodule, ckpt_path=str(resume_ckpt) if resume_ckpt else None)

        with write_artifact(
            output_dir,
            artifact_type=self.artifact_type,
            command=self.name.value,
            inputs=self.input_refs(cfg, source),
            config=cfg,
            do_overwrite=bool(cfg.get("do_overwrite", False)),
        ) as (staging, extras):
            extras.update(_persist_checkpoint(trainer, staging))
            extras["module_class"] = f"{type(model).__module__}.{type(model).__name__}"
            extras["training"] = {
                "seed": cfg.get("seed"),
                "epochs": int(trainer.current_epoch),
                "global_step": int(trainer.global_step),
            }
            extras.update(model_extras)

        if bool(cfg.get("clean_work_dir", True)):
            shutil.rmtree(work_dir, ignore_errors=True)
        logger.info("Training complete; artifact at %s.", output_dir)
        return output_dir

    def build_and_prepare(
        self, cfg: DictConfig, datamodule: pl.LightningDataModule, source: tuple[str | None, Any]
    ) -> tuple[pl.LightningModule, dict]:
        """Build the module and return it with any manifest fields describing how it was initialized."""
        raise NotImplementedError

    def input_refs(self, cfg: DictConfig, source: tuple[str | None, Any]) -> list:
        """The ``inputs`` block for the output manifest."""
        refs = [input_ref("input_data_dir", Path(cfg.input_data_dir))]
        role, value = source
        if role is not None:
            path = resolve_subdir(cfg.input_data_dir, value) if role.endswith("_subdir") else Path(value)
            refs.append(input_ref(role, path))
        return refs


class DefaultPretrainCommand(_TrainFlow, PretrainCommand):
    """Self-supervised / foundation-model pretraining with the default Lightning flow."""

    artifact_type: ClassVar[ArtifactType] = ArtifactType.pretrained_model
    output_key: ClassVar[str] = "output_pretrained_model_dir"

    def build_module(self, cfg: DictConfig, datamodule: pl.LightningDataModule) -> pl.LightningModule:
        from hydra.utils import instantiate

        return instantiate(cfg.model, vocab_size=datamodule.config.vocab_size)

    def build_and_prepare(self, cfg, datamodule, source):
        return self.build_module(cfg, datamodule), {}


class DefaultSupervisedTrainCommand(_TrainFlow, SupervisedTrainCommand):
    """Supervised training: from scratch, or fine-tuned from a pretrained model.

    Probing inference artifacts is a genuinely different training problem (dense feature vectors rather
    than tokenized sequences) and is implemented by :class:`ProbeTrainCommand`. This class declares only
    the sources it handles, so pointing it at ``input_inference_subdir`` fails with a clear message instead
    of quietly training from scratch.
    """

    artifact_type: ClassVar[ArtifactType] = ArtifactType.supervised_model
    output_key: ClassVar[str] = "output_supervised_model_dir"
    supported_sources: ClassVar[frozenset[str]] = frozenset({"input_pretrained_model_dir"})

    def build_module(self, cfg, datamodule, source):
        from hydra.utils import instantiate

        return instantiate(cfg.model, vocab_size=datamodule.config.vocab_size)

    def build_and_prepare(self, cfg, datamodule, source):
        model = self.build_module(cfg, datamodule, source)
        role, value = source
        if role != "input_pretrained_model_dir":
            return model, {"initialization": {"from": "scratch"}}
        stats = load_pretrained_weights(model, Path(value))
        return model, {"initialization": {"from": "pretrained_model", "path": str(value), **stats}}


class ProbeTrainCommand(_TrainFlow, SupervisedTrainCommand):
    """Train a probe on frozen inference artifacts (``input_inference_subdir``).

    Reads ``artifacts.parquet`` (validated to be of kind ``embeddings``), joins it to the task labels on
    ``(subject_id, prediction_time)``, and fits a small head. The pretrained model is never loaded — this
    is the cheap downstream half of the representation-probe chain.
    """

    artifact_type: ClassVar[ArtifactType] = ArtifactType.supervised_model
    output_key: ClassVar[str] = "output_supervised_model_dir"
    supported_sources: ClassVar[frozenset[str]] = frozenset({"input_inference_subdir"})
    require_source: ClassVar[bool] = True

    def build_module(self, cfg, datamodule, source):  # pragma: no cover - unused; see run()
        raise NotImplementedError("ProbeTrainCommand builds its module inside run().")

    def run(self, cfg: DictConfig) -> Path:
        import torch
        from lightning.pytorch import seed_everything

        from ..lightning import build_trainer
        from ..lightning.probe import LinearProbe, build_probe_dataloaders, load_probe_frames

        role, value = self.source
        inference_dir = resolve_subdir(cfg.input_data_dir, value)
        read_manifest(
            inference_dir,
            require_type=ArtifactType.inference,
            require_kind=InferenceKind.embeddings,
        )

        output_dir = Path(cfg[self.output_key])
        work_dir, resume_ckpt = prepare_work_dir(output_dir, cfg)

        if cfg.get("seed") is not None:
            seed_everything(cfg.seed, workers=True)
        torch.set_float32_matmul_precision("medium")

        frames, feature_column, coverage = load_probe_frames(
            inference_dir, resolve_subdir(cfg.input_data_dir, cfg.input_task_subdir)
        )
        loaders, input_dim = build_probe_dataloaders(frames, feature_column, cfg)
        model = LinearProbe(
            input_dim=input_dim,
            hidden_size=int(cfg.get("probe_hidden_size", 0) or 0),
            dropout=float(cfg.get("probe_dropout", 0.0) or 0.0),
        )

        trainer = build_trainer(cfg, checkpoint_dir=work_dir)
        trainer.fit(model, **loaders, ckpt_path=str(resume_ckpt) if resume_ckpt else None)

        with write_artifact(
            output_dir,
            artifact_type=self.artifact_type,
            command=self.name.value,
            inputs=self.input_refs(cfg, (role, value)),
            config=cfg,
            do_overwrite=bool(cfg.get("do_overwrite", False)),
        ) as (staging, extras):
            extras.update(_persist_checkpoint(trainer, staging))
            extras["module_class"] = f"{type(model).__module__}.{type(model).__name__}"
            extras["training"] = {
                "seed": cfg.get("seed"),
                "epochs": int(trainer.current_epoch),
                "global_step": int(trainer.global_step),
            }
            extras["initialization"] = {
                "from": "inference_artifacts",
                "path": str(inference_dir),
                "feature_column": feature_column,
                "input_dim": input_dim,
                **coverage,
            }

        if bool(cfg.get("clean_work_dir", True)):
            shutil.rmtree(work_dir, ignore_errors=True)
        return output_dir


def load_pretrained_weights(model: pl.LightningModule, pretrained_dir: Path) -> dict:
    """Warm-start ``model`` from a checkpoint, failing if nothing actually loaded.

    Returns the match statistics recorded in the manifest.

    Raises:
        RuntimeError: if no parameter in the checkpoint matched the model, which means the model is still
            randomly initialized and the "fine-tune" would silently be a from-scratch run.
    """
    import torch

    ckpt_fp = require_checkpoint(pretrained_dir)
    state = torch.load(ckpt_fp, map_location="cpu", weights_only=False)
    state_dict = state.get("state_dict", state)
    result = model.load_state_dict(state_dict, strict=False)

    model_keys = set(model.state_dict())
    matched = len(model_keys) - len(set(result.missing_keys))
    if matched == 0:
        raise RuntimeError(
            f"Warm start from {ckpt_fp} matched no parameters: every one of the model's {len(model_keys)} "
            "keys was missing from the checkpoint. The model would be randomly initialized despite "
            "'fine-tuning'. Check that the pretrained model and this model share an encoder."
        )
    logger.info(
        "Warm-started from %s (%d/%d parameters matched, %d unexpected).",
        ckpt_fp,
        matched,
        len(model_keys),
        len(result.unexpected_keys),
    )
    return {
        "checkpoint": str(ckpt_fp),
        "parameters_matched": matched,
        "parameters_total": len(model_keys),
        "unexpected_keys": len(result.unexpected_keys),
    }


def _persist_checkpoint(trainer: pl.Trainer, staging: Path) -> dict:
    """Copy the monitored best checkpoint into the staged artifact; return manifest fields."""
    dst = staging / BEST_CKPT_FILENAME
    cb = getattr(trainer, "checkpoint_callback", None)
    best_path = getattr(cb, "best_model_path", "") if cb is not None else ""
    if best_path and Path(best_path).is_file():
        shutil.copyfile(best_path, dst)
        score = getattr(cb, "best_model_score", None)
        return {
            "checkpoint": {
                "selection": "monitored_best",
                "monitor": getattr(cb, "monitor", None),
                "score": float(score) if score is not None else None,
            }
        }
    # No ModelCheckpoint, or no validation to monitor: snapshot the final weights so downstream commands
    # still have something to load, and say so in the manifest rather than implying a selected best.
    trainer.save_checkpoint(dst)
    return {"checkpoint": {"selection": "final_weights", "monitor": None, "score": None}}


__all__ = [
    "DefaultPretrainCommand",
    "DefaultSupervisedTrainCommand",
    "ProbeTrainCommand",
    "load_pretrained_weights",
]
