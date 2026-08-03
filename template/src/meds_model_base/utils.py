"""Small run-management helpers shared by the training, inference and prediction commands.

Artifact directories are published atomically by :mod:`meds_model_base.manifest`, so nothing here writes
into a final output location. Training, however, needs somewhere durable to keep in-progress checkpoints:
that is the *work directory*, a scratch sibling of the artifact that survives a crash and is what
``do_resume`` resumes from. The artifact itself only ever appears complete.

Kept dependency-light (omegaconf only at module load; lightning imported lazily).
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)

#: Name of the checkpoint copied into a published model artifact.
BEST_CKPT_FILENAME = "checkpoint"

#: Suffix for the scratch directory holding in-progress training state.
WORK_DIR_SUFFIX = ".work"


def work_dir_for(output_dir: Path | str) -> Path:
    """Resolve the scratch directory for a training run.

    Always a sibling of the artifact directory suffixed with ``.work``. This is deliberately *outside* the
    artifact so that a failed run leaves no partial artifact behind, while still leaving checkpoints on
    disk to resume from.

    Examples:
        >>> str(work_dir_for("/runs/models/pretrained"))
        '/runs/models/pretrained.work'
    """
    output_dir = Path(output_dir)
    return output_dir.with_name(output_dir.name + WORK_DIR_SUFFIX)


def prepare_work_dir(output_dir: Path | str, cfg=None) -> tuple[Path, Path | None]:
    """Create (or reuse) the work directory; return ``(work_dir, resume_ckpt)``.

    ``resume_ckpt`` is a checkpoint to resume from when ``cfg.do_resume`` is set and one exists. When
    ``do_resume`` is false any prior scratch state is discarded, so a fresh run never silently inherits
    checkpoints from an unrelated earlier attempt.

    **``do_resume`` ships as false, and that default is the safety property.** The work directory is
    derived from the output path alone; nothing in it records which config, task or dataset produced the
    checkpoint, and ``trainer.fit(ckpt_path=...)`` restores optimizer state, LR-schedule position and the
    epoch counter along with the weights. Scratch is removed once the artifact is published, so a
    checkpoint survives only a crash — which is exactly when the config gets changed before the
    re-run. Resuming across that change yields a model trained under two different configurations that
    trains, predicts and looks entirely plausible. Turn it on to continue the *same* run; a changed config
    needs a fresh one.
    """
    work_dir = work_dir_for(output_dir)
    do_resume = bool(cfg.get("do_resume", False)) if cfg is not None else False

    resume_ckpt: Path | None = None
    if work_dir.exists():
        if do_resume:
            resume_ckpt = find_checkpoint(work_dir)
            if resume_ckpt is not None:
                logger.info("Resuming training from %s.", resume_ckpt)
            else:
                logger.info("do_resume set but no checkpoint in %s; starting fresh.", work_dir)
        else:
            logger.info("Discarding stale work directory %s (do_resume is false).", work_dir)
            shutil.rmtree(work_dir, ignore_errors=True)
    work_dir.mkdir(parents=True, exist_ok=True)
    return work_dir, resume_ckpt


def find_checkpoint(run_dir: Path | str) -> Path | None:
    """Find a checkpoint in a run or work dir (``checkpoint`` → ``last.ckpt`` → most recent ``*.ckpt``)."""
    run_dir = Path(run_dir)
    best = run_dir / BEST_CKPT_FILENAME
    if best.is_file():
        return best
    ckpts = sorted(run_dir.rglob("*.ckpt"), key=lambda p: p.stat().st_mtime, reverse=True)
    for c in ckpts:
        if c.name == "last.ckpt":
            return c
    return ckpts[0] if ckpts else None


def require_checkpoint(model_dir: Path | str) -> Path:
    """Like :func:`find_checkpoint` but raises with an actionable message instead of returning None."""
    ckpt = find_checkpoint(model_dir)
    if ckpt is None:
        raise FileNotFoundError(
            f"No checkpoint found in {model_dir}. Expected a {BEST_CKPT_FILENAME!r} file written by a "
            "`pretrain` or `supervised_train` run."
        )
    return ckpt


def resolve_subdir(data_dir: Path | str, subdir: str | None) -> Path | None:
    """Resolve a ``*_subdir`` against the shared ``data_dir``; ``None`` passes through.

    Subdirectory arguments are always relative to ``data_dir`` — an absolute path is a caller error, since
    task and inference artifacts live inside the shared workspace by construction.

    Examples:
        >>> str(resolve_subdir("/runs/data", "tasks/mortality"))
        '/runs/data/tasks/mortality'
        >>> resolve_subdir("/runs/data", None) is None
        True
    """
    if subdir is None:
        return None
    p = Path(subdir)
    if p.is_absolute():
        raise ValueError(
            f"Subdirectory arguments are relative to data_dir, but got the absolute path {subdir!r}."
        )
    return Path(data_dir) / p
