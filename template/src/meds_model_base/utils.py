"""Small run-management helpers shared by the training / inference / prediction steps.

Kept dependency-light (omegaconf only at module load; lightning imported lazily) so importing a step
module doesn't force torch unless it is actually run.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)

#: Filenames written into a training run's ``output_dir`` / ``model_initialization_dir``.
CONFIG_FILENAME = "config.yaml"
RESOLVED_CONFIG_FILENAME = "resolved_config.yaml"
ENVIRONMENT_FILENAME = "environment.txt"
BEST_CKPT_FILENAME = "best_model.ckpt"


def save_environment_snapshot(fp: Path) -> None:
    """Write a ``pip freeze``-style snapshot of installed packages to ``fp`` (best-effort)."""
    try:
        from importlib.metadata import distributions

        lines = sorted(
            f"{d.metadata['Name']}=={d.version}" for d in distributions() if d.metadata["Name"] is not None
        )
        fp.write_text("\n".join(lines) + "\n")
    except Exception as e:  # pragma: no cover - snapshot is best-effort
        logger.warning("Could not write environment snapshot: %s", e)


def save_resolved_config(cfg: DictConfig, fp: Path) -> None:
    """Save a fully-resolved copy of ``cfg`` (interpolations expanded) to ``fp``."""
    resolved = OmegaConf.to_container(cfg, resolve=True)
    OmegaConf.save(OmegaConf.create(resolved), fp)


def prepare_output_dir(cfg: DictConfig) -> tuple[Path, Path | None]:
    """Resolve the run output dir, honoring ``do_overwrite`` / ``do_resume``.

    Returns ``(output_dir, resume_ckpt)`` where ``resume_ckpt`` is a checkpoint to resume from (or None).
    Raises ``FileExistsError`` if the directory is populated and neither flag is set.
    """
    output_dir = Path(cfg.output_dir)
    if output_dir.is_file():
        raise NotADirectoryError(f"output_dir {output_dir} is a file, not a directory.")

    config_fp = output_dir / CONFIG_FILENAME
    resume_ckpt: Path | None = None
    do_overwrite = bool(cfg.get("do_overwrite", False))
    do_resume = bool(cfg.get("do_resume", False))

    if config_fp.exists():
        if do_overwrite:
            logger.info("Overwriting existing output_dir %s.", output_dir)
            shutil.rmtree(output_dir, ignore_errors=True)
        elif do_resume:
            resume_ckpt = find_checkpoint(output_dir)
            logger.info("Resuming from %s.", resume_ckpt)
        else:
            raise FileExistsError(
                f"output_dir {output_dir} already exists and is populated. "
                "Set do_overwrite=True or do_resume=True to proceed."
            )
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir, resume_ckpt


def find_checkpoint(run_dir: Path) -> Path | None:
    """Find a checkpoint to resume/load from a run dir (``best_model.ckpt`` → ``last.ckpt`` → newest)."""
    run_dir = Path(run_dir)
    best = run_dir / BEST_CKPT_FILENAME
    if best.is_file():
        return best
    ckpts = sorted(run_dir.rglob("*.ckpt"), key=lambda p: p.stat().st_mtime, reverse=True)
    for name in ("last.ckpt",):
        for c in ckpts:
            if c.name == name:
                return c
    return ckpts[0] if ckpts else None


def write_run_metadata(cfg: DictConfig, output_dir: Path) -> None:
    """Persist ``config.yaml`` + ``resolved_config.yaml`` + ``environment.txt`` for a fresh run."""
    OmegaConf.save(cfg, output_dir / CONFIG_FILENAME)
    save_resolved_config(cfg, output_dir / RESOLVED_CONFIG_FILENAME)
    save_environment_snapshot(output_dir / ENVIRONMENT_FILENAME)
