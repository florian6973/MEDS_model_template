"""Driving a model's command DAG from a test, without hardcoding the chain.

A generated repository's tests must not assume which commands its model registers — that is exactly the
thing a profile varies, and a test that assumes it silently exercises a different chain than the one the
model declares. :func:`run_chain` reads the ``COMMANDS`` registry and each command's ``supported_sources``
and threads the artifacts together from those declarations alone.

This lives in the contract rather than in a repository's ``conftest.py`` for two reasons: it is contract
knowledge (which artifact each command consumes), and a ``conftest`` is not importable from another test
module — ``tests/`` is not a package, so ``from .conftest import ...`` fails at collection time.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

from ..commands import CommandName

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Mapping

    from ..commands.base import MEDSModelCommand

#: Subdirectory `infer` writes to in these harness runs.
INFERENCE_SUBDIR = "inference/artifacts"


def run_cli(args: list[str]) -> subprocess.CompletedProcess:
    """Run ``meds-model <args>`` and fail loudly (with captured output) on non-zero exit.

    Commands run through the real console script, so a test exercises the dispatcher, config composition
    and artifact publication exactly as a user would — not an internal call that bypasses them.
    """
    result = subprocess.run(["meds-model", *args], capture_output=True, text=True)
    if result.returncode != 0:
        raise AssertionError(
            "Command failed: meds-model "
            + " ".join(args)
            + f"\n--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}"
        )
    return result


def supported_sources(commands: Mapping[CommandName, type[MEDSModelCommand]], name: CommandName):
    """The input sources the class registered for ``name`` declares it handles."""
    cls = commands[name]
    return cls.supported_sources if cls.supported_sources is not None else frozenset(cls.sources)


def build_workspace(meds_root: Path, task_file: Path, workspace: Path) -> Path:
    """Run the two preprocessing commands; return the shared ``data_dir``.

    No model is involved, so this half of the contract is exercisable from the moment a repository is
    generated — before its ``model.py`` stub is implemented.
    """
    run_cli(
        [
            "preprocess_data",
            f"external_meds_dir={meds_root}",
            f"output_data_dir={workspace}",
            "do_reshard=true",
            "do_overwrite=true",
        ]
    )
    run_cli(
        [
            "preprocess_task",
            f"input_data_dir={workspace}",
            f"external_task_file={task_file}",
            "do_overwrite=true",
        ]
    )
    return workspace


def run_chain(
    commands: Mapping[CommandName, type[MEDSModelCommand]],
    data_dir: Path,
    out_dir: Path,
    *,
    epochs: int = 2,
    batch_size: int = 4,
) -> dict[str, Path]:
    """Run every command in ``commands``, in dependency order, over an existing ``data_dir``.

    Returns the artifacts produced, keyed by role (``pretrained``, ``inference``, ``supervised``,
    ``predictions``). Which optional source each command receives comes from the registered class's
    ``supported_sources``, so this follows the model's own declaration rather than an assumption about
    the profile.
    """
    train = [
        f"trainer.max_epochs={epochs}",
        "trainer.accelerator=cpu",
        f"batch_size={batch_size}",
        "num_workers=0",
        "do_overwrite=true",
    ]
    artifacts: dict[str, Path] = {"data": data_dir}

    if CommandName.pretrain in commands:
        pretrained = out_dir / "pretrained"
        run_cli(
            [
                "pretrain",
                f"input_data_dir={data_dir}",
                f"output_pretrained_model_dir={pretrained}",
                *train,
            ]
        )
        artifacts["pretrained"] = pretrained

    if CommandName.infer in commands:
        if "pretrained" not in artifacts:
            raise AssertionError("`infer` is registered but `pretrain` is not; the DAG cannot run.")
        run_cli(
            [
                "infer",
                f"input_data_dir={data_dir}",
                f"input_pretrained_model_dir={artifacts['pretrained']}",
                f"output_inference_subdir={INFERENCE_SUBDIR}",
                f"batch_size={batch_size}",
                "do_overwrite=true",
            ]
        )
        artifacts["inference"] = data_dir / INFERENCE_SUBDIR

    if CommandName.supervised_train in commands:
        supervised = out_dir / "supervised"
        args = [
            "supervised_train",
            f"input_data_dir={data_dir}",
            f"output_supervised_model_dir={supervised}",
            *train,
        ]
        sources = supported_sources(commands, CommandName.supervised_train)
        if "input_pretrained_model_dir" in sources and "pretrained" in artifacts:
            args.append(f"input_pretrained_model_dir={artifacts['pretrained']}")
        elif "input_inference_subdir" in sources and "inference" in artifacts:
            args.append(f"input_inference_subdir={INFERENCE_SUBDIR}")
        run_cli(args)
        artifacts["supervised"] = supervised

    if CommandName.predict in commands:
        predictions = out_dir / "predictions"
        args = ["predict", f"input_data_dir={data_dir}"]
        sources = supported_sources(commands, CommandName.predict)
        if "input_supervised_model_dir" in sources:
            args.append(f"input_supervised_model_dir={artifacts['supervised']}")
        elif "input_pretrained_model_dir" in sources:
            args.append(f"input_pretrained_model_dir={artifacts['pretrained']}")
        elif "input_inference_subdir" in sources:
            args.append(f"input_inference_subdir={INFERENCE_SUBDIR}")
        # An empty `sources` means this model ships its own weights and takes no source argument.
        run_cli(
            [
                *args,
                f"output_predictions_dir={predictions}",
                "splits=[held_out]",
                f"batch_size={batch_size}",
                "do_overwrite=true",
            ]
        )
        artifacts["predictions"] = predictions

    return artifacts


__all__ = [
    "INFERENCE_SUBDIR",
    "build_workspace",
    "run_chain",
    "run_cli",
    "supported_sources",
]
