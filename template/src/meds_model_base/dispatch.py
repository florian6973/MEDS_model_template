"""The ``meds-model`` dispatcher and its OmegaConf resolvers.

A generated repo's ``src/<model_slug>/__main__.py`` is a two-liner::

    from meds_model_base.dispatch import make_cli
    from .commands import COMMANDS
    cli = make_cli(COMMANDS, config_dir=...)   # registered as the `meds-model` console script

:func:`make_cli` returns a ``cli()`` that parses ``meds-model <command> [hydra overrides...]``, handles the
introspection sub-commands (``commands``, ``--help``), and hands the chosen command off to Hydra.

Commands are always invoked through ``MEDSModelCommand.__call__``, never ``run`` directly, so argument
arbitration always happens. Resolvers are registered **explicitly** at CLI start rather than as an import
side-effect.
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

from omegaconf import OmegaConf

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Mapping

    from .commands.base import CommandName, MEDSModelCommand

# ------------------------------------------------------------------------------------------------------
# OmegaConf resolvers
# ------------------------------------------------------------------------------------------------------


def _num_cores() -> int:
    import os

    return os.cpu_count() or 1


def _num_gpus() -> int:
    """Number of visible CUDA devices (0 if torch is absent or CPU-only). Imported lazily."""
    try:
        import torch

        return torch.cuda.device_count()
    except Exception:  # pragma: no cover - torch missing / driver error
        return 0


def _gpus_available() -> bool:
    return _num_gpus() > 0


def _prod(xs) -> float:
    out = 1
    for x in xs:
        out *= x
    return out


#: Resolver name → callable. Registered with ``replace=True`` so repeated CLI invocations in one process
#: (e.g. the pytest smoke tests) don't raise "already registered".
RESOLVERS = {
    "num_cores": _num_cores,
    "num_gpus": _num_gpus,
    "gpus_available": _gpus_available,
    # small arithmetic helpers used by the base configs' num_workers / device math
    "sub": lambda a, b: a - b,
    "oc_min": lambda *xs: min(xs),
    "oc_max": lambda *xs: max(xs),
    "int_prod": lambda *xs: int(_prod(xs)),
}


def register_resolvers() -> None:
    """Register every resolver in :data:`RESOLVERS` with OmegaConf (idempotent).

    Examples:
        >>> register_resolvers()
        >>> from omegaconf import OmegaConf
        >>> OmegaConf.create({"n": "${sub:10,3}"}).n
        7
        >>> OmegaConf.create({"m": "${oc_min:4,2,9}"}).m
        2
    """
    for name, fn in RESOLVERS.items():
        OmegaConf.register_new_resolver(name, fn, replace=True)


# ------------------------------------------------------------------------------------------------------
# CLI construction
# ------------------------------------------------------------------------------------------------------

_USAGE = """\
usage: meds-model <command> [hydra.overrides ...]

Commands supported by this model:
{implemented}

Other invocations:
  meds-model commands          list the commands this model supports
  meds-model <command> --help  show the command's Hydra config and overridable options
"""


def _implemented_block(commands: Mapping[CommandName, type[MEDSModelCommand]]) -> str:
    if not commands:
        return "  (none)"
    return "\n".join(f"  {name.value}" for name in commands)


def make_cli(commands: Mapping[CommandName, type[MEDSModelCommand]], config_dir: str):
    """Build the ``meds-model`` entry-point callable for a model's ``COMMANDS`` registry.

    Args:
        commands: mapping of :class:`~meds_model_base.commands.base.CommandName` → command class, for the
            commands this model supports.
        config_dir: absolute path to the model's packaged Hydra ``configs/`` directory.

    Returns:
        A zero-argument ``cli()`` suitable for a ``[project.scripts]`` console-script entry point.
    """

    def cli() -> None:
        from .commands.base import CommandName

        register_resolvers()

        argv = sys.argv
        first = argv[1] if len(argv) > 1 else None

        if first in (None, "-h", "--help", "help"):
            print(_USAGE.format(implemented=_implemented_block(commands)))
            sys.exit(0)

        if first == "commands":
            for name in commands:
                print(name.value)
            sys.exit(0)

        try:
            command_name = CommandName(first)
        except ValueError:
            valid = ", ".join(c.value for c in CommandName)
            sys.exit(f"error: unknown command {first!r}. Valid commands: {valid}.")

        if command_name not in commands:
            impl = ", ".join(c.value for c in commands) or "(none)"
            sys.exit(
                f"error: this model does not support the {command_name.value!r} command. Supported: {impl}."
            )

        # Hand the remaining argv (overrides / --help / --multirun) to Hydra. Pop the command token so
        # Hydra sees only its own arguments.
        command = commands[command_name]()
        del argv[1]
        _run_with_hydra(command, config_dir)

    return cli


def _run_with_hydra(command: MEDSModelCommand, config_dir: str) -> None:
    """Wrap the command in ``hydra.main`` and invoke it (consuming ``sys.argv`` overrides).

    Registers the meds-torch-data structured config just before composition so ``datamodule.config`` is a
    typed node. That import (and the torch it pulls) is deferred to here so the torch-free introspection
    paths (``meds-model commands`` / top-level ``--help``) stay cheap.

    The command is invoked via ``__call__``, not ``run``, so argument arbitration cannot be skipped.
    """
    import hydra

    from .lightning import register_structured_configs

    register_structured_configs()
    hydra.main(version_base="1.3", config_path=config_dir, config_name=command.config_name)(command)()


def run_command(
    commands: Mapping[CommandName, type[MEDSModelCommand]],
    command: CommandName | str,
    config_dir: str,
    overrides: list[str] | None = None,
) -> None:
    """Programmatically run one command (for tests / notebooks) without touching the real ``sys.argv``.

    Because ``hydra.main`` calls ``sys.exit`` on completion, callers generally run this in a subprocess; it
    is provided mainly for symmetry and documentation.
    """
    from .commands.base import CommandName

    name = CommandName(command) if not isinstance(command, CommandName) else command
    saved = sys.argv
    try:
        sys.argv = ["meds-model", name.value, *(overrides or [])]
        make_cli(commands, config_dir)()
    finally:
        sys.argv = saved
