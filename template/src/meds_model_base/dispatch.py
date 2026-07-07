"""The ``meds-model`` dispatcher and its OmegaConf resolvers.

A generated repo's ``src/<model_slug>/__main__.py`` is a two-liner::

    from meds_model_base.dispatch import make_cli
    from .steps import STEPS
    cli = make_cli(STEPS, config_dir=...)   # registered as the `meds-model` console script

:func:`make_cli` returns a ``cli()`` that parses ``meds-model <step> [hydra overrides...]``, handles the
introspection sub-commands (``steps``, ``--help``), and hands the chosen step off to Hydra. Resolvers are
registered **explicitly** (via :func:`register_resolvers`) at CLI start — never as an import side-effect,
which was a fragile pattern in the reference model.
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

from omegaconf import OmegaConf

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Mapping

    from .steps.base import MEDSModelStep, StepName

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


def _prod(xs) -> float:
    out = 1
    for x in xs:
        out *= x
    return out


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
usage: meds-model <step> [hydra.overrides ...]

Steps implemented by this model:
{implemented}

Other commands:
  meds-model steps          list the steps this model implements
  meds-model <step> --help  show the step's Hydra config and overridable options
"""


def _implemented_block(steps: Mapping[StepName, type[MEDSModelStep]]) -> str:
    if not steps:
        return "  (none)"
    return "\n".join(f"  {name.value}" for name in steps)


def make_cli(steps: Mapping[StepName, type[MEDSModelStep]], config_dir: str):
    """Build the ``meds-model`` entry-point callable for a model's ``STEPS`` registry.

    Args:
        steps: mapping of :class:`~meds_model_base.steps.base.StepName` → step class for the steps this
            model implements.
        config_dir: absolute path to the model's packaged Hydra ``configs/`` directory.

    Returns:
        A zero-argument ``cli()`` suitable for a ``[project.scripts]`` console-script entry point.
    """

    def cli() -> None:
        from .steps.base import StepName

        register_resolvers()

        argv = sys.argv
        first = argv[1] if len(argv) > 1 else None

        if first in (None, "-h", "--help", "help"):
            print(_USAGE.format(implemented=_implemented_block(steps)))
            sys.exit(0)

        if first == "steps":
            for name in steps:
                print(name.value)
            sys.exit(0)

        try:
            step_name = StepName(first)
        except ValueError:
            valid = ", ".join(s.value for s in StepName)
            sys.exit(f"error: unknown step {first!r}. Valid steps: {valid}.")

        if step_name not in steps:
            impl = ", ".join(s.value for s in steps) or "(none)"
            sys.exit(
                f"error: this model does not implement the {step_name.value!r} step. Implemented: {impl}."
            )

        # Hand the remaining argv (overrides / --help / --multirun) to Hydra. Pop the step token so
        # Hydra sees only its own arguments.
        step = steps[step_name]()
        del argv[1]
        _run_with_hydra(step, config_dir)

    return cli


def _run_with_hydra(step: MEDSModelStep, config_dir: str) -> None:
    """Wrap ``step.run`` in ``hydra.main`` and invoke it (consuming ``sys.argv`` overrides).

    Registers the meds-torch-data structured config just before composition so ``datamodule.config`` is a
    typed node. This import (and the torch it pulls) is deferred to here so the torch-free introspection
    paths (``meds-model steps`` / top-level ``--help``) stay cheap.
    """
    import hydra

    from .lightning import register_structured_configs

    register_structured_configs()
    hydra.main(version_base="1.3", config_path=config_dir, config_name=step.config_name)(step.run)()


def run_step(
    steps: Mapping[StepName, type[MEDSModelStep]],
    step: StepName | str,
    config_dir: str,
    overrides: list[str] | None = None,
) -> None:
    """Programmatically run one step (for tests / notebooks) without touching the real ``sys.argv``.

    Builds ``sys.argv`` as ``["meds-model", step, *overrides]`` and delegates to :func:`make_cli`. Because
    ``hydra.main`` calls ``sys.exit`` on completion, callers generally run this in a subprocess; it is
    provided mainly for symmetry and documentation.
    """
    from .steps.base import StepName

    name = StepName(step) if not isinstance(step, StepName) else step
    saved = sys.argv
    try:
        sys.argv = ["meds-model", name.value, *(overrides or [])]
        make_cli(steps, config_dir)()
    finally:
        sys.argv = saved
