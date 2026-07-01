"""``meds-model-new`` — a thin wrapper over ``copier copy`` for scaffolding a new MEDS model.

Usage::

    meds-model-new ./my-model                       # from the published template on GitHub
    meds-model-new ./my-model --src /path/to/template
    meds-model-new ./my-model --profile zero_shot_ar --defaults

Anything after ``--`` is forwarded verbatim to ``copier``.
"""

from __future__ import annotations

import argparse
import subprocess
import sys

#: Default template source used when ``--src`` is not given.
DEFAULT_SRC = "gh:mmcdermott/MEDS_model_template"


def main(argv: list[str] | None = None) -> int:
    """Entry point for the ``meds-model-new`` console script.

    Returns the exit code of the underlying ``copier`` process (or 1 if ``copier`` is missing).
    """
    parser = argparse.ArgumentParser(
        prog="meds-model-new",
        description="Scaffold a new MEDS model repository from MEDS_model_template (via Copier).",
    )
    parser.add_argument("destination", help="Directory to create the new model repository in.")
    parser.add_argument(
        "--src",
        default=DEFAULT_SRC,
        help=f"Template source (a git URL, gh: shorthand, or local path). Default: {DEFAULT_SRC}",
    )
    parser.add_argument(
        "--defaults",
        action="store_true",
        help="Accept all default answers (non-interactive).",
    )
    parser.add_argument(
        "copier_args",
        nargs="*",
        help="Extra arguments forwarded to `copier copy` (e.g. --data profile=zero_shot_ar).",
    )
    args = parser.parse_args(argv)

    cmd = ["copier", "copy", args.src, args.destination]
    if args.defaults:
        cmd.append("--defaults")
    cmd.extend(args.copier_args)

    try:
        return subprocess.run(cmd, check=False).returncode
    except FileNotFoundError:
        print(
            "error: `copier` is not installed. Install it with `uv tool install copier` or "
            "`pip install copier`.",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
