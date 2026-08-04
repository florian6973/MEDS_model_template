"""Install this repository's MEDS-DEV specification into a MEDS-DEV checkout.

``model.yaml`` and ``requirements.txt`` sit in this repository's root, but MEDS-DEV never reads them from
here. ``MEDS_DEV.models`` discovers models with ``files("MEDS_DEV.models").rglob("*/model.yaml")`` --
:mod:`importlib.resources` over the *installed* package -- so a model is registered only by physically
existing inside the installed ``MEDS_DEV`` package tree. That is why contributing requires an **editable**
install of a MEDS-DEV fork; with the PyPI distribution, files copied into a git checkout are never
imported, and files copied into ``site-packages`` vanish on the next reinstall and can never become a PR.

Two things have to happen, and only one of them is a copy:

- ``model.yaml`` is copied verbatim.
- ``requirements.txt`` is **rewritten**. MEDS-DEV builds an isolated venv and runs
  ``uv pip install -r <that file>`` from *its own* working directory, so a local checkout has to be
  referenced by absolute path. A relative ``-e .`` silently installs MEDS-DEV itself, after which the
  ``meds-model`` entry point does not exist and every command fails with ``command not found``.

Usage::

    meds-model-add-to-meds-dev --meds-dev ~/MEDS-DEV               # reference this checkout, editable
    meds-model-add-to-meds-dev --meds-dev ~/MEDS-DEV --published   # use this repo's requirements.txt
    meds-model-add-to-meds-dev --meds-dev ~/MEDS-DEV --dry-run     # print the plan, write nothing

``--published`` is the mode to use once the model is installable by URL -- a git reference or, if you have
released it, a PyPI pin. It reuses this repository's own ``requirements.txt`` rather than inventing one,
and refuses the unedited ``YOUR_ORG`` placeholder.
"""

from __future__ import annotations

import argparse
import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

#: Where a MEDS-DEV checkout keeps its models, relative to the repository root.
MODELS_SUBPATH = Path("src/MEDS_DEV/models")

#: The files MEDS-DEV's own contribution guide asks a model directory to carry.
SPEC_FILE = "model.yaml"
REQUIREMENTS_FILE = "requirements.txt"
README_FILE = "README.md"
#: Shipped alongside when the spec references {predicates_path}: the reference featurization bindings.
PREDICATES_FILE = "predicates.yaml"

_PROBE = (
    "import importlib.util as u; s = u.find_spec('MEDS_DEV'); "
    "print(next(iter(s.submodule_search_locations)) if s else '')"
)


def model_dir_name(repo: Path) -> str:
    """The MEDS-DEV model name to register under: the model slug, else the checkout's directory name.

    MEDS-DEV keys models by their directory name relative to ``src/MEDS_DEV/models``, so this is the
    ``$MODEL_NAME`` that ``meds-dev-model`` will expect.

    >>> import tempfile
    >>> with tempfile.TemporaryDirectory(suffix="-checkout") as d:
    ...     _ = (Path(d) / ".copier-answers.yml").write_text("model_slug: my_model\\nprofile: probe\\n")
    ...     model_dir_name(Path(d))
    'my_model'

    Without an answers file (it is not required to exist), the directory name stands in:

    >>> model_dir_name(Path("/home/me/MEDS-RETAIN"))
    'MEDS-RETAIN'
    """
    answers = repo / ".copier-answers.yml"
    if answers.is_file():
        recorded = yaml.safe_load(answers.read_text()) or {}
        slug = recorded.get("model_slug")
        if slug:
            return str(slug)
    return repo.resolve().name


def local_requirement(repo: Path) -> str:
    r"""Return the ``requirements.txt`` text that points MEDS-DEV at a local checkout.

    >>> print(local_requirement(Path("/home/me/my_model")), end="")
    # Written by `meds-model-add-to-meds-dev`. MEDS-DEV installs this file from its own working
    # directory, so the reference must stay absolute -- `-e .` would install MEDS-DEV instead.
    -e /home/me/my_model
    """
    return (
        "# Written by `meds-model-add-to-meds-dev`. MEDS-DEV installs this file from its own working\n"
        "# directory, so the reference must stay absolute -- `-e .` would install MEDS-DEV instead.\n"
        f"-e {repo.resolve()}\n"
    )


def published_requirement(text: str) -> str:
    r"""Reduce this repository's own ``requirements.txt`` to the lines MEDS-DEV should install.

    Comments in the shipped file explain the choice to a human reading the model repository; inside
    MEDS-DEV they are noise, so only real requirement lines survive.

    >>> print(published_requirement("# how to edit this\n\nmy-model @ git+https://x.com/me/m.git\n"))
    my-model @ git+https://x.com/me/m.git
    <BLANKLINE>

    A published reference may equally be a release pin:

    >>> print(published_requirement("my-model==0.1.0\n"))
    my-model==0.1.0
    <BLANKLINE>

    The placeholder the template ships is not a usable reference:

    >>> published_requirement("my-model @ git+https://github.com/YOUR_ORG/my_model.git\n")
    Traceback (most recent call last):
        ...
    ValueError: requirements.txt still contains the YOUR_ORG placeholder...

    Neither is a relative editable install, which is the documented way to install MEDS-DEV by accident:

    >>> published_requirement("-e .\n")
    Traceback (most recent call last):
        ...
    ValueError: `-e .` resolves against MEDS-DEV's working directory...
    """
    lines = [ln.rstrip() for ln in text.splitlines()]
    requirements = [ln for ln in lines if ln.strip() and not ln.lstrip().startswith("#")]

    if not requirements:
        raise ValueError(
            f"{REQUIREMENTS_FILE} declares no requirements. Add the reference that installs this model "
            "(a git URL or a released version), or drop --published to reference this checkout locally."
        )
    for line in requirements:
        if "YOUR_ORG" in line:
            raise ValueError(
                f"{REQUIREMENTS_FILE} still contains the YOUR_ORG placeholder. Replace it with the "
                "repository this model is published from, or drop --published to reference this "
                "checkout by absolute path instead."
            )
        if line.strip() in {"-e .", ".", "-e ./"}:
            raise ValueError(
                "`-e .` resolves against MEDS-DEV's working directory, not this repository, so it "
                "installs MEDS-DEV and leaves no `meds-model` entry point. Drop --published to write an "
                "absolute editable reference instead."
            )
    return "\n".join(requirements) + "\n"


def readme_stub(name: str, description: str, source: str | None) -> str:
    """A minimal ``README.md`` for the MEDS-DEV model directory, which its contribution guide requires.

    >>> print(readme_stub("my_model", "A demo model.", "git+https://github.com/me/my_model.git"))
    # my_model
    <BLANKLINE>
    A demo model.
    <BLANKLINE>
    Implementation: `git+https://github.com/me/my_model.git`.
    <BLANKLINE>
    Generated from MEDS_model_template. See `model.yaml` for the command graph this model runs.
    <BLANKLINE>
    """
    provenance = f"Implementation: `{source}`." if source else "<!-- TODO: link the model repository. -->"
    return (
        f"# {name}\n\n"
        f"{description}\n\n"
        f"{provenance}\n\n"
        "Generated from MEDS_model_template. See `model.yaml` for the command graph this model runs.\n"
    )


def spec_description(spec_text: str) -> str:
    """Pull ``metadata.description`` out of a ``model.yaml``, for the README stub.

    >>> spec_description("metadata:\\n  description: 'A demo model.'\\ncommands: {}\\n")
    'A demo model.'
    >>> spec_description("commands: {}\\n")
    ''
    """
    spec = yaml.safe_load(spec_text) or {}
    return str((spec.get("metadata") or {}).get("description", ""))


def source_reference(requirements_text: str) -> str | None:
    """The first real requirement line, used to point the README at where the model lives.

    >>> source_reference("# a comment\\nmy-model @ git+https://x.com/me/m.git\\n")
    'my-model @ git+https://x.com/me/m.git'
    >>> source_reference("# nothing but comments\\n") is None
    True

    An unedited placeholder is worse than no link at all — it would ship a dead URL into a MEDS-DEV PR,
    so the README falls back to a visible TODO instead:

    >>> source_reference("my-model @ git+https://github.com/YOUR_ORG/my_model.git\\n") is None
    True
    """
    for line in requirements_text.splitlines():
        if line.strip() and not line.lstrip().startswith("#"):
            return None if "YOUR_ORG" in line else line.strip()
    return None


def _import_location(python: Path | None = None) -> Path | None:
    """Where ``python`` (default: this interpreter) would import ``MEDS_DEV`` from, if it can at all."""
    if python is None:
        spec = importlib.util.find_spec("MEDS_DEV")
        locations = list(spec.submodule_search_locations or []) if spec else []
        return Path(locations[0]) if locations else None
    try:
        result = subprocess.run(
            [str(python), "-c", _PROBE], capture_output=True, text=True, timeout=60, check=False
        )
    except (OSError, subprocess.SubprocessError):  # pragma: no cover - environment dependent
        return None
    out = result.stdout.strip()
    return Path(out) if result.returncode == 0 and out else None


def check_registration(meds_dev: Path) -> tuple[bool, str]:
    """Check that ``meds_dev`` is the checkout an interpreter actually imports ``MEDS_DEV`` from.

    Returns ``(ok, message)``. ``ok`` is False only when we positively established a mismatch -- a
    MEDS-DEV importable from somewhere *other* than this checkout, which means files written here will
    never be discovered. When nothing importable is found we cannot tell, so the caller warns instead.
    """
    expected = (meds_dev / "src" / "MEDS_DEV").resolve()

    candidates: list[Path | None] = [None]
    for venv in (".venv", "venv"):
        interpreter = meds_dev / venv / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
        if interpreter.exists():
            candidates.append(interpreter)

    found = [location for location in map(_import_location, candidates) if location is not None]
    if any(location.resolve() == expected for location in found):
        return True, f"MEDS_DEV imports from {expected} -- this checkout."
    if found:
        return False, (
            f"MEDS_DEV imports from {', '.join(str(location) for location in found)}, not {expected}.\n"
            "That install shadows this checkout, so a model written here would never be discovered: "
            "MEDS-DEV finds models with `importlib.resources.files('MEDS_DEV.models')`, which resolves "
            "to the *installed* package. Contributing needs an editable install of your fork:\n"
            f"    cd {meds_dev} && pip install -e '.[dev,tests]'"
        )

    return True, (
        "Could not import MEDS_DEV from this interpreter or a venv inside the checkout, so registration "
        "was not verified. Confirm with:\n"
        "    python -c \"from importlib.resources import files; print(files('MEDS_DEV.models'))\"\n"
        f"It must print {expected / 'models'}; if it prints a site-packages path, MEDS-DEV is installed "
        "non-editably and models copied here will not be found."
    )


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="meds-model-add-to-meds-dev",
        description="Install this repository's model.yaml into a MEDS-DEV checkout.",
    )
    parser.add_argument(
        "--meds-dev", type=Path, required=True, help="path to your MEDS-DEV fork (editable install)"
    )
    parser.add_argument("--repo", type=Path, default=Path.cwd(), help="this model repository (default: cwd)")
    parser.add_argument("--name", help="MEDS-DEV model name (default: the model slug, else the repo dir)")
    parser.add_argument(
        "--published",
        action="store_true",
        help="install this model by the URL/version in its own requirements.txt, not as a local checkout",
    )
    parser.add_argument("--force", action="store_true", help="overwrite an existing model directory")
    parser.add_argument("--dry-run", action="store_true", help="print what would be written, write nothing")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Entry point for ``meds-model-add-to-meds-dev``."""
    args = _parse_args(argv)

    repo = args.repo.resolve()
    spec_path = repo / SPEC_FILE
    if not spec_path.is_file():
        print(
            f"error: no {SPEC_FILE} in {repo}. Run this from your model repository, or pass --repo.",
            file=sys.stderr,
        )
        return 1

    meds_dev = args.meds_dev.resolve()
    models_root = meds_dev / MODELS_SUBPATH
    if not models_root.is_dir():
        print(
            f"error: {meds_dev} does not look like a MEDS-DEV checkout ({models_root} is missing).\n"
            "Fork https://github.com/Medical-Event-Data-Standard/MEDS-DEV, clone it, and install it with "
            "`pip install -e '.[dev,tests]'`.",
            file=sys.stderr,
        )
        return 1

    ok, message = check_registration(meds_dev)
    if not ok:
        print(f"error: {message}", file=sys.stderr)
        return 1
    print(f"note: {message}")

    name = args.name or model_dir_name(repo)
    dest = models_root / name
    if dest.exists() and not args.force:
        print(f"error: {dest} already exists. Pass --force to overwrite it.", file=sys.stderr)
        return 1

    spec_text = spec_path.read_text()
    own_requirements = (repo / REQUIREMENTS_FILE).read_text() if (repo / REQUIREMENTS_FILE).is_file() else ""

    if args.published:
        try:
            requirements = published_requirement(own_requirements)
        except ValueError as e:
            print(f"error: {e}", file=sys.stderr)
            return 1
        source = source_reference(requirements)
    else:
        requirements = local_requirement(repo)
        source = source_reference(own_requirements)

    payload = {SPEC_FILE: spec_text, REQUIREMENTS_FILE: requirements}
    if args.force or not (dest / README_FILE).is_file():
        payload[README_FILE] = readme_stub(name, spec_description(spec_text), source)
    # A model whose commands reference {predicates_path} needs a reference predicates file benchmark
    # users can pass; shipping the repo's own predicates.yaml alongside model.yaml is what makes
    # `meds-dev-model ... predicates_path=<meds-dev>/models/<name>/predicates.yaml` possible.
    if (repo / PREDICATES_FILE).is_file() and "{predicates_path}" in spec_text:
        payload[PREDICATES_FILE] = (repo / PREDICATES_FILE).read_text()

    if args.dry_run:
        print(f"\nwould write {len(payload)} files under {dest}:")
        for filename, content in payload.items():
            print(f"\n--- {dest / filename} ---")
            print(content, end="")
        return 0

    if dest.exists() and args.force:
        shutil.rmtree(dest)
    dest.mkdir(parents=True, exist_ok=True)
    for filename, content in payload.items():
        (dest / filename).write_text(content)
        print(f"wrote {dest / filename}")

    print(
        f"\nRegistered as MEDS-DEV model `{name}`. Check that MEDS-DEV agrees:\n"
        f"    python -c \"from MEDS_DEV import MODELS; print('{name}' in MODELS)\"\n"
        f"Then run it:\n"
        f"    meds-dev-model model={name} dataset=... task=... output_dir=..."
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - module entry point
    raise SystemExit(main())
