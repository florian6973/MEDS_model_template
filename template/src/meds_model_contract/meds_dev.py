"""Install this repository's descriptor into a MEDS-DEV checkout."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def register(repo: Path, meds_dev: Path, name: str) -> Path:
    source = repo.resolve()
    target = meds_dev.resolve() / "src" / "MEDS_DEV" / "models" / name
    if not target.parent.is_dir():
        raise FileNotFoundError(f"{meds_dev} has no src/MEDS_DEV/models directory")
    target.mkdir()
    shutil.copy2(source / "model.yaml", target / "model.yaml")
    requirements = (source / "requirements.txt").read_text().replace("-e .", f"-e {source}")
    (target / "requirements.txt").write_text(requirements)
    predicates = source / "predicates.yaml"
    if predicates.is_file():
        shutil.copy2(predicates, target / "predicates.yaml")
    return target


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--meds-dev", type=Path, required=True)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--name", required=True)
    args = parser.parse_args()
    print(register(args.repo, args.meds_dev, args.name))
