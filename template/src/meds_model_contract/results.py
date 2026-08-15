"""Structured, commit-safe run evidence assembled from command manifests."""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
from pathlib import Path
from typing import Any

import yaml

from .manifest import digest_file, read_manifest


def _git(repo: Path, *args: str) -> str | None:
    try:
        return subprocess.run(
            ["git", "-C", str(repo), *args], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None


def write_run_result(
    run_dir: Path | str,
    *,
    repo: Path | str,
    artifacts: dict[str, Path | str],
    dataset: dict[str, Any],
    task: dict[str, Any],
    external_predicates_file: Path | str | None = None,
    status: str = "complete",
) -> Path:
    """Write `spec.yaml`, copied manifests, and a run-level `result.yaml`."""
    run_dir = Path(run_dir)
    repo = Path(repo).resolve()
    manifests_dir = run_dir / "manifests"
    manifests_dir.mkdir(parents=True, exist_ok=True)
    manifest_refs = {}
    for command, artifact in artifacts.items():
        manifest = read_manifest(artifact)
        destination = manifests_dir / f"{command}.yaml"
        shutil.copy2(Path(artifact) / "manifest.yaml", destination)
        manifest_refs[command] = {
            "path": str(destination.relative_to(run_dir)),
            "artifact_type": manifest["artifact"]["type"],
            "resources": manifest["resources"],
        }

    predicates = None
    if external_predicates_file is not None:
        predicates = {
            "path": str(Path(external_predicates_file).resolve()),
            "digest": digest_file(external_predicates_file),
        }
    commit = _git(repo, "rev-parse", "HEAD")
    dirty = bool(_git(repo, "status", "--porcelain"))
    spec = {"dataset": dataset, "task": task, "external_predicates_file": predicates}
    result = {
        "status": status,
        "source": {"repository": str(repo), "git_commit": commit, "dirty": dirty},
        "environment": {
            "hostname": platform.node(),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "cpu_count": os.cpu_count(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        },
        "commands": manifest_refs,
    }
    (run_dir / "spec.yaml").write_text(yaml.safe_dump(spec, sort_keys=False))
    output = run_dir / "result.yaml"
    output.write_text(yaml.safe_dump(result, sort_keys=False))
    return output
