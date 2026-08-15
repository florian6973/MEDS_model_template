"""Model-owned DAG adapters plus shared MEDS-DEV MIMIC demo provisioning."""

import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class E2EResult:
    """Artifacts plus the requested evaluation index used to verify prediction keys."""

    artifacts: Mapping[str, Path]
    labels_dir: Path
    splits: Sequence[str]


@dataclass(frozen=True)
class MEDSDevRun:
    """MEDS-DEV invocation plus the evaluation index its predictions must cover."""

    args: Sequence[str]
    predictions_dir: Path
    labels_dir: Path
    splits: Sequence[str]


def run_e2e(tmp_path, **inputs):
    raise NotImplementedError(
        "Prepare a small MEDS input/task, execute the selected DAG's real commands, and return "
        "E2EResult(artifacts, labels_dir, evaluated splits)."
    )


def prepare_meds_dev_run(meds_dev, tmp_path):
    """Return ``MEDSDevRun(args, predictions_dir, labels_dir, splits)`` for this implementation."""
    raise NotImplementedError("Configure the selected DAG's real MEDS-DEV run")


def prepare_mimic_demo(meds_dev, tmp_path):
    """Return MEDS-DEV's ``(MIMIC demo, task labels, MIMIC predicates)`` paths.

    Use ``MEDS_DEMO_DIR`` when set; otherwise invoke MEDS-DEV's own demo builder. Pin an existing
    demo-compatible MEDS-DEV task rather than defining a task in this repository.
    """
    from .integration import RUN_TIMEOUT, run, venv_bin

    configured = os.environ.get("MEDS_DEMO_DIR")
    dataset_dir = Path(configured).resolve() if configured else tmp_path / "mimic-iv-demo"
    if not configured:
        run(
            [
                venv_bin(meds_dev) / "meds-dev-dataset",
                "dataset=MIMIC-IV",
                f"output_dir={dataset_dir}",
                "demo=true",
            ],
            timeout=RUN_TIMEOUT,
        )
    labels_dir = tmp_path / "mortality-in-icu-first-24h"
    run(
        [
            venv_bin(meds_dev) / "meds-dev-task",
            "task=mortality/in_icu/first_24h",
            "dataset=MIMIC-IV",
            f"dataset_dir={dataset_dir}",
            f"output_dir={labels_dir}",
        ],
        timeout=RUN_TIMEOUT,
    )
    predicates_file = meds_dev / "src" / "MEDS_DEV" / "datasets" / "MIMIC-IV" / "predicates.yaml"
    return dataset_dir, labels_dir, predicates_file
