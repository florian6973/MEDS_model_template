"""Shared opt-in integration-test utilities."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

MEDS_DEV_URL = "https://github.com/Medical-Event-Data-Standard/MEDS-DEV.git"
MEDS_DEV_PREDICATES_REF = "0c21a2226964181dee7ed28c9aa5aa0abdfe9765"
INSTALL_TIMEOUT = 1800
RUN_TIMEOUT = 5400


def run(cmd, *, timeout=RUN_TIMEOUT, env=None):
    result = subprocess.run(
        [str(value) for value in cmd],
        capture_output=True,
        text=True,
        timeout=timeout,
        env=None if env is None else {**os.environ, **env},
    )
    if result.returncode:
        raise AssertionError(
            f"command failed ({result.returncode}): {' '.join(map(str, cmd))}\n"
            f"--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}"
        )
    return result


def venv_bin(checkout: Path) -> Path:
    return checkout / ".venv" / ("Scripts" if sys.platform == "win32" else "bin")


def assert_meds_dev_predicates_capability(checkout: Path) -> None:
    """Require the explicit predicates_path passthrough proposed in MEDS-DEV PR #325."""
    config = checkout / "src" / "MEDS_DEV" / "configs" / "_run_model.yaml"
    implementation = checkout / "src" / "MEDS_DEV" / "models" / "__init__.py"
    if not config.is_file() or not implementation.is_file():
        raise AssertionError(f"{checkout} is not a MEDS-DEV source checkout")
    if (
        "predicates_path:" not in config.read_text()
        or 'format_kwargs["predicates_path"]' not in implementation.read_text()
    ):
        raise AssertionError(
            "MEDS-DEV lacks the predicates_path model-command passthrough. Use MEDS_DEV_REF="
            f"{MEDS_DEV_PREDICATES_REF} or a revision containing MEDS-DEV PR #325."
        )


def assert_prediction_keys(predictions_dir: Path, labels_dir: Path, splits) -> None:
    """Compare actual prediction keys with requested keys, independent of row order."""
    import polars as pl

    key_columns = ["subject_id", "prediction_time"]

    def read_keys(root: Path, *, selected_splits=None):
        files = sorted(root.rglob("*.parquet"))
        if selected_splits:
            selected = set(selected_splits)
            matching = [
                path
                for path in files
                if path.stem in selected or any(part in selected for part in path.relative_to(root).parts)
            ]
            if matching:
                files = matching
        frames = []
        for path in files:
            schema = pl.read_parquet_schema(path)
            if all(column in schema for column in key_columns):
                frames.append(pl.read_parquet(path, columns=key_columns))
        if not frames:
            raise AssertionError(f"No parquet files with {key_columns} found under {root}")
        return pl.concat(frames)

    predictions = read_keys(Path(predictions_dir))
    expected = read_keys(Path(labels_dir), selected_splits=splits)
    prediction_unique = predictions.unique()
    expected_unique = expected.unique()
    assert prediction_unique.height == predictions.height, "predictions contain duplicate evaluation keys"
    assert expected_unique.height == expected.height, "requested evaluation index contains duplicate keys"
    missing = expected_unique.join(prediction_unique, on=key_columns, how="anti")
    extra = prediction_unique.join(expected_unique, on=key_columns, how="anti")
    assert missing.is_empty() and extra.is_empty(), (
        f"prediction key mismatch: {missing.height} missing and {extra.height} extra keys"
    )


def provision_meds_dev(request, tmp_path_factory) -> Path:
    source = os.environ.get("MEDS_DEV_DIR")
    expression = request.config.getoption("markexpr", default="") or ""
    selected = "meds_dev" in expression or "real_data" in expression
    if source is None and not selected:
        pytest.skip("select -m meds_dev or -m real_data, or set MEDS_DEV_DIR")
    uv = shutil.which("uv")
    git = shutil.which("git")
    if uv is None or git is None:
        pytest.skip("uv and git are required for MEDS-DEV integration")
    checkout = tmp_path_factory.mktemp("meds-dev") / "MEDS-DEV"
    run([git, "clone", source or MEDS_DEV_URL, checkout], timeout=INSTALL_TIMEOUT)
    if source is None:
        ref = os.environ.get("MEDS_DEV_REF", MEDS_DEV_PREDICATES_REF)
        run([git, "checkout", "--detach", ref], timeout=INSTALL_TIMEOUT, env=None)
    assert_meds_dev_predicates_capability(checkout)
    run([uv, "venv", checkout / ".venv"], timeout=INSTALL_TIMEOUT)
    run(
        [uv, "pip", "install", "-e", checkout, "--python", venv_bin(checkout) / "python"],
        timeout=INSTALL_TIMEOUT,
        env={"SETUPTOOLS_SCM_PRETEND_VERSION_FOR_MEDS_DEV": "0.0.0"},
    )
    return checkout
