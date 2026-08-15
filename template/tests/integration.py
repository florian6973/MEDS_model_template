"""Shared opt-in integration-test utilities."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

MEDS_DEV_URL = "https://github.com/Medical-Event-Data-Standard/MEDS-DEV.git"
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
    run([uv, "venv", checkout / ".venv"], timeout=INSTALL_TIMEOUT)
    run(
        [uv, "pip", "install", "-e", checkout, "--python", venv_bin(checkout) / "python"],
        timeout=INSTALL_TIMEOUT,
        env={"SETUPTOOLS_SCM_PRETEND_VERSION_FOR_MEDS_DEV": "0.0.0"},
    )
    return checkout
