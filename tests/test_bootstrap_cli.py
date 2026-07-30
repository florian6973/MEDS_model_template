"""Tests for ``meds-model-new``, the thin wrapper over ``copier copy``.

This file exists because the wrapper's documented invocations were broken and nothing noticed: the module
docstring advertised a ``--profile`` flag that was never defined, and the help text advertised
``--data profile=...``, which argparse rejects before it can reach a ``nargs="*"`` positional. Both now
have a test, so the docs and the parser cannot drift apart again.

``subprocess.run`` is patched throughout — the point is the argv this builds, not whether copier is
installed.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from meds_model_template.__main__ import DEFAULT_SRC, main


def invoke(argv: list[str]) -> list[str]:
    """Run ``main(argv)`` with copier stubbed out; return the command it would have executed."""
    with patch("subprocess.run") as run:
        run.return_value.returncode = 0
        assert main(argv) == 0
        return list(run.call_args[0][0])


def test_minimal_invocation():
    assert invoke(["./my-model"]) == ["copier", "copy", DEFAULT_SRC, "./my-model"]


def test_profile_is_forwarded_as_copier_data():
    """The documented `--profile` form must actually work."""
    cmd = invoke(["./my-model", "--profile", "zero_shot_direct", "--defaults"])
    assert cmd == [
        "copier",
        "copy",
        DEFAULT_SRC,
        "./my-model",
        "--defaults",
        "--data",
        "profile=zero_shot_direct",
    ]


@pytest.mark.parametrize(
    "extra",
    [
        ["--data", "model_slug=my_model"],
        ["--vcs-ref=HEAD"],
        ["--trust", "--data", "profile=probe"],
    ],
)
def test_unknown_options_pass_through(extra):
    """Copier flags this wrapper does not define reach copier untouched, with no `--` separator."""
    assert invoke(["./my-model", *extra])[4:] == extra


def test_explicit_separator_still_works():
    """The old `--` form keeps working, so existing scripts do not break."""
    assert invoke(["./my-model", "--", "--data", "profile=probe"])[4:] == ["--data", "profile=probe"]


def test_src_override():
    cmd = invoke(["./my-model", "--src", "/local/template"])
    assert cmd == ["copier", "copy", "/local/template", "./my-model"]


def test_missing_copier_reports_actionable_error(capsys):
    with patch("subprocess.run", side_effect=FileNotFoundError):
        assert main(["./my-model"]) == 1
    assert "copier` is not installed" in capsys.readouterr().err
