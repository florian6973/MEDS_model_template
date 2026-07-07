"""Template-level tests: render every profile and verify structure + syntactic validity.

These are fast and dependency-light (no torch): they render each profile with Copier and check the
generated files exist, the ``STEPS`` registry matches the profile, and every rendered Python module
byte-compiles. Running the *generated* repos' own test suites (which train models) is exercised separately
via `uv run pytest` inside a rendered repo.
"""

from __future__ import annotations

import compileall
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]

ALL_STEPS = {
    "preprocess",
    "unsupervised_train",
    "supervised_train",
    "task_agnostic_inference",
    "prediction",
}

#: profile → the exact set of steps its ``steps.py`` should register.
PROFILE_STEPS = {
    "supervised_basic": {"preprocess", "supervised_train", "prediction"},
    "zero_shot_ar": {"preprocess", "unsupervised_train", "task_agnostic_inference", "prediction"},
    "every_query": {"preprocess", "unsupervised_train", "prediction"},
    "motor_finetune": {"preprocess", "unsupervised_train", "supervised_train", "prediction"},
}


def _render(dst: Path, profile: str) -> str:
    """Render ``profile`` into ``dst`` (tasks skipped); return the model slug."""
    from copier import run_copy

    slug = f"m_{profile}"
    run_copy(
        str(REPO),
        str(dst),
        data={"model_slug": slug, "model_name": f"Demo {profile}", "profile": profile},
        defaults=True,
        unsafe=True,  # allow post-gen tasks (git init + best-effort `uvx ruff`, both no-ops if unavailable)
        quiet=True,
    )
    return slug


@pytest.mark.parametrize("profile,steps", sorted(PROFILE_STEPS.items()))
@pytest.mark.render
def test_render_profile(tmp_path, profile, steps):
    dst = tmp_path / profile
    slug = _render(dst, profile)

    for rel in [
        "pyproject.toml",
        "README.md",
        "model.yaml",
        ".copier-answers.yml",
        f"src/{slug}/__main__.py",
        f"src/{slug}/model.py",
        f"src/{slug}/steps.py",
        f"src/{slug}/configs/_prediction.yaml",
        f"src/{slug}/configs/model/default.yaml",
        "src/meds_model_base/dispatch.py",
        "src/meds_model_base/steps/base.py",
        "tests/test_smoke_pipeline.py",
    ]:
        assert (dst / rel).exists(), f"missing rendered file: {rel}"

    steps_txt = (dst / f"src/{slug}/steps.py").read_text()
    for step in steps:
        assert f"StepName.{step}" in steps_txt, f"{profile} steps.py missing {step}"
    # No step that shouldn't be there:
    for step in ALL_STEPS - steps:
        assert f"StepName.{step}:" not in steps_txt, f"{profile} steps.py unexpectedly registers {step}"

    # Every rendered Python module must byte-compile (catches broken Jinja output without importing torch).
    assert compileall.compile_dir(str(dst / "src"), quiet=1, force=True), "rendered code failed to compile"


@pytest.mark.render
def test_answers_file_records_profile(tmp_path):
    slug = _render(tmp_path / "sup", "supervised_basic")
    answers = (tmp_path / "sup" / ".copier-answers.yml").read_text()
    assert "profile: supervised_basic" in answers
    assert f"model_slug: {slug}" in answers
