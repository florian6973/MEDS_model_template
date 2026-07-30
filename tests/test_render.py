"""Template-level tests: render every profile and verify structure + syntactic validity.

Fast and dependency-light (no torch): render each profile with Copier and check that the generated files
exist, the ``COMMANDS`` registry matches the profile, the Hydra configs parse, and every rendered Python
module byte-compiles. Running the *generated* repos' own suites (which train models) is exercised
separately by rendering one and running ``uv run pytest`` inside it.
"""

from __future__ import annotations

import compileall
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]

ALL_COMMANDS = {
    "preprocess_data",
    "preprocess_task",
    "pretrain",
    "infer",
    "supervised_train",
    "predict",
}

#: Every profile supports these: each chain starts by materializing patient data and a task.
ALWAYS = {"preprocess_data", "preprocess_task"}

#: profile → the exact set of commands its ``commands.py`` should register.
PROFILE_COMMANDS = {
    "supervised_basic": ALWAYS | {"supervised_train", "predict"},
    "zero_shot_ar": ALWAYS | {"pretrain", "infer", "predict"},
    "every_query": ALWAYS | {"pretrain", "predict"},
    "motor_finetune": ALWAYS | {"pretrain", "supervised_train", "predict"},
    "probe": ALWAYS | {"pretrain", "infer", "supervised_train", "predict"},
}

#: One Hydra root config per command, named after it.
COMMAND_CONFIGS = sorted(ALL_COMMANDS)


def _render(dst: Path, profile: str) -> str:
    """Render ``profile`` into ``dst``; return the model slug."""
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


@pytest.mark.parametrize("profile,commands", sorted(PROFILE_COMMANDS.items()))
@pytest.mark.render
def test_render_profile(tmp_path, profile, commands):
    dst = tmp_path / profile
    slug = _render(dst, profile)

    for rel in [
        "pyproject.toml",
        "README.md",
        "model.yaml",
        ".copier-answers.yml",
        f"src/{slug}/__main__.py",
        f"src/{slug}/model.py",
        f"src/{slug}/commands.py",
        f"src/{slug}/configs/paths/default.yaml",
        f"src/{slug}/configs/profile/default.yaml",
        f"src/{slug}/configs/model/default.yaml",
        "src/meds_model_base/dispatch.py",
        "src/meds_model_base/manifest.py",
        "src/meds_model_base/commands/base.py",
        "tests/test_smoke_pipeline.py",
    ]:
        assert (dst / rel).exists(), f"missing rendered file: {rel}"

    # Every command ships in every generated repo — the profile decides only which are *registered*.
    for command in COMMAND_CONFIGS:
        assert (dst / f"src/{slug}/configs/{command}.yaml").is_file(), f"missing config root: {command}"

    commands_txt = (dst / f"src/{slug}/commands.py").read_text()
    for command in commands:
        assert f"CommandName.{command}:" in commands_txt, f"{profile} commands.py missing {command}"
    for command in ALL_COMMANDS - commands:
        assert f"CommandName.{command}:" not in commands_txt, (
            f"{profile} commands.py unexpectedly registers {command}"
        )

    # Every rendered Python module must byte-compile (catches broken Jinja output without importing torch).
    assert compileall.compile_dir(str(dst / "src"), quiet=1, force=True), "rendered code failed to compile"


@pytest.mark.parametrize("profile", sorted(PROFILE_COMMANDS))
@pytest.mark.render
def test_rendered_configs_parse(tmp_path, profile):
    """Every rendered YAML config must be valid YAML — Jinja can produce plausible-looking garbage."""
    dst = tmp_path / profile
    slug = _render(dst, profile)
    configs = dst / "src" / slug / "configs"
    for fp in sorted(configs.rglob("*.yaml")):
        try:
            yaml.safe_load(fp.read_text())
        except yaml.YAMLError as e:  # pragma: no cover - failure path
            pytest.fail(f"{fp.relative_to(dst)} is not valid YAML: {e}")

    profile_cfg = yaml.safe_load((configs / "profile" / "default.yaml").read_text())
    assert profile_cfg["profile"]["name"] == profile


@pytest.mark.parametrize("profile", sorted(PROFILE_COMMANDS))
@pytest.mark.render
def test_rendered_repo_passes_ruff(tmp_path, profile):
    """The rendered repo must pass its *own* ruff config.

    This repo excludes ``template/`` from linting (it is not valid Python until rendered), so without this
    the vendored contract is never linted here at all — only in a generated repo's CI, one commit too late.
    It also pins the property the ``commands.py`` template is built around: a profile must not import a
    command class it does not register, which would be an F401.
    """
    ruff = shutil.which("ruff")
    if ruff is None:  # pragma: no cover - depends on the environment
        pytest.skip("ruff is not installed")

    dst = tmp_path / f"lint_{profile}"
    _render(dst, profile)
    result = subprocess.run([ruff, "check", "--no-cache", "."], cwd=dst, capture_output=True, text=True)
    assert result.returncode == 0, f"ruff failed on the rendered {profile} repo:\n{result.stdout}"


@pytest.mark.render
def test_answers_file_records_profile(tmp_path):
    slug = _render(tmp_path / "sup", "supervised_basic")
    answers = (tmp_path / "sup" / ".copier-answers.yml").read_text()
    assert "profile: supervised_basic" in answers
    assert f"model_slug: {slug}" in answers
