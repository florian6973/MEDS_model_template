"""Template-level tests: render every DAG and verify its structure.

These test the *template*, never a model — a generated repository ships no model at all, so there is
nothing here to train. What is checked is that each profile produces a coherent, lintable, self-consistent
DAG: the right commands registered, the declared chain matching them, every artifact a command consumes
actually produced by another command in the same chain, and no command class left unreachable.

Fast and dependency-light: no torch, a few seconds.
"""

from __future__ import annotations

import ast
import compileall
import re
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

#: Every DAG starts by materializing patient data and a task.
ALWAYS = {"preprocess_data", "preprocess_task"}

#: profile → the exact set of commands its ``commands.py`` should register. One entry per DAG in
#: design-interface.md's "Supported pipeline chains".
PROFILE_COMMANDS = {
    "supervised": ALWAYS | {"supervised_train", "predict"},
    "finetune": ALWAYS | {"pretrain", "supervised_train", "predict"},
    "probe": ALWAYS | {"pretrain", "infer", "supervised_train", "predict"},
    "zero_shot_direct": ALWAYS | {"pretrain", "predict"},
    "zero_shot_materialized": ALWAYS | {"pretrain", "infer", "predict"},
    "packaged": ALWAYS | {"predict"},
}

#: Which command produces the artifact each ``*_dir`` / ``*_subdir`` source refers to.
SOURCE_PRODUCER = {
    "input_pretrained_model_dir": "pretrain",
    "input_supervised_model_dir": "supervised_train",
    "input_inference_subdir": "infer",
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


def _registry(dst: Path, slug: str) -> dict[str, str]:
    """Parse ``commands.py`` into ``{command: class}`` without importing torch."""
    text = (dst / f"src/{slug}/commands.py").read_text()
    return dict(re.findall(r"CommandName\.(\w+): (\w+),", text))


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

    # Every command ships in every generated repo — the DAG decides only which are *registered*.
    for command in COMMAND_CONFIGS:
        assert (dst / f"src/{slug}/configs/{command}.yaml").is_file(), f"missing config root: {command}"

    assert set(_registry(dst, slug)) == commands, f"{profile} registers the wrong commands"

    # Model implementations are the user's, so the contract must ship none of its own.
    assert not (dst / "src/meds_model_base/profiles").exists()

    assert compileall.compile_dir(str(dst / "src"), quiet=1, force=True), "rendered code failed to compile"


@pytest.mark.parametrize("profile", sorted(PROFILE_COMMANDS))
@pytest.mark.render
def test_model_is_a_stub(tmp_path, profile):
    """A generated repo ships no model: every hook raises, and the stub marker is set.

    This is the template's core promise. If a real implementation ever leaks back into the payload, the
    ``is_stub`` marker disappears and the rendered conformance tests silently start running against
    someone else's model.
    """
    dst = tmp_path / f"stub_{profile}"
    slug = _render(dst, profile)
    model_py = (dst / f"src/{slug}/model.py").read_text()
    predict_py = dst / f"src/{slug}/predict.py"
    user_code = model_py + (predict_py.read_text() if predict_py.is_file() else "")

    assert "is_stub = True" in model_py, f"{profile}: model.py must declare the stub marker"
    # `packaged` ships weights rather than a trainable module, so its unimplemented hook is in predict.py.
    assert "NotImplementedError" in user_code, f"{profile}: the generated hooks must be left unimplemented"

    # A stub cannot carry hyperparameters for an architecture the template did not choose.
    model_cfg = yaml.safe_load((dst / f"src/{slug}/configs/model/default.yaml").read_text())
    assert model_cfg["_target_"] == f"{slug}.model.Model"


@pytest.mark.parametrize("profile,commands", sorted(PROFILE_COMMANDS.items()))
@pytest.mark.render
def test_declared_chain_matches_registry(tmp_path, profile, commands):
    """``configs/profile/default.yaml`` states the DAG in YAML and ``commands.py`` states it in Python.

    They are two descriptions of one thing, so they are asserted equal rather than trusted to agree.
    """
    dst = tmp_path / f"chain_{profile}"
    slug = _render(dst, profile)
    cfg = yaml.safe_load((dst / f"src/{slug}/configs/profile/default.yaml").read_text())

    assert cfg["profile"]["name"] == profile
    assert set(cfg["profile"]["chain"]) == set(_registry(dst, slug)) == commands


@pytest.mark.parametrize("profile", sorted(PROFILE_COMMANDS))
@pytest.mark.render
def test_every_consumed_artifact_is_produced(tmp_path, profile):
    """A DAG must be runnable: everything required is produced, and everything produced is consumed.

    Two distinct failures, both of which type-check and lint perfectly:

    - a command that *requires* a source no registered command produces — the chain simply cannot run;
    - a command that produces an artifact nothing consumes — a dangling ``infer`` burns GPU time writing
      a dead end, and every other test still passes.

    Note the distinction between *can* and *must* consume. ``supervised_train`` accepts a pretrained model
    but does not require one, so a DAG without ``pretrain`` is perfectly valid; only ``require_source``
    commands impose an obligation.
    """
    dst = tmp_path / f"dag_{profile}"
    slug = _render(dst, profile)
    registry = _registry(dst, slug)
    classes = _class_table(dst, slug)

    for command, cls in registry.items():
        spec = _resolve(classes, cls)
        if not spec["require_source"] or spec["packaged_model"]:
            continue  # optional sources, or weights that ship with the repo
        producible = {s for s in spec["supported_sources"] if SOURCE_PRODUCER[s] in registry}
        assert producible, (
            f"{profile}: `{command}` requires one of {sorted(spec['supported_sources'])}, but no registered "
            f"command produces any of them"
        )

    consumable = set()
    for cls in registry.values():
        consumable |= _resolve(classes, cls)["supported_sources"]
    for source, producer in SOURCE_PRODUCER.items():
        # `pretrain` and `supervised_train` also produce the artifact the DAG is *for*, so they are never
        # dead ends. `infer` exists only to be consumed.
        if producer == "infer" and producer in registry:
            assert source in consumable, (
                f"{profile}: `infer` is registered but nothing consumes {source} — the artifact it writes "
                "is a dead end"
            )


def _class_table(dst: Path, slug: str) -> dict[str, dict]:
    """Map class name → its declared command spec, parsed with ``ast`` (no torch import).

    Regex cannot do this: ``supported_sources`` is declared in some classes and inherited in others, and a
    pattern loose enough to span a class body is loose enough to match the *next* class's declaration.
    """
    files = sorted((dst / "src/meds_model_base/commands").glob("*.py"))
    user_predict = dst / f"src/{slug}/predict.py"
    if user_predict.is_file():
        files.append(user_predict)

    table: dict[str, dict] = {}
    for fp in files:
        for node in ast.walk(ast.parse(fp.read_text())):
            if not isinstance(node, ast.ClassDef):
                continue
            spec: dict = {
                "bases": [b.id for b in node.bases if isinstance(b, ast.Name)],
                "sources": None,
                "supported_sources": None,
                "require_source": None,
                "packaged_model": None,
            }
            for stmt in node.body:
                name, value = _assignment(stmt)
                if name in ("sources", "supported_sources"):
                    spec[name] = _string_set(value)
                elif name in ("require_source", "packaged_model"):
                    spec[name] = value.value if isinstance(value, ast.Constant) else None
            table[node.name] = spec
    return table


def _assignment(stmt) -> tuple[str | None, object]:
    """Return ``(name, value_node)`` for a simple class-body assignment, else ``(None, None)``."""
    if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
        return stmt.target.id, stmt.value
    if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name):
        return stmt.targets[0].id, stmt.value
    return None, None


def _string_set(node) -> set[str] | None:
    """Extract the string literals from a tuple, set, or ``frozenset({...})`` node."""
    if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "frozenset":
        return _string_set(node.args[0]) if node.args else set()
    if isinstance(node, ast.Tuple | ast.Set | ast.List):
        return {e.value for e in node.elts if isinstance(e, ast.Constant)}
    return None


def _resolve(table: dict[str, dict], cls: str) -> dict:
    """Resolve a class's effective spec by walking its bases for the nearest declaration of each field."""
    resolved = {"supported_sources": None, "sources": None, "require_source": False, "packaged_model": None}
    for field in resolved:
        for name in _lineage(table, cls):
            value = table.get(name, {}).get(field)
            if value is not None:
                resolved[field] = value
                break
    # `supported_sources = None` means "every source this command's interface defines".
    if resolved["supported_sources"] is None:
        resolved["supported_sources"] = resolved["sources"] or set()
    return resolved


def _lineage(table: dict[str, dict], cls: str) -> list[str]:
    """``cls`` followed by its bases, depth-first — close enough to an MRO for single-declaration lookup."""
    order, stack = [], [cls]
    while stack:
        name = stack.pop(0)
        if name in order:
            continue
        order.append(name)
        stack.extend(table.get(name, {}).get("bases", []))
    return order


@pytest.mark.render
def test_every_command_class_is_reachable():
    """Every default command class the contract exports is registered by at least one DAG.

    A class nobody registers is dead code that still has to be maintained and still looks supported in the
    docs. This is what would have flagged `MaterializedPredictCommand` the day it was written.
    """
    exported = (REPO / "template/src/meds_model_base/commands/__init__.py").read_text()
    defaults = set(re.findall(r'"(\w+)": "\w+",', exported))
    # Abstract bases are extension points, not implementations: they are meant to be subclassed by a repo.
    abstract = {"ZeroShotPredictCommand", "PackagedPredictCommand"}

    entries = (REPO / "template/src/{{ model_slug }}/commands.py.jinja").read_text()
    registered = set(re.findall(r"'\w+', '(\w+)'", entries))

    unreachable = defaults - abstract - registered
    assert not unreachable, f"command classes no profile registers: {sorted(unreachable)}"


@pytest.mark.parametrize("profile", sorted(PROFILE_COMMANDS))
@pytest.mark.render
def test_rendered_tests_are_importable(tmp_path, profile):
    """Rendered test modules must be importable as top-level modules.

    ``tests/`` has no ``__init__.py``, so pytest imports each module top-level and a relative import raises
    ``ImportError: attempted relative import with no known parent package``. That happens at *collection*
    time, which is what makes it nasty: it fires before ``-m "not slow"`` can deselect anything, so one bad
    import in a slow-marked module takes down the entire run.

    Byte-compiling does not catch it — the syntax is valid — and this repo cannot run the rendered suite
    (it needs torch), so the check has to be structural.
    """
    dst = tmp_path / f"imports_{profile}"
    _render(dst, profile)

    for fp in sorted((dst / "tests").glob("*.py")):
        for node in ast.walk(ast.parse(fp.read_text())):
            if isinstance(node, ast.ImportFrom) and node.level:
                pytest.fail(
                    f"{profile}: tests/{fp.name} line {node.lineno} uses a relative import "
                    f"(`from {'.' * node.level}{node.module or ''} import ...`). tests/ is not a package; "
                    "import from `meds_model_base.testing` instead."
                )


@pytest.mark.parametrize("profile", sorted(PROFILE_COMMANDS))
@pytest.mark.render
def test_rendered_repo_passes_ruff(tmp_path, profile):
    """The rendered repo must pass its *own* ruff config.

    This repo excludes ``template/`` from linting (it is not valid Python until rendered), so without this
    the vendored contract is never linted here at all — only in a generated repo's CI, one commit too late.
    """
    ruff = shutil.which("ruff")
    if ruff is None:  # pragma: no cover - depends on the environment
        pytest.skip("ruff is not installed")

    dst = tmp_path / f"lint_{profile}"
    _render(dst, profile)
    result = subprocess.run([ruff, "check", "--no-cache", "."], cwd=dst, capture_output=True, text=True)
    assert result.returncode == 0, f"ruff failed on the rendered {profile} repo:\n{result.stdout}"


@pytest.mark.parametrize("profile", sorted(PROFILE_COMMANDS))
@pytest.mark.render
def test_rendered_configs_parse(tmp_path, profile):
    """Every rendered YAML config must be valid YAML — Jinja can produce plausible-looking garbage."""
    dst = tmp_path / f"yaml_{profile}"
    slug = _render(dst, profile)
    for fp in sorted((dst / "src" / slug / "configs").rglob("*.yaml")):
        try:
            yaml.safe_load(fp.read_text())
        except yaml.YAMLError as e:  # pragma: no cover - failure path
            pytest.fail(f"{fp.relative_to(dst)} is not valid YAML: {e}")


@pytest.mark.render
def test_answers_file_records_profile(tmp_path):
    slug = _render(tmp_path / "sup", "supervised")
    answers = (tmp_path / "sup" / ".copier-answers.yml").read_text()
    assert "profile: supervised" in answers
    assert f"model_slug: {slug}" in answers
