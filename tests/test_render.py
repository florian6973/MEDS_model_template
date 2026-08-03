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
import importlib.util
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]

ALL_COMMANDS = {
    "preprocess_data",
    "pretrain",
    "infer",
    "supervised_train",
    "predict",
}

#: Every DAG starts by materializing patient data. A task is not a stage: the commands that need one
#: take ``external_labels_dir`` directly, which is already what MEDS-DEV hands a model.
ALWAYS = {"preprocess_data"}

#: profile → the exact set of commands its ``commands.py`` should register. One entry per DAG in
#: docs/design-interface.md's "Supported pipeline chains".
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


def _render(dst: Path, profile: str, skip_tasks: bool = False, **answers) -> str:
    """Render ``profile`` into ``dst``; return the model slug. ``answers`` overrides other questions.

    ``skip_tasks`` suppresses the post-copy tasks, which is how a test asserts something about the
    *payload* rather than about the payload plus whatever `uvx ruff --fix` rewrote afterwards. See
    ``test_rendered_repo_is_clean_as_written``.
    """
    from copier import run_copy

    slug = f"m_{profile}"
    run_copy(
        str(REPO),
        str(dst),
        data={"model_slug": slug, "model_name": f"Demo {profile}", "profile": profile, **answers},
        defaults=True,
        unsafe=True,  # allow post-gen tasks (git init + best-effort `uvx ruff`, both no-ops if unavailable)
        skip_tasks=skip_tasks,
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
        # The agent-facing pair: the always-loaded contract summary, and the porting procedure it makes
        # non-optional. Both are only useful if they travel with the render — the knowledge they carry
        # lived in the template repo until they did, where a port never saw it.
        "CLAUDE.md",
        "docs/PORTING-A-MODEL.md",
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
        "tests/test_meds_dev_e2e.py",
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


@pytest.mark.parametrize("profile,commands", sorted(PROFILE_COMMANDS.items()))
@pytest.mark.render
def test_claude_md_states_the_same_chain(tmp_path, profile, commands):
    """``CLAUDE.md`` states the DAG a third time, so it is asserted equal like the other two.

    It is loaded into every agent session in a generated repo, which makes a stale chain there worse than
    a stale chain in a document nobody opens: it is the description the agent acts on instead of reading
    ``commands.py``.
    """
    dst = tmp_path / f"claude_{profile}"
    slug = _render(dst, profile)
    text = (dst / "CLAUDE.md").read_text()

    chain = text.split("This DAG's chain:\n\n", 1)[1].split("\n\n", 1)[0]
    assert {c for c in ALL_COMMANDS if f"`{c}`" in chain} == commands, (
        f"{profile}: the chain in CLAUDE.md disagrees with commands.py:\n{chain}"
    )

    # A path the agent cannot open is worse than no path: it reads as a file that was deleted.
    assert f"src/{slug}/model.py" in text
    assert (dst / "docs/PORTING-A-MODEL.md").is_file(), "CLAUDE.md links the porting procedure"


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

    Byte-compiling does not catch it — the syntax is valid — and this job deliberately installs no torch,
    so it cannot collect the rendered suite for real. The check therefore has to be structural. The
    `rendered-smoke` CI job does install and run it, but a whole install later.
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
def test_rendered_repo_is_clean_as_written(tmp_path, profile):
    """The rendered repo must pass its *own* ruff config **without** the post-copy tasks.

    This repo excludes ``template/`` from linting (it is not valid Python until rendered), so without this
    the vendored contract is never linted here at all — only in a generated repo's CI, one commit too late.

    ``skip_tasks`` is the load-bearing part. ``copier.yml`` runs a best-effort ``uvx ruff check --fix`` and
    ``uvx ruff format`` after a copy, so rendering normally and *then* linting checks a file ruff already
    repaired a second earlier — the test passes and the payload stays dirty. That is how four defects
    survived here: unsorted imports in ``train.py``, ``test_cli_smoke.py`` and ``test_smoke_pipeline.py``,
    and four unused imports in the ``packaged`` model stub. They are only visible where the post-copy task
    cannot run, which ``copier.yml`` explicitly supports ("skipped cleanly if offline") — and there a fresh
    repo fails its own CI and pre-commit on the first commit.

    ``format`` is asserted too, because ``uvx ruff format`` masks it the same way and Jinja produces
    formatting artefacts nothing else catches: a conditional paragraph inside a docstring leaves a
    ``\"\"\"Summary.\\n        \"\"\"`` that ruff collapses to one line in exactly the profiles where the
    branch is off.
    """
    ruff = shutil.which("ruff")
    if ruff is None:  # pragma: no cover - depends on the environment
        pytest.skip("ruff is not installed")

    dst = tmp_path / f"lint_{profile}"
    _render(dst, profile, skip_tasks=True)
    for args in (["check"], ["format", "--check"]):
        result = subprocess.run([ruff, *args, "--no-cache", "."], cwd=dst, capture_output=True, text=True)
        assert result.returncode == 0, (
            f"`ruff {args[0]}` failed on the rendered {profile} payload:\n{result.stdout}"
        )


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


@pytest.mark.parametrize("profile", sorted(PROFILE_COMMANDS))
@pytest.mark.render
def test_rendered_files_end_with_exactly_one_newline(tmp_path, profile):
    """A generated repo must survive its own pre-commit hooks on the very first commit.

    ``end-of-file-fixer`` rewrites any file not ending in exactly one newline, which turns `git commit`
    into a failure before a new user has done anything wrong. Jinja makes this easy to produce: with
    ``keep_trailing_newline``, a template whose last line is ``{% endif %}`` emits the branch's newline
    *and* the template file's own, so the rendered file ends with a blank line. That is how `model.yaml`,
    `.copier-answers.yml` and `configs/profile/default.yaml` all shipped with a trailing blank line.
    """
    dst = tmp_path / f"eof_{profile}"
    _render(dst, profile, skip_tasks=True)  # `uvx ruff format` would repair this before it was measured
    offenders = _bad_eof(dst)
    assert not offenders, f"{profile}: files not ending in exactly one newline: {offenders}"


def _bad_eof(dst: Path) -> list[str]:
    """Rendered files that do not end in exactly one newline (see the test above)."""
    # Tool caches are created by the post-copy `uvx ruff` task, not rendered from the payload; .gitignore
    # is what has to cover them, which `test_gitignore_covers_the_default_workspace` asserts.
    generated = {".git", "__pycache__", ".ruff_cache", ".pytest_cache", ".venv"}

    offenders = []
    for fp in sorted(dst.rglob("*")):
        if not fp.is_file() or generated.intersection(fp.parts):
            continue
        try:
            text = fp.read_text()
        except UnicodeDecodeError:  # pragma: no cover - the payload is all text today
            continue
        if text and not (text.endswith("\n") and not text.endswith("\n\n")):
            offenders.append(str(fp.relative_to(dst)))
    return offenders


#: ``use_<name>`` question → the logger group member it is responsible for rendering, and the
#: ``pyproject.toml`` extra that goes with it. The pairing is the contract: a question that adds a
#: dependency but no config buys nothing, which is what these two entries used to do.
OPTIONAL_LOGGERS = {"use_wandb": "wandb", "use_mlflow": "mlflow"}


@pytest.mark.parametrize("enabled", [False, True], ids=["disabled", "enabled"])
@pytest.mark.render
def test_optional_loggers_render_with_their_extra(tmp_path, enabled):
    """Answering yes to ``use_wandb`` / ``use_mlflow`` must render a config, not just a dependency.

    Both questions used to add a ``pyproject.toml`` extra and nothing else, so the option they advertised
    failed at composition — ``meds-model pretrain logger=wandb`` raised ``MissingConfigException`` on a
    repository that had answered yes and installed the package. The config and the extra are therefore
    asserted together, in both directions: rendered when enabled, and *absent* when not, so a name with no
    package behind it never becomes selectable either.
    """
    dst = tmp_path / f"loggers_{enabled}"
    slug = _render(dst, "supervised", **dict.fromkeys(OPTIONAL_LOGGERS, enabled))

    group = dst / f"src/{slug}/configs/logger"
    pyproject = (dst / "pyproject.toml").read_text()

    assert (group / "csv.yaml").is_file(), "csv needs no service or credentials: it always ships"
    for question, name in OPTIONAL_LOGGERS.items():
        config, extra = group / f"{name}.yaml", f'{name} = ["{name}"'
        assert config.is_file() is enabled, f"{question}={enabled} but {config.name} exists is {not enabled}"
        assert (extra in pyproject) is enabled, f"{question}={enabled} disagrees with the {name} extra"

    rendered = {p.name for p in group.iterdir()}
    expected = {"csv.yaml"} | ({f"{n}.yaml" for n in OPTIONAL_LOGGERS.values()} if enabled else set())
    assert rendered == expected, f"unexpected logger group contents: {sorted(rendered)}"

    # Both answers also gate a branch in README.md.jinja, and the per-profile newline test above only ever
    # renders the default (disabled) one.
    assert not _bad_eof(dst), f"files not ending in exactly one newline: {_bad_eof(dst)}"


@pytest.mark.render
def test_logger_group_members_nest_under_their_own_name(tmp_path):
    """Each logger config must be a mapping of *name → logger*, not a bare ``_target_``.

    ``build_trainer`` calls ``instantiate_group``, which instantiates every ``_target_`` **child** of
    ``cfg.logger``. A config whose ``_target_`` sits at the top level therefore contributes no logger at
    all — and because ``instantiate_group`` returns ``[]`` for anything it does not recognise, the Trainer
    is simply built with ``logger=False`` and the run succeeds having recorded nothing. That is precisely
    the shape these files are adapted from (MEDS-EIC-AR nests its loggers one level deeper, under
    ``trainer/logger/``), so it is the mistake a copy is most likely to reintroduce.
    """
    dst = tmp_path / "logger_shape"
    slug = _render(dst, "supervised", use_wandb=True, use_mlflow=True)

    for fp in sorted((dst / f"src/{slug}/configs/logger").glob("*.yaml")):
        cfg = yaml.safe_load(fp.read_text())
        assert "_target_" not in cfg, (
            f"{fp.name}: `_target_` at the top level of a logger config builds no logger — nest it under "
            f"a `{fp.stem}:` key, as instantiate_group expects"
        )
        assert set(cfg) == {fp.stem}, f"{fp.name} should declare exactly one logger, named {fp.stem}"
        assert cfg[fp.stem]["_target_"].startswith("lightning.pytorch.loggers."), (
            f"{fp.name}: a logger group member must target a Lightning logger"
        )


@pytest.mark.render
def test_gitignore_covers_the_default_workspace(tmp_path):
    """``env.sh`` points ``OUTPUT_DIR`` at ``./runs`` and the README writes artifacts there.

    Left untracked, a first `git add -A` sweeps in a whole training workspace — checkpoints, tensorized
    data, predictions — which is how a fresh repo ends up trying to commit hundreds of megabytes.
    """
    dst = tmp_path / "ignore"
    _render(dst, "supervised")

    ignored = (dst / ".gitignore").read_text().split()
    assert "runs/" in ignored, ".gitignore must exclude runs/"
    assert "runs" in (dst / "env.sh").read_text(), "env.sh should still default OUTPUT_DIR to ./runs"

    # `uvx ruff` runs as a post-copy task, so a fresh repo has a .ruff_cache before the user touches it.
    assert ".ruff_cache/" in ignored, ".gitignore must exclude the cache the post-copy task creates"


@pytest.mark.render
def test_meds_dev_helper_writes_where_the_loader_looks(tmp_path):
    """``meds-model-add-to-meds-dev`` must place files where MEDS-DEV's discovery actually finds them.

    MEDS-DEV registers models by walking ``files("MEDS_DEV.models").rglob("*/model.yaml")`` and keying each
    on ``path.relative_to(models_root).parent``. That expression is reproduced here rather than described,
    so a layout change on either side fails the test instead of producing a model MEDS-DEV silently never
    lists. The requirements rewrite is checked too: it is the only file that is *not* a copy, and getting
    it wrong fails inside MEDS-DEV's isolated venv with a bare ``meds-model: command not found``.
    """
    if importlib.util.find_spec("MEDS_DEV") is not None:  # pragma: no cover - depends on the environment
        pytest.skip("MEDS_DEV is importable here, so the helper's registration check governs the outcome")

    dst = tmp_path / "meds_dev_repo"
    _render(dst, "supervised")

    models_root = tmp_path / "MEDS-DEV" / "src/MEDS_DEV/models"
    models_root.mkdir(parents=True)

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "meds_model_base.meds_dev",
            "--meds-dev",
            str(tmp_path / "MEDS-DEV"),
            "--repo",
            str(dst),
        ],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(dst / "src")},
    )
    assert result.returncode == 0, f"helper failed:\n{result.stdout}\n{result.stderr}"

    found = list(models_root.rglob("*/model.yaml"))
    assert len(found) == 1, f"MEDS-DEV's discovery glob found {len(found)} models, expected 1"
    assert found[0].relative_to(models_root).parent.as_posix() == "m_supervised"

    model_dir = found[0].parent
    assert yaml.safe_load(found[0].read_text()) == yaml.safe_load((dst / "model.yaml").read_text())
    assert (model_dir / "README.md").is_file(), "MEDS-DEV's contribution guide requires a README.md"

    requirements = (model_dir / "requirements.txt").read_text()
    assert f"-e {dst.resolve()}" in requirements, "a local checkout must be referenced by absolute path"
    assert "-e .\n" not in requirements, "a relative editable install would install MEDS-DEV instead"


@pytest.mark.render
def test_meds_dev_helper_is_not_a_command(tmp_path):
    """The helper is packaging tooling: an entry point, never a sixth ``CommandName``."""
    dst = tmp_path / "meds_dev_entry"
    slug = _render(dst, "supervised")

    pyproject = (dst / "pyproject.toml").read_text()
    assert '"meds-model-add-to-meds-dev" = "meds_model_base.meds_dev:main"' in pyproject

    base = (REPO / "template/src/meds_model_base/commands/base.py").read_text()
    assert "meds_dev" not in base, "the MEDS-DEV helper must not leak into the command contract"
    body = base.split("class CommandName(StrEnum):", 1)[1].split("\nclass ", 1)[0]
    assert set(re.findall(r'^    (\w+) = "\1"$', body, re.M)) == ALL_COMMANDS
    assert "meds_dev" not in (dst / f"src/{slug}/commands.py").read_text()


@pytest.mark.render
def test_answers_file_records_profile(tmp_path):
    slug = _render(tmp_path / "sup", "supervised")
    answers = (tmp_path / "sup" / ".copier-answers.yml").read_text()
    assert "profile: supervised" in answers
    assert f"model_slug: {slug}" in answers
