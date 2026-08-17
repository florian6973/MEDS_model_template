"""Contract-level tests for the minimal MEDS model Copier template."""

from __future__ import annotations

import compileall
import importlib.util
import os
import subprocess
import sys
import types
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
TEMPLATE_REPO = REPO

DAGS = {
    "supervised": ["preprocess_data", "supervised_train", "predict"],
    "finetune": ["preprocess_data", "pretrain", "supervised_train", "predict"],
    "probe": ["preprocess_data", "pretrain", "infer", "supervised_train", "predict"],
    "zero_shot_direct": ["preprocess_data", "pretrain", "predict"],
    "zero_shot_materialized": ["preprocess_data", "pretrain", "infer", "predict"],
    "packaged": ["preprocess_data", "predict"],
}


def render(dst: Path, profile: str, *, uses_predicates: bool = False) -> str:
    from copier import run_copy

    slug = f"v2_{profile}"
    run_copy(
        str(TEMPLATE_REPO),
        str(dst),
        data={
            "model_name": f"V2 {profile}",
            "model_slug": slug,
            "profile": profile,
            "uses_predicates": uses_predicates,
        },
        defaults=True,
        unsafe=True,
        quiet=True,
    )
    return slug


def load_integration_module(rendered: Path):
    path = rendered / "tests" / "integration.py"
    spec = importlib.util.spec_from_file_location(f"rendered_integration_{id(rendered)}", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("profile", DAGS)
def test_renders_six_minimal_dags(tmp_path, profile):
    dst = tmp_path / profile
    slug = render(dst, profile, uses_predicates=True)
    required = [
        "model.yaml",
        ".copier-answers.yml",
        "AGENTS.md",
        "SPEC.md",
        "IMPLEMENTATION_REPORT.md",
        "docs/IMPLEMENTATION_GUIDE.md",
        "requirements.txt",
        "slurm/config.sh",
        "slurm/job.sbatch",
        "slurm/submit.sh",
        "scripts/github-sync.sh",
        "tests/test_end_to_end.py",
        "tests/test_primary_behavior.py",
        "tests/signal.py",
        "tests/test_meds_dev_e2e.py",
        "tests/test_mimic_demo_e2e.py",
        "tests/e2e.py",
        "src/meds_model_contract/contract.py",
        "src/meds_model_contract/manifest.py",
        f"src/{slug}/commands.py",
    ]
    assert all((dst / path).is_file() for path in required)
    assert compileall.compile_dir(dst, quiet=1, force=True)
    assert yaml.safe_load((dst / "model.yaml").read_text())["commands"]["supervised"]["predict"]
    assert "{predicates_path}" in (dst / "model.yaml").read_text()

    env = {**os.environ, "PYTHONPATH": str(dst / "src")}
    command_list = subprocess.run(
        ["python", "-m", slug, "commands"], env=env, text=True, capture_output=True, check=True
    ).stdout.splitlines()
    assert command_list == DAGS[profile]

    syntax = subprocess.run(
        ["bash", "-n", "slurm/config.sh", "slurm/job.sbatch", "slurm/submit.sh", "scripts/github-sync.sh"],
        cwd=dst,
        text=True,
        capture_output=True,
    )
    assert syntax.returncode == 0, syntax.stderr


def test_copier_answers_record_portable_update_metadata(tmp_path):
    dst = tmp_path / "answers"
    render(dst, "supervised")
    answers = yaml.safe_load((dst / ".copier-answers.yml").read_text())
    assert answers["_src_path"]
    assert answers["model_slug"] == "v2_supervised"
    assert answers["profile"] == "supervised"


def test_mimic_task_prepends_pinned_meds_dev_venv_only_for_task(tmp_path, monkeypatch):
    dst = tmp_path / "mimic-path"
    render(dst, "supervised")
    calls = []

    integration = types.ModuleType("rendered_tests.integration")
    integration.RUN_TIMEOUT = 123
    integration.venv_bin = lambda checkout: checkout / ".venv" / "bin"
    integration.run = lambda command, **kwargs: calls.append(([str(value) for value in command], kwargs))
    package = types.ModuleType("rendered_tests")
    package.__path__ = [str(dst / "tests")]
    monkeypatch.setitem(sys.modules, "rendered_tests", package)
    monkeypatch.setitem(sys.modules, "rendered_tests.integration", integration)

    path = dst / "tests" / "e2e.py"
    spec = importlib.util.spec_from_file_location("rendered_tests.e2e", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    meds_dev = tmp_path / "MEDS-DEV"
    monkeypatch.delenv("MEDS_DEMO_DIR", raising=False)
    monkeypatch.setenv("PATH", "/ambient/bin")
    module.prepare_mimic_demo(meds_dev, tmp_path)

    assert len(calls) == 2
    dataset_command, dataset_kwargs = calls[0]
    task_command, task_kwargs = calls[1]
    assert dataset_command[0] == str(meds_dev / ".venv/bin/meds-dev-dataset")
    assert "env" not in dataset_kwargs
    assert task_command[0] == str(meds_dev / ".venv/bin/meds-dev-task")
    assert task_kwargs["env"]["PATH"] == f"{meds_dev / '.venv/bin'}{os.pathsep}/ambient/bin"


def test_example_signal_dataset_has_no_length_or_position_leak(tmp_path):
    import polars as pl

    dst = tmp_path / "signal-example"
    render(dst, "supervised")
    path = dst / "tests" / "signal.py"
    spec = importlib.util.spec_from_file_location("rendered_signal", path)
    signal = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(signal)

    root = signal.build_signal_dataset(tmp_path / "cohort", n_train=40, n_tuning=10, n_held_out=10)
    events = pl.scan_parquet(root / "data" / "*" / "*.parquet").collect()
    labels = pl.concat(
        [pl.read_parquet(path) for path in sorted((root / "task_labels/signal_task").glob("*.parquet"))]
    )
    observed = events.group_by("subject_id").agg(
        pl.len().alias("length"),
        (pl.col("code") == signal.SIGNAL_CODE).any().alias("has_signal"),
    )
    checked = labels.join(observed, on="subject_id")
    assert checked["boolean_value"].to_list() == checked["has_signal"].to_list()
    assert checked["length"].min() >= 5
    assert checked["length"].max() <= 10
    assert checked.filter(pl.col("boolean_value")).height > 0
    assert checked.filter(~pl.col("boolean_value")).height > 0


def test_primary_behavior_is_a_required_failing_placeholder(tmp_path):
    dst = tmp_path / "primary-behavior"
    render(dst, "supervised")
    test = (dst / "tests/test_primary_behavior.py").read_text()
    assert "pytest.fail" in test
    assert "shuffled-label negative control" in test
    assert "selected codes, vocabulary, predicates, or derived features" in test


def test_predicates_are_external_not_a_rendered_semantic_format(tmp_path):
    dst = tmp_path / "without-predicates"
    render(dst, "supervised", uses_predicates=False)
    assert not (dst / "semantics.yaml").exists()
    assert "external_predicates_file" not in (dst / "model.yaml").read_text()


def test_implementation_prompt_belongs_to_template_readme_not_rendered_repo(tmp_path):
    dst = tmp_path / "prompt-location"
    render(dst, "supervised")
    prompt_token = "<PAPER_PATH_OR_URL_AND_OR_REPOSITORY_PATH_OR_URL>"
    assert prompt_token in (TEMPLATE_REPO / "README.md").read_text()
    assert prompt_token not in (dst / "README.md").read_text()
    assert "Copyable model-implementation prompt" not in (dst / "README.md").read_text()


def test_spec_has_one_template_source_and_renders_to_root(tmp_path):
    dst = tmp_path / "single-spec"
    render(dst, "supervised")
    assert not (TEMPLATE_REPO / "SPEC.md").exists()
    assert (dst / "SPEC.md").read_text() == (TEMPLATE_REPO / "template" / "SPEC.md").read_text()


@pytest.mark.parametrize(
    ("prediction_rows", "error"),
    [
        ([(2, "2020-01-02"), (1, "2020-01-01")], None),
        ([(1, "2020-01-01")], "missing"),
        ([(1, "2020-01-01"), (2, "2020-01-02"), (3, "2020-01-03")], "extra"),
        ([(1, "2020-01-01"), (1, "2020-01-01"), (2, "2020-01-02")], "duplicate"),
    ],
)
def test_prediction_keys_are_compared_directly(tmp_path, prediction_rows, error):
    import polars as pl

    dst = tmp_path / "key-validation"
    render(dst, "supervised")
    integration = load_integration_module(dst)
    labels = tmp_path / "labels" / "held_out"
    predictions = tmp_path / "predictions"
    labels.mkdir(parents=True)
    predictions.mkdir()

    def frame(rows):
        return pl.DataFrame(
            {
                "subject_id": [row[0] for row in rows],
                "prediction_time": [row[1] for row in rows],
            }
        ).with_columns(pl.col("prediction_time").str.to_datetime())

    frame([(1, "2020-01-01"), (2, "2020-01-02")]).write_parquet(labels / "part.parquet")
    frame(prediction_rows).write_parquet(predictions / "predictions.parquet")
    if error is None:
        integration.assert_prediction_keys(predictions, labels.parent, ["held_out"])
    else:
        with pytest.raises(AssertionError, match=error):
            integration.assert_prediction_keys(predictions, labels.parent, ["held_out"])


def test_meds_dev_predicates_capability_is_explicit(tmp_path):
    dst = tmp_path / "capability"
    render(dst, "supervised")
    integration = load_integration_module(dst)
    checkout = tmp_path / "MEDS-DEV"
    config = checkout / "src/MEDS_DEV/configs/_run_model.yaml"
    implementation = checkout / "src/MEDS_DEV/models/__init__.py"
    config.parent.mkdir(parents=True)
    implementation.parent.mkdir(parents=True)
    config.write_text("predicates_path: null\n")
    implementation.write_text('format_kwargs["predicates_path"] = str(cfg.predicates_path)\n')
    integration.assert_meds_dev_predicates_capability(checkout)
    implementation.write_text("format_kwargs = {}\n")
    with pytest.raises(AssertionError, match="PR #325"):
        integration.assert_meds_dev_predicates_capability(checkout)


def test_meds_dev_ref_is_checked_out_in_clone(tmp_path, monkeypatch):
    dst = tmp_path / "provisioning"
    render(dst, "supervised")
    integration = load_integration_module(dst)
    commands = []

    class Request:
        class Config:
            @staticmethod
            def getoption(name, default=""):
                assert name == "markexpr"
                return "meds_dev"

        config = Config()

    class TempPathFactory:
        @staticmethod
        def mktemp(name):
            assert name == "meds-dev"
            path = tmp_path / name
            path.mkdir()
            return path

    def fake_run(command, **kwargs):
        command = [str(value) for value in command]
        commands.append(command)
        if command[1] == "clone":
            checkout = Path(command[-1])
            config = checkout / "src/MEDS_DEV/configs/_run_model.yaml"
            implementation = checkout / "src/MEDS_DEV/models/__init__.py"
            config.parent.mkdir(parents=True)
            implementation.parent.mkdir(parents=True)
            config.write_text("predicates_path: null\n")
            implementation.write_text('format_kwargs["predicates_path"] = str(cfg.predicates_path)\n')

    monkeypatch.delenv("MEDS_DEV_DIR", raising=False)
    monkeypatch.setenv("MEDS_DEV_REF", "test-predicates-ref")
    monkeypatch.setattr(integration.shutil, "which", lambda command: f"/usr/bin/{command}")
    monkeypatch.setattr(integration, "run", fake_run)
    checkout = integration.provision_meds_dev(Request(), TempPathFactory())
    assert [
        "/usr/bin/git",
        "-C",
        str(checkout),
        "checkout",
        "--detach",
        "test-predicates-ref",
    ] in commands


def test_measurement_manifest_is_plausible(tmp_path, monkeypatch):
    dst = tmp_path / "measurement"
    render(dst, "packaged")
    monkeypatch.syspath_prepend(str(dst / "src"))
    from meds_model_contract import measured_artifact, read_manifest, write_run_result
    from meds_model_contract.meds_dev import register

    artifact = tmp_path / "artifact"
    with measured_artifact(
        artifact,
        artifact_type="patients",
        command="preprocess_data",
        producer="test",
        payload_format="test/v1",
    ) as staging:
        (staging / "payload.txt").write_text("ok")
    manifest = read_manifest(artifact)
    assert manifest["resources"]["wall_seconds"] >= 0
    memory = manifest["resources"]["memory"]
    assert memory["peak_process_tree_rss_bytes"] > 0
    if memory["source"] == "linux_proc_smaps_rollup":
        assert memory["peak_process_tree_pss_bytes"] > 0
        assert memory["includes_descendants"] is True
    assert manifest["resources"]["gpu"]["peak_allocated_bytes"] is None
    assert "semantics_digest" in manifest
    assert manifest["semantics_digest"] is None

    run_dir = tmp_path / "run"
    result = write_run_result(
        run_dir,
        repo=dst,
        artifacts={"preprocess_data": artifact},
        dataset={"name": "synthetic", "subjects": 1, "events": 1},
        task={"name": "smoke", "timepoints": 1},
    )
    assert result.is_file()
    assert (run_dir / "spec.yaml").is_file()
    assert (run_dir / "manifests/preprocess_data.yaml").is_file()

    meds_dev = tmp_path / "MEDS-DEV"
    (meds_dev / "src/MEDS_DEV/models").mkdir(parents=True)
    registered = register(dst, meds_dev, "measurement-model")
    assert (registered / "model.yaml").is_file()
    assert str(dst.resolve()) in (registered / "requirements.txt").read_text()
