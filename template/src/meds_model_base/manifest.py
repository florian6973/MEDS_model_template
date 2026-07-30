"""Artifact manifests: provenance that commands *read*, not just write.

Every artifact directory this template produces carries a ``manifest.yaml`` describing what it is, what
produced it, and which artifacts it was built from. The point is validation: a command reads the manifests
of its inputs and rejects a mismatch **before** doing any work, so pointing ``predict`` at an inference
directory of the wrong kind fails in a second rather than after an hour on a GPU.

Two conventions make that safe:

- **Atomic publication.** :func:`write_artifact` stages into a temporary sibling directory and renames it
  into place, so a visible artifact directory is always complete. There is no window in which a reader can
  observe a half-written artifact, and no partial directory to clean up after a crash.
- **No aggregate manifest.** Each artifact describes only itself. The state of a ``data_dir`` is derived by
  scanning ``tasks/*/manifest.yaml`` and ``inference/*/manifest.yaml``. A root manifest would have to be
  rewritten on every append, which races as soon as two jobs materialize different tasks concurrently.

Dependency-light: yaml + stdlib only, so the introspection paths stay cheap.
"""

from __future__ import annotations

import hashlib
import logging
import os
import shutil
import subprocess
import tempfile
from contextlib import contextmanager
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

#: Bumped when the manifest layout changes incompatibly.
MANIFEST_VERSION = 1

MANIFEST_FILENAME = "manifest.yaml"

#: Packages whose versions are worth recording for reproducibility.
_TRACKED_PACKAGES = (
    "meds",
    "meds-torch-data",
    "MEDS-transforms",
    "meds-evaluation",
    "es-aces",
    "torch",
    "lightning",
    "polars",
)


class ArtifactType(StrEnum):
    """The kinds of artifact directory the command graph produces.

    Examples:
        >>> ArtifactType("task") is ArtifactType.task
        True
        >>> [a.value for a in ArtifactType]
        ['data', 'task', 'inference', 'pretrained_model', 'supervised_model', 'predictions']
    """

    data = "data"
    task = "task"
    inference = "inference"
    pretrained_model = "pretrained_model"
    supervised_model = "supervised_model"
    predictions = "predictions"


class InferenceKind(StrEnum):
    """What an ``infer`` run materialized, recorded in the inference manifest.

    Consumers validate against this: a probe expects ``embeddings``, a materialized zero-shot ``predict``
    expects ``trajectories`` or ``scores``.

    Examples:
        >>> InferenceKind.embeddings.value
        'embeddings'
    """

    embeddings = "embeddings"
    trajectories = "trajectories"
    hazards = "hazards"
    scores = "scores"
    token_probabilities = "token_probabilities"


class ManifestError(RuntimeError):
    """Raised when an input artifact is missing, malformed, or of the wrong type/kind."""


# ------------------------------------------------------------------------------------------------------
# Provenance capture
# ------------------------------------------------------------------------------------------------------


def _package_versions() -> dict[str, str]:
    """Best-effort version lookup for the tracked ecosystem packages (absent ones are omitted)."""
    from importlib.metadata import PackageNotFoundError, version

    out: dict[str, str] = {}
    for name in _TRACKED_PACKAGES:
        try:
            out[name] = version(name)
        except PackageNotFoundError:  # pragma: no cover - depends on the install
            continue
    return out


def _git_provenance(start: Path | None = None) -> dict[str, Any] | None:
    """Best-effort ``{commit, dirty}`` for the repo containing ``start`` (None if not a git checkout)."""
    cwd = str(start or Path.cwd())
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=cwd, capture_output=True, text=True, check=False
        )
        if commit.returncode != 0:
            return None
        status = subprocess.run(
            ["git", "status", "--porcelain"], cwd=cwd, capture_output=True, text=True, check=False
        )
        return {"commit": commit.stdout.strip(), "dirty": bool(status.stdout.strip())}
    except (OSError, subprocess.SubprocessError):  # pragma: no cover - git missing
        return None


def _template_commit(start: Path | None = None) -> str | None:
    """The template revision from ``.copier-answers.yml`` (``_commit``), searching upward from ``start``."""
    here = (start or Path.cwd()).resolve()
    for candidate in (here, *here.parents):
        answers = candidate / ".copier-answers.yml"
        if answers.is_file():
            try:
                data = yaml.safe_load(answers.read_text()) or {}
            except yaml.YAMLError:  # pragma: no cover - malformed answers file
                return None
            return data.get("_commit")
    return None


def _model_package() -> dict[str, str]:
    """Identify the installed model package (the distribution providing the ``meds-model`` script)."""
    from importlib.metadata import PackageNotFoundError, distributions, version

    for dist in distributions():
        entry_points = getattr(dist, "entry_points", [])
        if any(ep.name == "meds-model" for ep in entry_points):
            name = dist.metadata["Name"]
            try:
                return {"name": name, "version": version(name)}
            except PackageNotFoundError:  # pragma: no cover
                break
    return {"name": "unknown", "version": "unknown"}


def build_provenance() -> dict[str, Any]:
    """Assemble the ``provenance`` block. Never raises — provenance must not fail a command."""
    import sys

    prov: dict[str, Any] = {
        "model_package": _model_package(),
        "env": {
            "python": f"{sys.version_info.major}.{sys.version_info.minor}",
            "packages": _package_versions(),
        },
    }
    if (template_commit := _template_commit()) is not None:
        prov["template_commit"] = template_commit
    if (git := _git_provenance()) is not None:
        prov["git"] = git
    return prov


# ------------------------------------------------------------------------------------------------------
# Digests
# ------------------------------------------------------------------------------------------------------


def file_digest(path: Path) -> str:
    """SHA-256 of a file, as ``sha256:<hex>``.

    Examples:
        >>> import tempfile, pathlib
        >>> with tempfile.TemporaryDirectory() as d:
        ...     p = pathlib.Path(d) / "x.txt"
        ...     _ = p.write_text("hello")
        ...     file_digest(p)
        'sha256:2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824'
    """
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return f"sha256:{h.hexdigest()}"


def manifest_digest(artifact_dir: Path) -> str | None:
    """Digest of ``artifact_dir/manifest.yaml``, or None when the artifact has no manifest."""
    fp = Path(artifact_dir) / MANIFEST_FILENAME
    return file_digest(fp) if fp.is_file() else None


def input_ref(role: str, path: Path | str | None) -> dict[str, Any] | None:
    """Build one ``inputs`` entry: the role, the resolved path, and the input's manifest digest.

    Recording the digest is what lets a downstream command detect that it is running against a *different*
    artifact than a sibling command used, instead of silently producing an incoherent result.
    """
    if path is None:
        return None
    p = Path(path)
    ref: dict[str, Any] = {"role": role, "path": str(p)}
    if (digest := manifest_digest(p)) is not None:
        ref["manifest_digest"] = digest
    return ref


# ------------------------------------------------------------------------------------------------------
# Reading
# ------------------------------------------------------------------------------------------------------


def read_manifest(
    artifact_dir: Path | str,
    *,
    require_type: ArtifactType | str | None = None,
    require_kind: InferenceKind | str | None = None,
) -> dict[str, Any]:
    """Load and validate the manifest of an input artifact.

    Args:
        artifact_dir: the artifact directory (not the manifest file).
        require_type: if given, fail unless the manifest declares this ``artifact.type``.
        require_kind: if given, fail unless the manifest declares this ``artifact.kind`` (inference only).

    Raises:
        ManifestError: if the manifest is missing, unparseable, or fails a requirement.
    """
    d = Path(artifact_dir)
    fp = d / MANIFEST_FILENAME
    if not fp.is_file():
        raise ManifestError(
            f"No {MANIFEST_FILENAME} in {d}. Either it was not produced by this template, or the command "
            "that should have created it did not finish."
        )
    try:
        data = yaml.safe_load(fp.read_text()) or {}
    except yaml.YAMLError as e:
        raise ManifestError(f"Could not parse {fp}: {e}") from e

    artifact = data.get("artifact", {})
    if require_type is not None:
        want = str(require_type)
        got = artifact.get("type")
        if got != want:
            raise ManifestError(f"{d} is a {got!r} artifact; this input requires a {want!r} artifact.")
    if require_kind is not None:
        want_kind = str(require_kind)
        got_kind = artifact.get("kind")
        if got_kind != want_kind:
            raise ManifestError(
                f"{d} holds {got_kind!r} inference artifacts; this input requires {want_kind!r}."
            )
    return data


# ------------------------------------------------------------------------------------------------------
# Writing
# ------------------------------------------------------------------------------------------------------


@contextmanager
def write_artifact(
    dest: Path | str,
    *,
    artifact_type: ArtifactType | str,
    command: str,
    name: str | None = None,
    kind: InferenceKind | str | None = None,
    inputs: list[dict[str, Any] | None] | None = None,
    config: Any = None,
    do_overwrite: bool = False,
):
    """Stage an artifact directory and publish it atomically with its manifest.

    Yields ``(staging_dir, extras)``: write outputs into ``staging_dir``, and add any type-specific manifest
    fields to the ``extras`` dict. On clean exit the manifest is written and the staging directory is renamed
    onto ``dest``; on exception the staging directory is removed and ``dest`` is left untouched.

    Args:
        dest: final artifact directory.
        artifact_type: the :class:`ArtifactType` being produced.
        command: the command producing it (recorded in the manifest).
        name: artifact name; defaults to ``dest.name``.
        kind: for inference artifacts, what was materialized (see :class:`InferenceKind`). Consumers
            validate against this, so it is recorded in the ``artifact`` block rather than as a loose field.
        inputs: entries from :func:`input_ref` (``None`` entries are dropped).
        config: the resolved config to serialize alongside the manifest, if any.
        do_overwrite: replace an existing ``dest`` instead of failing.

    Raises:
        FileExistsError: if ``dest`` exists and ``do_overwrite`` is false.
    """
    dest = Path(dest)
    if dest.exists():
        if not do_overwrite:
            raise FileExistsError(
                f"{dest} already exists. Artifacts are immutable; pass do_overwrite=true to replace it."
            )
        logger.warning("Overwriting existing artifact at %s.", dest)

    dest.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{dest.name}.", dir=dest.parent))
    extras: dict[str, Any] = {}
    try:
        yield staging, extras

        artifact_block: dict[str, Any] = {"type": str(artifact_type), "name": name or dest.name}
        if kind is not None:
            artifact_block["kind"] = str(kind)
        manifest: dict[str, Any] = {
            "manifest_version": MANIFEST_VERSION,
            "artifact": artifact_block,
            "created_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "command": command,
            "provenance": build_provenance(),
            "inputs": [ref for ref in (inputs or []) if ref is not None],
        }
        if config is not None:
            manifest["config"] = _write_resolved_config(staging, config)
        manifest.update(extras)
        manifest["outputs"] = _describe_outputs(staging)

        (staging / MANIFEST_FILENAME).write_text(yaml.safe_dump(manifest, sort_keys=False))
        _publish(staging, dest, do_overwrite=do_overwrite)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    logger.info("Published %s artifact to %s.", artifact_type, dest)


def _write_resolved_config(staging: Path, config: Any) -> dict[str, str]:
    """Serialize the resolved config into the staging dir; return the manifest's ``config`` block."""
    from omegaconf import OmegaConf

    fp = staging / "resolved_config.yaml"
    if OmegaConf.is_config(config):
        fp.write_text(OmegaConf.to_yaml(config, resolve=True))
    else:  # pragma: no cover - plain mappings are accepted for testability
        fp.write_text(yaml.safe_dump(config, sort_keys=False))
    return {"resolved": fp.name, "digest": file_digest(fp)}


def _describe_outputs(staging: Path) -> list[dict[str, Any]]:
    """Describe the published files (relative path + digest), skipping the manifest itself."""
    outputs = []
    for fp in sorted(staging.rglob("*")):
        if not fp.is_file() or fp.name == MANIFEST_FILENAME:
            continue
        entry: dict[str, Any] = {"file": str(fp.relative_to(staging)), "digest": file_digest(fp)}
        outputs.append(entry)
    return outputs


def _publish(staging: Path, dest: Path, *, do_overwrite: bool) -> None:
    """Rename ``staging`` onto ``dest``, swapping any existing directory aside first."""
    if dest.exists():
        if not do_overwrite:  # pragma: no cover - guarded by the caller
            raise FileExistsError(dest)
        victim = dest.with_name(f".{dest.name}.replaced.{os.getpid()}")
        dest.rename(victim)
        try:
            staging.rename(dest)
        finally:
            shutil.rmtree(victim, ignore_errors=True)
    else:
        staging.rename(dest)
