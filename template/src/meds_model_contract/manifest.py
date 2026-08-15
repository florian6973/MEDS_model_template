"""Atomic artifact publication with mandatory, portable resource measurements."""

from __future__ import annotations

import hashlib
import os
import resource
import shutil
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import yaml


def _rss_bytes(value: int) -> int:
    # Linux reports KiB; macOS reports bytes. sys.platform is deliberately avoided so unusual Unix
    # implementations can override/report their method rather than receiving a false portable claim.
    return value * 1024 if os.uname().sysname != "Darwin" else value


class _ProcessTreeMemorySampler:
    """Sample aggregate PSS/RSS for this process and its live descendants on Linux."""

    def __init__(self, interval_seconds: float = 0.25) -> None:
        self.interval_seconds = interval_seconds
        self.peak_pss_bytes: int | None = None
        self.peak_rss_bytes: int | None = None
        self.source = "linux_proc_smaps_rollup" if Path("/proc/self/smaps_rollup").is_file() else "getrusage"
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="meds-memory-sampler", daemon=True)

    def start(self) -> None:
        self._sample()
        self._thread.start()

    def stop(self) -> None:
        self._sample()
        self._stop.set()
        self._thread.join(timeout=max(1.0, self.interval_seconds * 4))
        self._sample()
        if self.peak_rss_bytes is None:
            usage = resource.getrusage(resource.RUSAGE_SELF)
            self.peak_rss_bytes = _rss_bytes(usage.ru_maxrss)

    def _run(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            self._sample()

    @staticmethod
    def _process_tree(root_pid: int) -> set[int]:
        found: set[int] = set()
        pending = [root_pid]
        while pending:
            pid = pending.pop()
            if pid in found:
                continue
            found.add(pid)
            children = Path(f"/proc/{pid}/task/{pid}/children")
            try:
                pending.extend(int(value) for value in children.read_text().split())
            except (FileNotFoundError, PermissionError, ProcessLookupError):
                continue
        return found

    @staticmethod
    def _smaps(pid: int) -> tuple[int, int] | None:
        try:
            values = {}
            for line in Path(f"/proc/{pid}/smaps_rollup").read_text().splitlines():
                key, separator, rest = line.partition(":")
                if separator and key in {"Pss", "Rss"}:
                    values[key] = int(rest.split()[0]) * 1024
            return values["Pss"], values["Rss"]
        except (FileNotFoundError, KeyError, PermissionError, ProcessLookupError, ValueError):
            return None

    def _sample(self) -> None:
        if self.source != "linux_proc_smaps_rollup":
            return
        samples = [sample for pid in self._process_tree(os.getpid()) if (sample := self._smaps(pid))]
        if not samples:
            return
        pss = sum(sample[0] for sample in samples)
        rss = sum(sample[1] for sample in samples)
        self.peak_pss_bytes = max(self.peak_pss_bytes or 0, pss)
        self.peak_rss_bytes = max(self.peak_rss_bytes or 0, rss)


def read_manifest(directory: Path | str) -> dict[str, Any]:
    path = Path(directory) / "manifest.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"Artifact {directory} has no manifest.yaml")
    value = yaml.safe_load(path.read_text())
    if not isinstance(value, dict) or value.get("contract_version") != "2":
        raise ValueError(f"{path} is not a version 2 MEDS model artifact manifest")
    return value


def digest_file(path: Path | str) -> str:
    digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    return f"sha256:{digest}"


@contextmanager
def measured_artifact(
    destination: Path | str,
    *,
    artifact_type: str,
    command: str,
    producer: str,
    payload_format: str,
    inputs: dict[str, Any] | None = None,
    config_digest: str | None = None,
    external_predicates_file: Path | str | None = None,
    overwrite: bool = False,
) -> Iterator[Path]:
    """Stage an artifact, measure the command, write its manifest last, and publish atomically."""
    destination = Path(destination)
    staging = destination.with_name(f".{destination.name}.staging-{os.getpid()}")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    started = time.monotonic()
    sampler = _ProcessTreeMemorySampler()
    sampler.start()
    try:
        yield staging
        sampler.stop()
        predicates = None
        if external_predicates_file is not None:
            predicates = {
                "path": str(Path(external_predicates_file).resolve()),
                "digest": digest_file(external_predicates_file),
            }
        recorded_inputs = dict(inputs or {})
        if predicates is not None:
            recorded_inputs["external_predicates_file"] = predicates
        manifest = {
            "contract_version": "2",
            "artifact": {
                "type": artifact_type,
                "payload_format": payload_format,
                "producer": producer,
            },
            "command": {"name": command, "config_digest": config_digest},
            "inputs": recorded_inputs,
            "resources": {
                "wall_seconds": time.monotonic() - started,
                "memory": {
                    "peak_process_tree_pss_bytes": sampler.peak_pss_bytes,
                    "peak_process_tree_rss_bytes": sampler.peak_rss_bytes,
                    "sampling_interval_seconds": sampler.interval_seconds,
                    "source": sampler.source,
                    "includes_descendants": sampler.source == "linux_proc_smaps_rollup",
                },
                "gpu": {
                    "peak_allocated_bytes": None,
                    "peak_reserved_bytes": None,
                    "source": None,
                },
                "measurement": {
                    "schema_version": "1",
                },
            },
        }
        (staging / "manifest.yaml").write_text(yaml.safe_dump(manifest, sort_keys=False))
        if destination.exists():
            if not overwrite:
                raise FileExistsError(f"{destination} already exists; pass overwrite=true to replace it")
            shutil.rmtree(destination)
        staging.replace(destination)
    except BaseException:
        sampler.stop()
        shutil.rmtree(staging, ignore_errors=True)
        raise
