"""``infer`` — materialize reusable outputs from a pretrained model.

Writes ``<data_dir>/<output_inference_subdir>/artifacts.parquet``, keyed on
``(subject_id, prediction_time)`` plus whatever the model's ``predict_step`` returns: embeddings, generated
trajectories, hazards, native scores, token probabilities.

There is no ``inference.kind`` argument, because the kind is a property of the model, not a choice the
caller makes. It is *recorded* in the manifest — along with the column schema — and downstream consumers
validate against it. That is what turns this command's output from a dead-end artifact into something a
probe or a materialized zero-shot ``predict`` can safely consume.

The model declares its own kind via ``LightningModule.inference_kind``; the default is ``embeddings``.

``external_labels_dir`` fixes the timepoints. Its split layout is materialized into a temporary
directory and discarded — only the inference artifact is published.
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

import polars as pl

from ..manifest import ArtifactType, InferenceKind, dir_digest, input_ref, read_manifest, write_artifact
from ..schemas import TaskAgnosticOutputSchema
from ..tasks import materialize_labels
from ..utils import resolve_subdir
from ._runtime import KEYS, load_index, load_trained_module, resolve_splits, run_predict_step, stack_outputs
from .base import InferCommand
from .preprocess_data import PATIENTS_SUBDIR

if TYPE_CHECKING:  # pragma: no cover - typing only
    import lightning.pytorch as pl_light
    from omegaconf import DictConfig

logger = logging.getLogger(__name__)

ARTIFACTS_FILENAME = "artifacts.parquet"


class DefaultInferCommand(InferCommand):
    """Default inference: persist whatever the model's ``predict_step`` emits at each timepoint.

    A model exposes its outputs by returning a dict of ``[B]`` (scalar column) or ``[B, D]`` (list column)
    tensors from ``predict_step``. Override ``predict_step`` in ``model.py`` to emit embeddings, generated
    tokens, or hazards, and set ``inference_kind`` so consumers can check what they are being handed.
    """

    default_kind: ClassVar[InferenceKind] = InferenceKind.embeddings

    def run(self, cfg: DictConfig) -> Path:
        data_dir = Path(cfg.input_data_dir)
        read_manifest(data_dir / PATIENTS_SUBDIR, require_type=ArtifactType.data)

        model_dir = Path(cfg.input_pretrained_model_dir)
        read_manifest(model_dir, require_type=ArtifactType.pretrained_model)
        module = load_trained_module(model_dir)

        if not cfg.get("external_labels_dir"):
            raise ValueError(
                "external_labels_dir is required by this implementation: it fixes the timepoints to infer "
                "at. Pass a MEDS labels directory, or override `run` to derive an index some other way."
            )
        # Materialized beside the artifact being built, then discarded: the split layout is what the
        # datamodule reads, not something worth publishing.
        with tempfile.TemporaryDirectory(prefix=".labels.", dir=data_dir) as tmp:
            labels_dir, label_summary = materialize_labels(
                cfg.external_labels_dir,
                data_dir / PATIENTS_SUBDIR,
                Path(tmp) / "labels",
                # Inference never sees ground truth: without boolean_value, meds-torch-data leaves
                # `batch.boolean_value` absent rather than handing the model the answer.
                include_labels=False,
            )
            cfg.datamodule.config.task_labels_dir = str(labels_dir)
            splits = resolve_splits(cfg)
            index = load_index(labels_dir, splits)
            frame = self.infer(cfg, module, index)
        dest = resolve_subdir(data_dir, cfg.output_inference_subdir)
        kind = str(getattr(module, "inference_kind", self.default_kind))
        with write_artifact(
            dest,
            artifact_type=ArtifactType.inference,
            command=self.name.value,
            name=Path(cfg.output_inference_subdir).name,
            kind=kind,
            inputs=[
                input_ref("input_data_dir", data_dir),
                input_ref("input_pretrained_model_dir", model_dir),
                _labels_ref(cfg.external_labels_dir),
            ],
            config=cfg,
            do_overwrite=bool(cfg.get("do_overwrite", False)),
        ) as (staging, extras):
            table = TaskAgnosticOutputSchema.align(frame.to_arrow())
            pl.from_arrow(table).write_parquet(staging / ARTIFACTS_FILENAME)
            extras["columns"] = _column_schema(frame)
            extras["labels"] = label_summary
            extras["n_rows"] = len(frame)
            extras["splits"] = sorted(index["split"].unique().to_list())

        logger.info("Wrote %d %s rows to %s.", len(frame), kind, dest)
        return dest

    def infer(self, cfg: DictConfig, module: pl_light.LightningModule, index: pl.DataFrame) -> pl.DataFrame:
        """Run the model over every split in ``index`` and assemble one frame of per-timepoint outputs."""
        driver = _InferStepAdapter(module)
        frames = []
        for split in index["split"].unique(maintain_order=True).to_list():
            keys, outputs = run_predict_step(cfg, driver, split)
            if not len(keys):
                continue
            frames.append(keys.with_columns(stack_outputs(outputs)))
        if not frames:
            raise RuntimeError(
                "The datamodule produced no rows for any requested split. Check that external_labels_dir "
                "and the tensorized cohort refer to the same subjects."
            )
        return pl.concat(frames, how="vertical_relaxed").unique(subset=KEYS, maintain_order=True)


def _labels_ref(labels_dir) -> dict:
    """An ``inputs`` entry for a plain labels directory: no manifest of its own, so digest the contents."""
    ref: dict = {"role": "external_labels_dir", "path": str(labels_dir)}
    if (digest := dir_digest(labels_dir)) is not None:
        ref["digest"] = digest
    return ref


def _InferStepAdapter(module: pl_light.LightningModule) -> pl_light.LightningModule:  # noqa: N802
    """Present ``module.infer_step`` as ``predict_step`` so Lightning's predict loop drives inference.

    ``infer`` and ``predict`` ask a model for different things — reusable artifacts versus task
    probabilities — but Lightning only calls ``predict_step``. Rather than overload one hook with a mode
    flag, the module is wrapped for the duration of the run.
    """
    import lightning.pytorch as L

    class _Adapter(L.LightningModule):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def predict_step(self, batch, batch_idx: int = 0):
            return self.inner.infer_step(batch)

    adapter = _Adapter(module)
    adapter.eval()
    return adapter


def _column_schema(frame: pl.DataFrame) -> list[dict]:
    """Describe the non-key columns for the manifest: name, dtype, and width for vector columns."""
    described = []
    for name, dtype in zip(frame.columns, frame.dtypes, strict=True):
        if name in KEYS:
            continue
        entry: dict = {"name": name, "dtype": str(dtype)}
        if isinstance(dtype, pl.List) and len(frame):
            first = frame[name][0]
            if first is not None:
                entry["width"] = len(first)
        described.append(entry)
    return described
