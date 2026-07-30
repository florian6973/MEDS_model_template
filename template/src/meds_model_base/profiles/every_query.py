"""The EveryQuery profile: query-conditioned pretraining, then zero-shot prediction by querying.

``EveryQueryModel`` pretrains a multi-label *occurrence* head — from a subject's pooled representation,
predict which codes occur in their timeline (``pretrain``; a simplified proxy for EveryQuery's "does code X
occur within horizon" objective). At ``predict``, :class:`EveryQueryPredictCommand` translates the ACES task
definition into a code *query* and reads that code's predicted occurrence probability — it *parses* the
ACES file into an EQ query rather than running ACES over data.

The task definition is recovered from the task artifact's manifest, so ``predict`` needs no extra argument
and cannot be handed a definition that disagrees with the materialized labels.

Translation is best-effort (the first plain-predicate code), but a failure to translate is an **error**, not
a fallback: a zero-shot model that cannot resolve the task has nothing to say, and emitting a placeholder
probability would produce a schema-valid predictions file of pure noise.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import ClassVar

import torch
import torch.nn.functional as F
from meds_torchdata import MEDSTorchBatch
from torch import Tensor, nn

from ..commands.predict import ZeroShotPredictCommand
from ..lightning.modules import (
    PAD_INDEX,
    BaseLightningModule,
    CodeEmbedder,
    GRUEncoder,
    TransformerEncoder,
    masked_mean,
    padding_mask,
)

logger = logging.getLogger(__name__)


class EveryQueryModel(BaseLightningModule):
    """Encoder + multi-label code-occurrence head (a query-conditioned pretraining proxy)."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 64,
        encoder: str = "gru",
        num_layers: int = 1,
        nhead: int = 4,
        dropout: float = 0.1,
        use_numeric_value: bool = True,
        optimizer: Callable | None = None,
        scheduler: Callable | None = None,
    ) -> None:
        super().__init__(optimizer=optimizer, scheduler=scheduler)
        self.save_hyperparameters(ignore=["optimizer", "scheduler"])

        self.vocab_size = vocab_size
        self.embedder = CodeEmbedder(vocab_size, d_model, use_numeric_value=use_numeric_value)
        if encoder == "gru":
            self.encoder = GRUEncoder(d_model, num_layers=num_layers, dropout=dropout)
        elif encoder == "transformer":
            self.encoder = TransformerEncoder(d_model, nhead=nhead, num_layers=num_layers, dropout=dropout)
        else:  # pragma: no cover
            raise ValueError(f"Unknown encoder {encoder!r}.")
        self.query_head = nn.Linear(d_model, vocab_size)  # per-code occurrence logits

    #: `infer` materializes per-code occurrence probabilities, not embeddings.
    inference_kind: ClassVar[str] = "scores"

    def infer_step(self, batch: MEDSTorchBatch) -> dict[str, Tensor]:
        """Materialize the full occurrence-probability vector so any code can be queried later."""
        return {"query_probs": torch.sigmoid(self.occurrence_logits(batch)).detach().cpu()}

    def _pooled(self, batch: MEDSTorchBatch) -> Tensor:
        mask = padding_mask(batch)
        return masked_mean(self.encoder(self.embedder(batch), mask), mask)

    def occurrence_logits(self, batch: MEDSTorchBatch) -> Tensor:
        """Per-code occurrence logits ``[B, vocab_size]``."""
        return self.query_head(self._pooled(batch))

    def compute_loss(self, batch: MEDSTorchBatch) -> tuple[Tensor, dict[str, Tensor]]:
        logits = self.occurrence_logits(batch)
        target = _multi_hot(batch.code, self.vocab_size)
        loss = F.binary_cross_entropy_with_logits(logits, target)
        return loss, {}

    def predict_step(self, batch: MEDSTorchBatch, batch_idx: int = 0) -> dict[str, Tensor]:
        """Per-code occurrence probabilities ``[B, vocab_size]`` — query any code downstream."""
        return {"query_probs": torch.sigmoid(self.occurrence_logits(batch)).detach().cpu()}


def _multi_hot(codes: Tensor, vocab_size: int) -> Tensor:
    """Build a ``[B, vocab_size]`` multi-hot of which (non-pad) codes appear in each row of ``codes``."""
    batch_size = codes.shape[0]
    target = torch.zeros(batch_size, vocab_size, device=codes.device)
    target.scatter_(1, codes.clamp(min=0), 1.0)
    target[:, PAD_INDEX] = 0.0
    return target


class EveryQueryPredictCommand(ZeroShotPredictCommand):
    """Zero-shot prediction by translating the task definition into a code-occurrence query."""

    def resolve(self, cfg, module, index):
        import polars as pl

        from ..commands._runtime import KEYS, run_predict_step
        from ..commands.predict import PROBABILITY_COLUMN
        from ..utils import resolve_subdir

        task_dir = resolve_subdir(cfg.input_data_dir, cfg.input_task_subdir)
        col = self._target_vocab_index(cfg, task_dir)

        frames = []
        for split in index["split"].unique(maintain_order=True).to_list():
            keys, outputs = run_predict_step(cfg, module, split)
            if not len(keys):
                continue
            if not outputs or "query_probs" not in outputs[0]:
                raise ValueError(
                    "The model's predict_step did not emit `query_probs`, which EveryQuery prediction "
                    "queries. Check that model.py still returns it."
                )
            probs = torch.cat([o["query_probs"] for o in outputs], dim=0)
            column = min(col, probs.shape[1] - 1)
            frames.append(
                keys.with_columns(
                    pl.Series(PROBABILITY_COLUMN, probs[:, column].float().tolist(), dtype=pl.Float32),
                    pl.lit(split).alias("split"),
                )
            )
        if not frames:
            raise ValueError("The datamodule produced no rows for any requested split.")
        return pl.concat(frames, how="vertical_relaxed").select([*KEYS, "split", PROBABILITY_COLUMN])

    @staticmethod
    def _target_vocab_index(cfg, task_dir) -> int:
        """Translate the task definition into a vocabulary index, or fail loudly."""
        from ..commands.predict import task_definition_path
        from ..tasks import load_task_config

        definition = task_definition_path(task_dir)
        if definition is None:
            raise ValueError(
                f"The task at {task_dir} was materialized from pre-extracted labels, so it carries no ACES "
                "definition for EveryQuery to translate into a query. Re-run preprocess_task with the ACES "
                "YAML as external_task_file."
            )
        code = _first_plain_code(load_task_config(definition))
        if code is None:
            raise ValueError(f"No plain-predicate code found in {definition} to query on.")
        vocab = _load_vocab(Path(cfg.input_data_dir) / "patients")
        if code not in vocab:
            raise ValueError(
                f"Task code {code!r} is not in this cohort's vocabulary, so the query cannot be answered. "
                "EveryQuery can only score tasks whose target code was seen during preprocessing."
            )
        logger.info("EveryQuery: querying vocab index %d for code %s.", vocab[code], code)
        return int(vocab[code])


def _first_plain_code(task_cfg) -> str | None:
    """Return the first plain-predicate code string in an ACES ``TaskExtractorConfig`` (best-effort)."""
    predicates = getattr(task_cfg, "predicates", {}) or {}
    for pred in predicates.values():
        code = getattr(pred, "code", None)
        if isinstance(code, str):
            return code
    return None


def _load_vocab(tensorized_cohort_dir: Path) -> dict[str, int]:
    """Map ``code`` → ``code/vocab_index`` from the tensorized cohort's ``metadata/codes.parquet``."""
    import polars as pl

    codes = pl.read_parquet(tensorized_cohort_dir / "metadata" / "codes.parquet")
    return dict(zip(codes["code"].to_list(), codes["code/vocab_index"].to_list(), strict=False))
