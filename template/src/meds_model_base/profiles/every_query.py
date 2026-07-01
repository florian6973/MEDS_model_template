"""The EveryQuery profile: query-conditioned pretraining, then zero-shot prediction by querying.

``EveryQueryModel`` pretrains a multi-label *occurrence* head — from a subject's pooled representation,
predict which codes occur in their timeline (``unsupervised_train``; a simplified proxy for EveryQuery's
"does code X occur within horizon" objective). At ``prediction``, ``EveryQueryPredictionStep`` translates
the ACES task definition into a code *query* and reads that code's predicted occurrence probability — i.e.
it *parses* the ACES file into an EQ query rather than running ACES over data.

The task→code translation is best-effort (first plain-predicate code); when no task is given or the code is
unknown it falls back to the ``ZeroShotPredictionStep`` placeholder. Real EQ query translation is
model-specific and can be as rich as needed.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from meds_torchdata import MEDSTorchBatch

from ..lightning.modules import (
    PAD_INDEX,
    BaseLightningModule,
    CodeEmbedder,
    GRUEncoder,
    TransformerEncoder,
    masked_mean,
    padding_mask,
)
from ..steps.predict import ZeroShotPredictionStep

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

    def predict_step(self, batch: MEDSTorchBatch, batch_idx: int = 0) -> dict[str, Tensor]:  # noqa: ARG002
        """Per-code occurrence probabilities ``[B, vocab_size]`` — query any code downstream."""
        return {"query_probs": torch.sigmoid(self.occurrence_logits(batch)).detach().cpu()}


def _multi_hot(codes: Tensor, vocab_size: int) -> Tensor:
    """Build a ``[B, vocab_size]`` multi-hot of which (non-pad) codes appear in each row of ``codes``."""
    batch_size = codes.shape[0]
    target = torch.zeros(batch_size, vocab_size, device=codes.device)
    target.scatter_(1, codes.clamp(min=0), 1.0)
    target[:, PAD_INDEX] = 0.0
    return target


class EveryQueryPredictionStep(ZeroShotPredictionStep):
    """Zero-shot prediction by translating the ACES ``task`` into a code-occurrence query."""

    def resolve(self, cfg, keys, outputs) -> list[float]:
        target_idx = self._target_vocab_index(cfg)
        if target_idx is None or not outputs or "query_probs" not in outputs[0]:
            return super().resolve(cfg, keys, outputs)
        probs = torch.cat([o["query_probs"] for o in outputs], dim=0)
        col = min(int(target_idx), probs.shape[1] - 1)
        logger.info("EveryQuery: querying vocab index %d for the task.", col)
        return probs[:, col].float().tolist()

    @staticmethod
    def _target_vocab_index(cfg) -> int | None:
        task = cfg.get("task")
        if not task:
            return None
        try:
            from ..tasks import load_task_config

            task_cfg = load_task_config(task)
            code = _first_plain_code(task_cfg)
            if code is None:
                return None
            vocab = _load_vocab(Path(cfg.datamodule.config.tensorized_cohort_dir))
            return vocab.get(code)
        except Exception as e:  # pragma: no cover - best-effort translation
            logger.warning("Could not translate task to a query code: %s", e)
            return None


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
