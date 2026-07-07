"""The supervised-basic profile: a sequence encoder + binary-classification head.

``SupervisedClassifier`` embeds a ``MEDSTorchBatch``, encodes it with a GRU or Transformer, mean-pools to
a subject representation at the prediction time, and applies a linear head to produce a single logit.
Trained with binary cross-entropy on ``batch.boolean_value`` (``supervised_train``); at ``prediction`` time
it emits ``sigmoid(logit)`` as ``predicted_boolean_probability``.

This is the concrete, runnable default so ``supervised-basic`` works end-to-end out of the box. A generated
repo subclasses it in ``model.py``.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn.functional as F
from meds_torchdata import MEDSTorchBatch
from torch import Tensor, nn

from ..lightning.modules import (
    BaseLightningModule,
    CodeEmbedder,
    GRUEncoder,
    TransformerEncoder,
    masked_mean,
    padding_mask,
)


class SupervisedClassifier(BaseLightningModule):
    """A binary classifier over MEDS sequences.

    Args:
        vocab_size: code-vocabulary size (injected from ``datamodule.config.vocab_size`` at build time).
        d_model: hidden size.
        encoder: ``"gru"`` or ``"transformer"``.
        num_layers: encoder depth.
        nhead: attention heads (transformer only).
        dropout: encoder dropout.
        use_numeric_value: fold numeric values into the embedding.
        threshold: probability threshold for the optional ``predicted_boolean_value`` column.
        optimizer / scheduler: Hydra ``_partial_`` factories (see ``BaseLightningModule``).
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 64,
        encoder: str = "gru",
        num_layers: int = 1,
        nhead: int = 4,
        dropout: float = 0.1,
        use_numeric_value: bool = True,
        threshold: float = 0.5,
        optimizer: Callable[..., torch.optim.Optimizer] | None = None,
        scheduler: Callable | None = None,
    ) -> None:
        super().__init__(optimizer=optimizer, scheduler=scheduler)
        self.save_hyperparameters(ignore=["optimizer", "scheduler"])

        self.embedder = CodeEmbedder(vocab_size, d_model, use_numeric_value=use_numeric_value)
        if encoder == "gru":
            self.encoder = GRUEncoder(d_model, num_layers=num_layers, dropout=dropout)
        elif encoder == "transformer":
            self.encoder = TransformerEncoder(d_model, nhead=nhead, num_layers=num_layers, dropout=dropout)
        else:  # pragma: no cover - guarded by config choices
            raise ValueError(f"Unknown encoder {encoder!r}; expected 'gru' or 'transformer'.")
        self.head = nn.Linear(d_model, 1)
        self.threshold = threshold

    def encode(self, batch: MEDSTorchBatch) -> Tensor:
        """Pool a batch to a ``[B, d_model]`` subject representation at the prediction time."""
        mask = padding_mask(batch)
        x = self.embedder(batch)
        h = self.encoder(x, mask)
        return masked_mean(h, mask)

    def forward(self, batch: MEDSTorchBatch) -> Tensor:
        """Return per-subject logits ``[B]``."""
        return self.head(self.encode(batch)).squeeze(-1)

    def compute_loss(self, batch: MEDSTorchBatch) -> tuple[Tensor, dict[str, Tensor]]:
        logits = self(batch)
        target = batch.boolean_value.to(logits.dtype)
        loss = F.binary_cross_entropy_with_logits(logits, target)
        with torch.no_grad():
            acc = ((logits > 0).to(target.dtype) == target).float().mean()
        return loss, {"acc": acc}

    def predict_step(self, batch: MEDSTorchBatch, batch_idx: int = 0) -> dict[str, Tensor]:
        """Return per-subject probabilities (aligned to the dataloader's sample order)."""
        probs = torch.sigmoid(self(batch))
        return {
            "predicted_boolean_probability": probs.detach().cpu(),
            "predicted_boolean_value": (probs > self.threshold).detach().cpu(),
        }
