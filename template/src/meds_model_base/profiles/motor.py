"""The MOTOR-style profile: time-to-event pretraining, then supervised fine-tuning.

``MotorModel`` is one module with two objectives, auto-selected by whether the batch carries labels:

- **``pretrain`` (no labels)** — a time-to-event pretraining objective: from each position's contextual
  state, predict the gap to the next measurement under an exponential hazard (NLL). This teaches the
  encoder the temporal structure of a subject's timeline (the essence of MOTOR).
- **``supervised_train`` (labels present)** — a binary classification head on the pooled representation,
  **warm-started** from the pretrained encoder (via ``input_pretrained_model_dir``; ``load_state_dict``
  ``strict=False`` transfers the shared embedder + encoder, leaves the fresh classification head).

The warm start is checked: transferring *nothing* raises rather than quietly training from scratch, since
a fine-tune that matched no parameters is a different experiment than the one that was requested.

At ``predict`` it applies the classification head (``SupervisedPredictCommand``). This is a compact,
native reimplementation of the MOTOR *approach* on the modern stack — no FEMR dependency.
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


class MotorModel(BaseLightningModule):
    """Time-to-event pretraining + supervised fine-tuning in one module (objective auto-selected)."""

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
        optimizer: Callable | None = None,
        scheduler: Callable | None = None,
    ) -> None:
        super().__init__(optimizer=optimizer, scheduler=scheduler)
        self.save_hyperparameters(ignore=["optimizer", "scheduler"])

        self.embedder = CodeEmbedder(vocab_size, d_model, use_numeric_value=use_numeric_value)
        if encoder == "gru":
            self.encoder = GRUEncoder(d_model, num_layers=num_layers, dropout=dropout)
        elif encoder == "transformer":
            self.encoder = TransformerEncoder(d_model, nhead=nhead, num_layers=num_layers, dropout=dropout)
        else:  # pragma: no cover
            raise ValueError(f"Unknown encoder {encoder!r}.")
        self.tte_head = nn.Linear(d_model, 1)  # log-rate of an exponential time-to-next-event
        self.cls_head = nn.Linear(d_model, 1)  # supervised classification
        self.threshold = threshold

    def _sequence(self, batch: MEDSTorchBatch) -> Tensor:
        return self.encoder(self.embedder(batch), padding_mask(batch))

    def encode(self, batch: MEDSTorchBatch) -> Tensor:
        """Pooled subject representation — also what ``infer`` materializes as embeddings."""
        return self._pooled(batch)

    def _pooled(self, batch: MEDSTorchBatch) -> Tensor:
        return masked_mean(self._sequence(batch), padding_mask(batch))

    def compute_loss(self, batch: MEDSTorchBatch) -> tuple[Tensor, dict[str, Tensor]]:
        if batch.has_labels:
            logits = self.cls_head(self._pooled(batch)).squeeze(-1)
            target = batch.boolean_value.to(logits.dtype)
            loss = F.binary_cross_entropy_with_logits(logits, target)
            with torch.no_grad():
                acc = ((logits > 0).to(target.dtype) == target).float().mean()
            return loss, {"acc": acc, "mode_classify": torch.ones((), device=loss.device)}

        # Time-to-event pretraining: predict gap to next measurement (exponential NLL).
        hidden = self._sequence(batch)
        rate = F.softplus(self.tte_head(hidden[:, :-1]).squeeze(-1)) + 1e-4
        # log1p-compress the day-scale gaps so the exponential NLL stays well-conditioned across the
        # wide dynamic range of inter-event times (minutes to years).
        target_dt = torch.log1p(torch.clamp(torch.nan_to_num(batch.time_delta_days[:, 1:]), min=0.0))
        mask = padding_mask(batch)[:, 1:].to(rate.dtype)
        nll = (-torch.log(rate) + rate * target_dt) * mask
        loss = nll.sum() / mask.sum().clamp(min=1.0)
        return loss, {"mode_tte": torch.ones((), device=loss.device)}

    def predict_step(self, batch: MEDSTorchBatch, batch_idx: int = 0) -> dict[str, Tensor]:
        probs = torch.sigmoid(self.cls_head(self._pooled(batch)).squeeze(-1))
        return {
            "predicted_boolean_probability": probs.detach().cpu(),
            "predicted_boolean_value": (probs > self.threshold).detach().cpu(),
        }
