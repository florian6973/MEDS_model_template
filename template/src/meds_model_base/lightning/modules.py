"""Reusable ``nn.Module`` building blocks and a shared ``BaseLightningModule``.

These are the pieces the four reference profiles compose. They operate on a ``MEDSTorchBatch`` in the
default ``SM`` (subject-measurement, flattened) mode, whose relevant tensors are:

- ``batch.code`` — ``[B, L]`` int64 code (vocab) indices; ``PAD_INDEX == 0`` marks padding.
- ``batch.numeric_value`` / ``batch.numeric_value_mask`` — ``[B, L]`` float value + presence mask.
- ``batch.time_delta_days`` — ``[B, L]`` float inter-measurement gaps.
- ``batch.boolean_value`` — ``[B]`` labels (present only when the datamodule has a ``task_labels_dir``).

Nothing here is task-specific: the profiles add the head (classification / autoregressive LM / query /
time-to-event) on top.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import lightning.pytorch as L
import torch
from meds_torchdata import MEDSTorchBatch
from torch import Tensor, nn

if TYPE_CHECKING:  # pragma: no cover - typing only
    from torch.optim import Optimizer
    from torch.optim.lr_scheduler import LRScheduler

PAD_INDEX = MEDSTorchBatch.PAD_INDEX


def padding_mask(batch: MEDSTorchBatch) -> Tensor:
    """Boolean ``[B, L]`` mask; ``True`` at real (non-pad) positions.

    Examples:
        >>> import torch
        >>> from unittest.mock import Mock
        >>> b = Mock(code=torch.tensor([[1, 2, 0], [3, 0, 0]]))
        >>> padding_mask(b).int().tolist()
        [[1, 1, 0], [1, 0, 0]]
    """
    return batch.code != PAD_INDEX


def masked_mean(x: Tensor, mask: Tensor) -> Tensor:
    """Mean-pool ``x`` (``[B, L, H]``) over the length axis using a ``[B, L]`` boolean ``mask``.

    Rows with an all-``False`` mask pool to zeros (avoids a divide-by-zero).

    Examples:
        >>> import torch
        >>> x = torch.tensor([[[1.0], [3.0], [99.0]]])           # [1, 3, 1]
        >>> m = torch.tensor([[True, True, False]])
        >>> masked_mean(x, m).tolist()
        [[2.0]]
    """
    m = mask.unsqueeze(-1).to(x.dtype)
    summed = (x * m).sum(dim=1)
    counts = m.sum(dim=1).clamp(min=1.0)
    return summed / counts


class CodeEmbedder(nn.Module):
    """Embed a ``MEDSTorchBatch`` into ``[B, L, d_model]``: code embedding (+ numeric value + time delta).

    Args:
        vocab_size: size of the code vocabulary (``datamodule.config.vocab_size``).
        d_model: embedding dimension.
        use_numeric_value: add a projection of the (masked) numeric value.
        use_time_delta: add a projection of the inter-measurement time delta.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        *,
        use_numeric_value: bool = True,
        use_time_delta: bool = False,
    ) -> None:
        super().__init__()
        self.code_embed = nn.Embedding(vocab_size, d_model, padding_idx=PAD_INDEX)
        self.num_proj = nn.Linear(1, d_model) if use_numeric_value else None
        self.time_proj = nn.Linear(1, d_model) if use_time_delta else None

    def forward(self, batch: MEDSTorchBatch) -> Tensor:
        x = self.code_embed(batch.code)
        if self.num_proj is not None:
            nv = torch.nan_to_num(batch.numeric_value) * batch.numeric_value_mask.to(x.dtype)
            x = x + self.num_proj(nv.unsqueeze(-1))
        if self.time_proj is not None:
            td = torch.nan_to_num(batch.time_delta_days).unsqueeze(-1)
            x = x + self.time_proj(td)
        return x


class GRUEncoder(nn.Module):
    """A small GRU sequence encoder: ``[B, L, H] → [B, L, H]`` (contextualized per-position states)."""

    def __init__(self, d_model: int, num_layers: int = 1, dropout: float = 0.0) -> None:
        super().__init__()
        self.gru = nn.GRU(
            d_model,
            d_model,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        out, _ = self.gru(x)
        return out


class TransformerEncoder(nn.Module):
    """A learned-positional Transformer encoder: ``[B, L, H] → [B, L, H]`` with key-padding masking."""

    def __init__(
        self,
        d_model: int,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int | None = None,
        dropout: float = 0.1,
        max_len: int = 2048,
    ) -> None:
        super().__init__()
        self.pos = nn.Embedding(max_len, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model,
            nhead,
            dim_feedforward=dim_feedforward or 4 * d_model,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        positions = torch.arange(x.shape[1], device=x.device)
        x = x + self.pos(positions).unsqueeze(0)
        key_padding_mask = ~mask if mask is not None else None  # True == ignore
        return self.encoder(x, src_key_padding_mask=key_padding_mask)


class BaseLightningModule(L.LightningModule):
    """Shared Lightning plumbing: optimizer/scheduler wiring + train/val logging.

    Subclasses implement :meth:`compute_loss` (returns ``(loss, metrics_dict)``); this base handles
    ``training_step`` / ``validation_step`` (logging the loss and any metrics) and ``configure_optimizers``.
    ``optimizer`` / ``scheduler`` are *partials* (from ``_partial_: true`` Hydra configs) — an optimizer
    factory ``params -> Optimizer`` and a scheduler factory ``optimizer -> LRScheduler``.
    """

    def __init__(
        self,
        optimizer: Callable[..., Optimizer] | None = None,
        scheduler: Callable[..., LRScheduler] | None = None,
    ) -> None:
        super().__init__()
        self._optimizer_factory = optimizer
        self._scheduler_factory = scheduler

    def compute_loss(self, batch: MEDSTorchBatch) -> tuple[Tensor, dict[str, Tensor]]:
        """Return ``(loss, metrics)`` for a batch. Implemented by subclasses."""
        raise NotImplementedError

    def training_step(self, batch: MEDSTorchBatch, batch_idx: int) -> Tensor:
        loss, metrics = self.compute_loss(batch)
        self.log("train/loss", loss, prog_bar=True, batch_size=batch.batch_size)
        for k, v in metrics.items():
            self.log(f"train/{k}", v, batch_size=batch.batch_size)
        return loss

    def validation_step(self, batch: MEDSTorchBatch, batch_idx: int) -> Tensor:
        loss, metrics = self.compute_loss(batch)
        self.log("val/loss", loss, prog_bar=True, batch_size=batch.batch_size, sync_dist=True)
        for k, v in metrics.items():
            self.log(f"val/{k}", v, batch_size=batch.batch_size, sync_dist=True)
        return loss

    def configure_optimizers(self):
        if self._optimizer_factory is not None:
            optimizer = self._optimizer_factory(self.parameters())
        else:
            optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3)
        if self._scheduler_factory is None:
            return optimizer
        scheduler = self._scheduler_factory(optimizer)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}
