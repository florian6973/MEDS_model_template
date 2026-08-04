"""The MEDS-batch adapter layer: reading a ``MEDSTorchBatch``, and the Lightning contract base.

Everything here is about the *data format* or the *command contract*, not about modelling. Encoders,
heads and losses belong in a generated repo's ``model.py``, where they can be edited without fighting
``copier update``.

The batch tensors this layer speaks to (default ``SM`` — subject-measurement, flattened — mode):

- ``batch.code`` — ``[B, L]`` int64 code (vocab) indices; ``PAD_INDEX == 0`` marks padding.
- ``batch.numeric_value`` / ``batch.numeric_value_mask`` — ``[B, L]`` float value + presence mask.
- ``batch.time_delta_days`` — ``[B, L]`` float inter-measurement gaps.
- ``batch.boolean_value`` — ``[B]`` labels (present only when the datamodule has a ``task_labels_dir``).

If that format changes, this module changes once and ``copier update`` propagates it. That is the whole
reason these few pieces are contract rather than model code.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, ClassVar

import lightning.pytorch as L
import torch
from torch import Tensor, nn

if TYPE_CHECKING:  # pragma: no cover - typing only
    from torch.optim import Optimizer
    from torch.optim.lr_scheduler import LRScheduler

try:
    from meds_torchdata import MEDSTorchBatch

    PAD_INDEX = MEDSTorchBatch.PAD_INDEX
except ImportError:
    # meds-torch-data is optional (a data_backend=custom_featurization repo does not install it). This
    # module must still *import* — the generated model stub subclasses BaseLightningModule — but the
    # MTD-batch helpers (padding_mask, CodeEmbedder) are then unusable, which is correct: there is no
    # MEDSTorchBatch to hand them. PAD_INDEX keeps MTD's value; annotations are strings (PEP 563).
    MEDSTorchBatch = None
    PAD_INDEX = 0


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

    This is the batch-to-tensor boundary — the one place that has to know which fields a MEDS batch
    carries. What happens to the resulting tensor is your model's business.

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


class BaseLightningModule(L.LightningModule):
    """Shared Lightning plumbing plus the two hooks the commands read.

    Subclasses implement :meth:`compute_loss` (returning ``(loss, metrics_dict)``); this base handles
    ``training_step`` / ``validation_step`` and ``configure_optimizers``. ``optimizer`` / ``scheduler`` are
    *partials* (from ``_partial_: true`` Hydra configs) — an optimizer factory ``params -> Optimizer`` and a
    scheduler factory ``optimizer -> LRScheduler``.

    Two hooks separate the two things a trained model is asked for downstream:

    - ``predict_step`` — task probabilities, consumed by ``predict``.
    - :meth:`infer_step` — reusable per-timepoint artifacts, consumed by ``infer``. The default emits the
      pooled representation, which is what a probe needs; a generative model overrides it to emit
      trajectories. :attr:`inference_kind` declares which, and is recorded in the artifact manifest so a
      consumer can reject artifacts of the wrong kind instead of misinterpreting them.
    """

    #: What :meth:`infer_step` produces; see ``meds_model_base.manifest.InferenceKind``.
    inference_kind: ClassVar[str] = "embeddings"

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

    def infer_step(self, batch: MEDSTorchBatch) -> dict[str, Tensor]:
        """Per-timepoint reusable outputs. Defaults to the pooled representation from ``encode``."""
        encode = getattr(self, "encode", None)
        if encode is None:
            raise NotImplementedError(
                f"{type(self).__name__} defines neither `encode` nor `infer_step`, so `infer` does not know "
                "what to materialize. Implement one of them in your model."
            )
        return {"embedding": encode(batch).detach().cpu()}

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
