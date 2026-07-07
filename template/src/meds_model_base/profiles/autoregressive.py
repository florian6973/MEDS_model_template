"""The zero-shot autoregressive profile: an "everything-is-code" causal LM over MEDS token sequences.

``AutoregressiveModel`` is a small GPT-style causal Transformer trained with next-code cross-entropy
(``unsupervised_train``). At ``task_agnostic_inference`` it autoregressively **generates** future codes for
each subject; a downstream zero-shot ``prediction`` resolves a task over those generations (that resolution
is model-specific and external — see ``ZeroShotPredictionStep``).

This is a deliberately compact, readable reference — not a performance-tuned generator (it generates
codes only, greedily, and does not special-case padding/EOS the way a production AR model would). A
generated repo subclasses it in ``model.py`` and can override generation.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn.functional as F
from meds_torchdata import MEDSTorchBatch
from torch import Tensor, nn

from ..lightning.modules import PAD_INDEX, BaseLightningModule


class AutoregressiveModel(BaseLightningModule):
    """A causal-Transformer language model over MEDS code sequences.

    Args:
        vocab_size: code-vocabulary size (injected from ``datamodule.config.vocab_size``).
        d_model, nhead, num_layers, dropout: Transformer size.
        max_seq_len: maximum context length (for the learned positional embedding).
        max_new_tokens: number of codes to generate per subject at inference.
        optimizer / scheduler: Hydra ``_partial_`` factories.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        max_seq_len: int = 256,
        max_new_tokens: int = 32,
        optimizer: Callable | None = None,
        scheduler: Callable | None = None,
    ) -> None:
        super().__init__(optimizer=optimizer, scheduler=scheduler)
        self.save_hyperparameters(ignore=["optimizer", "scheduler"])

        self.vocab_size = vocab_size
        self.max_new_tokens = max_new_tokens
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=PAD_INDEX)
        self.pos = nn.Embedding(max_seq_len, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward=4 * d_model, dropout=dropout, batch_first=True, norm_first=True
        )
        self.blocks = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.ln = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)

    def _hidden(self, codes: Tensor) -> Tensor:
        seq_len = codes.shape[1]
        positions = torch.arange(seq_len, device=codes.device)
        x = self.embed(codes) + self.pos(positions).unsqueeze(0)
        causal = nn.Transformer.generate_square_subsequent_mask(seq_len, device=codes.device)
        pad = codes == PAD_INDEX
        h = self.blocks(x, mask=causal, src_key_padding_mask=pad, is_causal=True)
        return self.ln(h)

    def forward(self, codes: Tensor) -> Tensor:
        """Next-code logits ``[B, L, vocab_size]``."""
        return self.lm_head(self._hidden(codes))

    def compute_loss(self, batch: MEDSTorchBatch) -> tuple[Tensor, dict[str, Tensor]]:
        codes = batch.code
        logits = self(codes[:, :-1])
        target = codes[:, 1:]
        loss = F.cross_entropy(
            logits.reshape(-1, self.vocab_size), target.reshape(-1), ignore_index=PAD_INDEX
        )
        return loss, {}

    @torch.no_grad()
    def generate(self, codes: Tensor, max_new_tokens: int) -> Tensor:
        """Greedily append ``max_new_tokens`` codes to each row of ``codes`` (``[B, L] → [B, L + T]``)."""
        max_ctx = self.pos.num_embeddings
        for _ in range(max_new_tokens):
            logits = self(codes[:, -max_ctx:])[:, -1]
            next_code = logits.argmax(dim=-1, keepdim=True)
            codes = torch.cat([codes, next_code], dim=1)
        return codes

    def predict_step(self, batch: MEDSTorchBatch, batch_idx: int = 0) -> dict[str, Tensor]:
        """Generate future codes for each subject (used by ``task_agnostic_inference``)."""
        generated = self.generate(batch.code, self.max_new_tokens)
        return {"generated_codes": generated.detach().cpu()}
