"""The zero-shot autoregressive profile: an "everything-is-code" causal LM over MEDS token sequences.

``AutoregressiveModel`` is a small GPT-style causal Transformer trained with next-code cross-entropy
(``pretrain``). At ``infer`` it autoregressively **generates** future codes for each subject, published as
an inference artifact of kind ``trajectories``; a downstream ``predict`` resolves a task over those
generations (that resolution is model-specific — see
:class:`~meds_model_base.commands.predict.ZeroShotPredictCommand`).

This is a deliberately compact, readable reference — not a performance-tuned generator (it generates
codes only, greedily, and does not special-case padding/EOS the way a production AR model would). A
generated repo subclasses it in ``model.py`` and can override generation.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import ClassVar

import torch
import torch.nn.functional as F
from meds_torchdata import MEDSTorchBatch
from torch import Tensor, nn

from ..commands.predict import ZeroShotPredictCommand
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

    #: `infer` materializes generated futures.
    inference_kind: ClassVar[str] = "trajectories"

    def infer_step(self, batch: MEDSTorchBatch) -> dict[str, Tensor]:
        """Generate future codes for each subject (published as an inference artifact)."""
        return {"generated_codes": self.generate(batch.code, self.max_new_tokens).detach().cpu()}

    def predict_step(self, batch: MEDSTorchBatch, batch_idx: int = 0) -> dict[str, Tensor]:
        """Same generations as :meth:`infer_step`, for a `predict` that resolves a task in-process."""
        return self.infer_step(batch)


class AutoregressiveZeroShotPredictCommand(ZeroShotPredictCommand):
    """Resolve a task over the model's own generated futures — the MEDS-EIC-AR approach.

    Samples ``n_samples`` trajectories per subject and reports the fraction in which the task's target code
    appears. That fraction *is* the zero-shot probability: no task-specific training, no calibration, just
    the model's own beliefs about what happens next.

    Sampling (not greedy decoding) is what makes the estimate meaningful — a greedy trajectory yields only
    0.0 or 1.0, which is a decision rather than a probability.
    """

    #: Number of sampled futures per subject. More samples, finer probability resolution.
    n_samples: int = 16

    def resolve(self, cfg, module, index):
        import polars as pl

        from ..commands._runtime import KEYS, split_dataset_and_loader
        from ..commands.predict import PROBABILITY_COLUMN

        target = self._target_vocab_index(cfg)
        frames = []
        for split in index["split"].unique(maintain_order=True).to_list():
            dataset, dataloader = split_dataset_and_loader(cfg, split)
            keys = dataset.schema_df.select(KEYS)
            if not len(keys):
                continue
            probs: list[float] = []
            for batch in dataloader:
                probs.extend(self._occurrence_probability(module, batch.code, target))
            frames.append(
                keys.with_columns(
                    pl.Series(PROBABILITY_COLUMN, probs, dtype=pl.Float32),
                    pl.lit(split).alias("split"),
                )
            )
        if not frames:
            raise ValueError("The datamodule produced no rows for any requested split.")
        return pl.concat(frames, how="vertical_relaxed").select([*KEYS, "split", PROBABILITY_COLUMN])

    @torch.no_grad()
    def _occurrence_probability(self, module, codes: Tensor, target: int) -> list[float]:
        """Fraction of sampled futures in which ``target`` occurs, per row of ``codes``."""
        batch_size = codes.shape[0]
        hits = torch.zeros(batch_size)
        for _ in range(self.n_samples):
            sampled = _sample_future(module, codes, module.max_new_tokens)
            hits += (sampled == target).any(dim=1).float().cpu()
        return (hits / self.n_samples).tolist()

    @staticmethod
    def _target_vocab_index(cfg) -> int:
        """The vocabulary index of the task's target code, or a clear error."""
        from ..commands.predict import task_definition_path
        from ..tasks import load_task_config
        from ..utils import resolve_subdir
        from .every_query import _first_plain_code, _load_vocab

        task_dir = resolve_subdir(cfg.input_data_dir, cfg.input_task_subdir)
        definition = task_definition_path(task_dir)
        if definition is None:
            raise ValueError(
                f"The task at {task_dir} was materialized from pre-extracted labels, so it carries no "
                "definition of *what* to predict. A zero-shot model needs one: re-run preprocess_task with "
                "the ACES YAML as external_task_file."
            )
        code = _first_plain_code(load_task_config(definition))
        if code is None:
            raise ValueError(f"No plain-predicate code found in {definition} to resolve against.")
        vocab = _load_vocab(Path(cfg.input_data_dir) / "patients")
        if code not in vocab:
            raise ValueError(
                f"Task code {code!r} is absent from this cohort's vocabulary, so it can never appear in a "
                "generated trajectory and the task cannot be resolved."
            )
        return int(vocab[code])


@torch.no_grad()
def _sample_future(module, codes: Tensor, max_new_tokens: int) -> Tensor:
    """Sample (rather than greedily decode) ``max_new_tokens`` continuations for each row of ``codes``."""
    max_ctx = module.pos.num_embeddings
    generated = codes
    for _ in range(max_new_tokens):
        logits = module(generated[:, -max_ctx:])[:, -1]
        next_code = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
        generated = torch.cat([generated, next_code], dim=1)
    return generated[:, codes.shape[1] :]
