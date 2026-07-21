"""DSpark semi-autoregressive heads: the serial Markov head and the confidence head.

Faithful port of ``sglang/srt/models/dspark.py`` (``VanillaMarkov``,
``GatedMarkovHead``, ``RNNHead``, ``DSparkConfidenceHead``, ``run_markov_block``),
rewritten in mini-sglang's ``BaseOP`` + bare-``torch.Tensor`` parameter style so the
weights participate in ``state_dict`` / ``load_state_dict`` like the rest of the engine.

Design in one paragraph: the DFlash backbone proposes the base logits of a whole
``gamma``-wide block in parallel (fast, but positions are conditionally independent,
so acceptance decays with depth). The Markov head fixes that by adding a small,
data-dependent bias to each position's logits that depends on the *previous*
sampled token, and the block is sampled serially position-by-position. That single
serial dependency is what makes DSpark "semi-autoregressive" and is why its
accepted-block length beats a purely parallel drafter (DFlash) by ~16-18%.
"""

from __future__ import annotations

from typing import Callable, List, Optional, Tuple

import torch
import torch.nn.functional as F
from minisgl.layers.base import BaseOP

# A StepSampler maps (step_logits[bs, vocab], step_idx) -> next_tokens[bs].
StepSampler = Callable[[torch.Tensor, int], torch.Tensor]


class VanillaMarkov(BaseOP):
    """logits_bias(prev_token) = W2 @ (W1[prev_token]).

    W1 is an embedding table ``[vocab, rank]`` and W2 is a linear ``[vocab, rank]``.
    Checkpoint keys ``markov_head.markov_w1.weight`` / ``markov_head.markov_w2.weight``
    are remapped to ``w1`` / ``w2`` by the draft loader.
    """

    markov_head_type = "vanilla"

    def __init__(self, *, vocab_size: int, markov_rank: int, hidden_size: int = 0) -> None:
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.markov_rank = int(markov_rank)
        # w1: embedding weight [vocab, rank];  w2: linear weight [vocab, rank]
        self.w1 = torch.empty(self.vocab_size, self.markov_rank)
        self.w2 = torch.empty(self.vocab_size, self.markov_rank)

    def prev_embeddings(self, token_ids: torch.Tensor) -> torch.Tensor:
        return F.embedding(token_ids.long(), self.w1)

    def project_bias(self, latent: torch.Tensor) -> torch.Tensor:
        return F.linear(latent, self.w2)

    def step_bias(
        self, token_ids: torch.Tensor, hidden_states: Optional[torch.Tensor]
    ) -> torch.Tensor:
        del hidden_states  # unused by the vanilla head
        return self.project_bias(self.prev_embeddings(token_ids))

    def apply_step_logits(
        self,
        logits: torch.Tensor,
        *,
        token_ids: torch.Tensor,
        hidden_states: Optional[torch.Tensor],
    ) -> torch.Tensor:
        return logits + self.step_bias(token_ids, hidden_states)

    def forward(self, *args, **kwargs):  # pragma: no cover - not used directly
        raise NotImplementedError("Use apply_step_logits / sample_block.")

    def sample_block(
        self,
        base_logits: torch.Tensor,
        *,
        first_prev_tokens: torch.Tensor,
        hidden_states: Optional[torch.Tensor],
        sampler: StepSampler,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return run_markov_block(
            self,
            base_logits,
            first_prev_tokens=first_prev_tokens,
            hidden_states=hidden_states,
            sampler=sampler,
        )


class GatedMarkovHead(VanillaMarkov):
    """Vanilla head with a sigmoid gate on the previous-token embedding.

    gate = sigmoid(gate_proj([hidden, prev_emb])); bias = W2 @ (gate * prev_emb).
    """

    markov_head_type = "gated"

    def __init__(self, *, vocab_size: int, markov_rank: int, hidden_size: int) -> None:
        super().__init__(vocab_size=vocab_size, markov_rank=markov_rank)
        self.hidden_size = int(hidden_size)
        # gate_proj: Linear(hidden + rank, rank)
        self.gate_proj_w = torch.empty(markov_rank, self.hidden_size + markov_rank)
        self.gate_proj_b = torch.empty(markov_rank)

    def step_bias(
        self, token_ids: torch.Tensor, hidden_states: Optional[torch.Tensor]
    ) -> torch.Tensor:
        if hidden_states is None:
            raise ValueError("GatedMarkovHead requires hidden_states.")
        prev_emb = self.prev_embeddings(token_ids)
        gate_in = torch.cat([hidden_states, prev_emb], dim=-1)
        gate = torch.sigmoid(F.linear(gate_in, self.gate_proj_w, self.gate_proj_b))
        gate = gate.to(dtype=prev_emb.dtype)
        return self.project_bias(gate * prev_emb)


def run_markov_block(
    head: VanillaMarkov,
    base_logits: torch.Tensor,
    *,
    first_prev_tokens: torch.Tensor,
    hidden_states: Optional[torch.Tensor],
    sampler: StepSampler,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Serially sample a ``[bs, proposal_len]`` block from parallel ``base_logits``.

    Port of ``sglang/srt/models/dspark.py::run_markov_block``. At each step the
    head adds its previous-token bias to that position's base logits, the block
    sampler picks the token, and that token conditions the *next* step. Returns
    ``(sampled_tokens[bs, L], corrected_logits[bs, L, vocab])``.
    """
    batch_size, proposal_len = base_logits.shape[:2]
    if proposal_len == 0:
        empty = torch.empty(batch_size, 0, dtype=torch.long, device=base_logits.device)
        return empty, base_logits

    sampled_tokens: List[torch.Tensor] = []
    corrected_logits: List[torch.Tensor] = []
    prev_tokens = first_prev_tokens.long()
    for step_idx in range(proposal_len):
        step_hidden = None if hidden_states is None else hidden_states[:, step_idx, ...]
        step_logits = head.apply_step_logits(
            base_logits[:, step_idx, :], token_ids=prev_tokens, hidden_states=step_hidden
        )
        next_tokens = sampler(step_logits, step_idx)
        sampled_tokens.append(next_tokens)
        corrected_logits.append(step_logits.unsqueeze(1))
        prev_tokens = next_tokens
    return torch.stack(sampled_tokens, dim=1), torch.cat(corrected_logits, dim=1)


class DSparkConfidenceHead(BaseOP):
    """Predicts a per-position acceptance probability for confidence scheduling.

    ``proj: Linear(hidden + rank, 1)``. ``forward`` returns the raw score; the
    planner turns it into a probability with ``apply_sts`` (temperature-scaled
    sigmoid). Port of ``sglang/srt/models/dspark.py::DSparkConfidenceHead``.
    """

    def __init__(
        self, *, hidden_size: int, markov_rank: int, with_markov: bool = True
    ) -> None:
        super().__init__()
        self.with_markov = bool(with_markov)
        in_dim = int(hidden_size) + (int(markov_rank) if self.with_markov else 0)
        self.proj_w = torch.empty(1, in_dim)
        self.proj_b = torch.empty(1)
        # Non-persistent calibration temperature (STS). Fit offline; defaults to 1.
        self._sts_temperature = torch.ones((), dtype=torch.float32)

    def forward(
        self, hidden_states: torch.Tensor, markov_embed_stack: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if self.with_markov:
            if markov_embed_stack is None:
                raise ValueError("confidence head with_markov=True needs markov_embed_stack.")
            feats = torch.cat(
                [hidden_states, markov_embed_stack.to(dtype=hidden_states.dtype)], dim=-1
            )
        else:
            feats = hidden_states
        feats = feats.to(dtype=self.proj_w.dtype)
        return F.linear(feats, self.proj_w, self.proj_b).squeeze(-1)

    def apply_sts(self, confidence_raw: torch.Tensor) -> torch.Tensor:
        """Temperature-scaled sigmoid -> per-position acceptance probability in [0, 1]."""
        return torch.sigmoid(confidence_raw.float() / self._sts_temperature)


def build_markov_head(cfg) -> VanillaMarkov:
    """Instantiate the Markov head requested by a :class:`DSparkConfig`."""
    kw = dict(vocab_size=cfg.vocab_size, markov_rank=cfg.markov_rank, hidden_size=cfg.hidden_size)
    if cfg.markov_head_type == "vanilla":
        return VanillaMarkov(**kw)
    if cfg.markov_head_type == "gated":
        return GatedMarkovHead(**kw)
    raise ValueError(f"markov_head_type={cfg.markov_head_type!r} not implemented in mini-sglang.")


def build_confidence_head(cfg) -> Optional[DSparkConfidenceHead]:
    if not cfg.require_confidence():
        return None
    return DSparkConfidenceHead(
        hidden_size=cfg.hidden_size,
        markov_rank=cfg.markov_rank,
        with_markov=cfg.confidence_head_with_markov,
    )
