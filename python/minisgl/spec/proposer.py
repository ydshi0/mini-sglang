"""Draft-block proposal: build the mask block, and serially sample the gamma draft
tokens with the Markov head.

The parallel backbone forward that turns the block token embeddings into hidden
states lives in the worker (it needs the draft forward context / KV). Everything
here operates on the resulting ``block_hidden`` and is pure head logic, so it is
unit-testable on CPU with random tensors.

Ports the head-sampling half of ``DraftBlockProposer`` in
``sglang/srt/speculative/dspark_components/dspark_draft.py``.
"""

from __future__ import annotations

from typing import Callable, NamedTuple, Optional

import torch

from .heads import StepSampler


class DraftProposal(NamedTuple):
    draft_tokens: torch.Tensor       # [bs, gamma] proposed token ids
    corrected_logits: torch.Tensor   # [bs, gamma, vocab] Markov-corrected logits
    confidence_probs: Optional[torch.Tensor]  # [bs, gamma] per-position accept prob, or None


def make_greedy_sampler() -> StepSampler:
    def sampler(step_logits: torch.Tensor, step_idx: int) -> torch.Tensor:
        del step_idx
        return step_logits.argmax(dim=-1)

    return sampler


def make_temperature_sampler(temperature: float) -> StepSampler:
    if temperature <= 0:
        return make_greedy_sampler()

    def sampler(step_logits: torch.Tensor, step_idx: int) -> torch.Tensor:
        del step_idx
        probs = torch.softmax(step_logits.float() / temperature, dim=-1)
        return torch.multinomial(probs, 1).squeeze(-1)

    return sampler


def build_mask_block(
    bonus_tokens: torch.Tensor, gamma: int, mask_token_id: int
) -> torch.Tensor:
    """``[bs, gamma]`` block: slot 0 = last step's bonus token, slots 1.. = mask token.

    The parallel backbone fills in the real content; slot 0 anchors the block on the
    already-accepted token so the first draft is conditioned correctly.
    """
    bs = bonus_tokens.shape[0]
    block = torch.full((bs, gamma), mask_token_id, dtype=torch.int64, device=bonus_tokens.device)
    block[:, 0] = bonus_tokens.view(-1)
    return block


def sample_draft_block(
    draft_model,
    block_hidden: torch.Tensor,
    first_prev_tokens: torch.Tensor,
    sampler: StepSampler,
) -> DraftProposal:
    """Serially sample the gamma draft tokens from the backbone hidden states.

    Args:
        draft_model:       :class:`~minisgl.spec.draft_model.DSparkDraftModel`
        block_hidden:      [bs, gamma, hidden] backbone output over the block
        first_prev_tokens: [bs] the anchor (previous bonus) token
        sampler:           per-step token sampler (greedy or temperature)
    """
    base_logits = draft_model.compute_base_logits(block_hidden)  # [bs, gamma, vocab]
    draft_tokens, corrected_logits = draft_model.markov_head.sample_block(
        base_logits,
        first_prev_tokens=first_prev_tokens,
        hidden_states=block_hidden,
        sampler=sampler,
    )

    confidence_probs: Optional[torch.Tensor] = None
    if draft_model.confidence_head is not None:
        # markov_embed_stack: embedding of the token conditioning each position
        # (anchor for position 0, previous draft token thereafter).
        cond_tokens = torch.cat(
            [first_prev_tokens.view(-1, 1), draft_tokens[:, :-1]], dim=1
        )
        markov_embed_stack = draft_model.markov_head.prev_embeddings(cond_tokens)
        confidence_probs = draft_model.confidence_probs(block_hidden, markov_embed_stack)

    return DraftProposal(
        draft_tokens=draft_tokens,
        corrected_logits=corrected_logits,
        confidence_probs=confidence_probs,
    )
