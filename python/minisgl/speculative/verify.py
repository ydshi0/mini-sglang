"""
Speculative Decoding Verification Algorithms.

Two strategies for comparing draft tokens against target model predictions:

1. Greedy: accept if target's argmax matches draft. Guarantees output
   identical to standalone target model (for temperature=0).

2. Stochastic (Rejection Sampling): accept with probability
   min(1, p_target / p_draft). Preserves target's distribution exactly.
   On rejection, resample from residual: p'(x) ∝ max(0, p_target - p_draft).

Both functions expect target_logits of shape (K+1, vocab_size):
    logits[0] verifies draft_tokens[0]
    logits[i] verifies draft_tokens[i]
    logits[K] produces the bonus token (if all K accepted)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import torch


@dataclass
class VerifyResult:
    """Result of speculative verification."""
    accepted_tokens: List[int]   # accepted token ids (may include corrected/bonus)
    num_draft_accepted: int      # how many of the K draft tokens matched
    bonus_token: int | None      # bonus token if all K accepted, else None
    total_accepted: int          # total new tokens this round


def verify_greedy(
    target_logits: torch.Tensor,
    draft_tokens: List[int],
) -> VerifyResult:
    """
    Greedy verification.

    Args:
        target_logits: (K+1, vocab_size). logits[i] verifies draft_tokens[i].
        draft_tokens: K draft token ids.

    Returns:
        VerifyResult.
    """
    K = len(draft_tokens)
    assert target_logits.shape[0] >= K + 1, (
        f"Need K+1={K+1} logits, got {target_logits.shape[0]}"
    )

    accepted: List[int] = []
    for i in range(K):
        target_token = torch.argmax(target_logits[i]).item()
        if target_token == draft_tokens[i]:
            accepted.append(draft_tokens[i])
        else:
            # Reject: use target's prediction as the corrected token
            accepted.append(target_token)
            return VerifyResult(
                accepted_tokens=accepted,
                num_draft_accepted=i,
                bonus_token=None,
                total_accepted=i + 1,
            )

    # All K accepted → sample bonus token from logits[K]
    bonus = torch.argmax(target_logits[K]).item()
    accepted.append(bonus)
    return VerifyResult(
        accepted_tokens=accepted,
        num_draft_accepted=K,
        bonus_token=bonus,
        total_accepted=K + 1,
    )


def verify_stochastic(
    target_logits: torch.Tensor,
    draft_tokens: List[int],
    draft_probs: List[torch.Tensor],
    temperature: float = 1.0,
) -> VerifyResult:
    """
    Stochastic verification via rejection sampling.

    Preserves the target model's probability distribution exactly.

    Args:
        target_logits: (K+1, vocab_size).
        draft_tokens: K draft tokens.
        draft_probs: K probability distributions from the draft model (CPU).
        temperature: Sampling temperature.

    Returns:
        VerifyResult.
    """
    K = len(draft_tokens)
    assert target_logits.shape[0] >= K + 1

    # Compute target probabilities
    if temperature > 0:
        target_probs = torch.softmax(target_logits[:K] / temperature, dim=-1)
    else:
        target_probs = torch.softmax(target_logits[:K], dim=-1)

    accepted: List[int] = []
    for i in range(K):
        d_i = draft_tokens[i]
        p_target = target_probs[i, d_i].item()
        p_draft = draft_probs[i][d_i].item()

        # Acceptance probability
        if p_draft < 1e-10:
            accept_prob = 1.0 if p_target > 1e-10 else 0.0
        else:
            accept_prob = min(1.0, p_target / p_draft)

        if torch.rand(1).item() <= accept_prob:
            accepted.append(d_i)
        else:
            # Reject → resample from residual distribution
            residual = torch.clamp(target_probs[i].cpu() - draft_probs[i], min=0.0)
            total = residual.sum()
            if total > 1e-10:
                resampled = torch.multinomial(residual / total, num_samples=1).item()
            else:
                resampled = torch.multinomial(target_probs[i].cpu(), num_samples=1).item()
            accepted.append(resampled)
            return VerifyResult(
                accepted_tokens=accepted,
                num_draft_accepted=i,
                bonus_token=None,
                total_accepted=i + 1,
            )

    # All K accepted → bonus token
    if temperature > 0:
        bonus_probs = torch.softmax(target_logits[K] / temperature, dim=-1)
        bonus = torch.multinomial(bonus_probs.cpu(), num_samples=1).item()
    else:
        bonus = torch.argmax(target_logits[K]).item()
    accepted.append(bonus)

    return VerifyResult(
        accepted_tokens=accepted,
        num_draft_accepted=K,
        bonus_token=bonus,
        total_accepted=K + 1,
    )