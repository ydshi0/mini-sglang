"""Verification / acceptance: turn one target forward over the ``gamma+1`` window
into (accepted-prefix-length, bonus token).

Pure tensor logic, unit-testable on CPU. Ports the acceptance rules from
``sglang/srt/speculative/dspark_components/kernels/dspark_accept.py`` and
``dflash_utils.compute_dflash_correct_drafts_and_bonus``.

Window layout per request (all length ``W = gamma + 1``):
    col 0        : anchor  = the previous step's bonus token (already accepted)
    col 1..gamma : the gamma draft tokens proposed this step

The target forward over the window yields ``target_predict[:, k]`` = the token the
target itself would emit given the prefix through window position ``k``. A draft
token at block position ``k`` (1-indexed) is accepted iff it equals the target's
prediction at the *previous* position, and acceptance stops at the first mismatch
(the "longest correct prefix" rule). The token the target predicts at the mismatch
(or one past the last draft, if all matched) becomes the free **bonus** token, so a
step always commits at least one and at most ``gamma + 1`` tokens.
"""

from __future__ import annotations

from typing import List, NamedTuple, Optional

import torch


class AcceptResult(NamedTuple):
    correct_len: torch.Tensor  # [bs] int64, number of accepted draft tokens in [0, gamma]
    bonus: torch.Tensor        # [bs] int64, the free bonus token appended after the prefix
    commit_len: torch.Tensor   # [bs] int64, tokens committed this step == correct_len + 1


def accept_greedy(draft_tokens: torch.Tensor, target_predict: torch.Tensor) -> AcceptResult:
    """Longest-correct-prefix acceptance for greedy decoding.

    Args:
        draft_tokens:   [bs, gamma] the gamma proposed tokens.
        target_predict: [bs, gamma + 1] target argmax at each window position.

    Returns:
        AcceptResult with ``correct_len`` = length of the leading run where
        ``draft_tokens[:, k] == target_predict[:, k]`` (0-indexed over the gamma
        drafts), and ``bonus = target_predict[arange, correct_len]``.
    """
    bs, gamma = draft_tokens.shape
    assert target_predict.shape == (bs, gamma + 1), (
        f"expected target_predict [bs, gamma+1]=({bs},{gamma + 1}), "
        f"got {tuple(target_predict.shape)}"
    )
    # matches[:, k] : draft token k == what the target predicted after the same prefix.
    matches = (draft_tokens == target_predict[:, :gamma]).to(torch.int32)
    # cumprod then sum == length of the leading all-ones run (stops at first 0).
    correct_len = matches.cumprod(dim=1).sum(dim=1).to(torch.int64)  # [bs] in [0, gamma]
    bonus = target_predict.gather(1, correct_len.unsqueeze(1)).squeeze(1)  # [bs]
    commit_len = correct_len + 1
    return AcceptResult(correct_len=correct_len, bonus=bonus, commit_len=commit_len)


def accept_sampling(
    draft_tokens: torch.Tensor,
    draft_probs: torch.Tensor,
    target_probs: torch.Tensor,
    *,
    uniforms: Optional[torch.Tensor] = None,
) -> AcceptResult:
    """Chain speculative sampling (lossless) for temperature/top-p decoding.

    Standard speculative-sampling acceptance (Leviathan et al. 2023): draft token
    ``t_k`` with draft prob ``q`` and target prob ``p`` is accepted with prob
    ``min(1, p/q)``; on the first rejection we sample the bonus from the residual
    ``normalize(relu(p - q))`` and stop. If the whole block is accepted the bonus
    is drawn from the target distribution one position past the last draft.

    Args:
        draft_tokens: [bs, gamma]
        draft_probs:  [bs, gamma, vocab]     q at each of the gamma draft positions
        target_probs: [bs, gamma + 1, vocab] p at each window position (incl. trailing)
        uniforms:     [bs, gamma] optional pre-sampled U(0,1) (for deterministic tests)
    """
    bs, gamma = draft_tokens.shape
    device = draft_tokens.device
    if uniforms is None:
        uniforms = torch.rand(bs, gamma, device=device)

    correct_len = torch.zeros(bs, dtype=torch.int64, device=device)
    bonus = torch.zeros(bs, dtype=torch.int64, device=device)
    done = torch.zeros(bs, dtype=torch.bool, device=device)

    ar = torch.arange(bs, device=device)
    for k in range(gamma):
        tok = draft_tokens[:, k]
        p = target_probs[ar, k, tok]
        q = draft_probs[ar, k, tok].clamp_min(1e-20)
        accept = (uniforms[:, k] < (p / q)) & (~done)
        # advance the accepted-prefix counter for still-alive, accepting rows
        correct_len = torch.where(accept, correct_len + 1, correct_len)
        # rows that reject *now* (and were not already done) sample the residual bonus
        reject_now = (~accept) & (~done)
        if reject_now.any():
            residual = torch.relu(target_probs[:, k, :] - draft_probs[:, k, :])
            residual = residual / residual.sum(dim=-1, keepdim=True).clamp_min(1e-20)
            sampled = torch.multinomial(residual, 1).squeeze(1)
            bonus = torch.where(reject_now, sampled, bonus)
            done = done | reject_now
    # rows that accepted the whole block draw the bonus from the trailing target dist
    all_accepted = ~done
    if all_accepted.any():
        sampled = torch.multinomial(target_probs[:, gamma, :], 1).squeeze(1)
        bonus = torch.where(all_accepted, sampled, bonus)

    return AcceptResult(correct_len=correct_len, bonus=bonus, commit_len=correct_len + 1)


def gather_committed_tokens(
    draft_tokens: torch.Tensor, result: AcceptResult
) -> List[List[int]]:
    """Materialize the per-request committed token ids (accepted prefix + bonus).

    Host-side helper used by the worker to append tokens to each request. Returns
    a python list because requests commit different numbers of tokens per step.
    """
    draft_cpu = draft_tokens.tolist()
    correct = result.correct_len.tolist()
    bonus = result.bonus.tolist()
    out: List[List[int]] = []
    for i in range(len(correct)):
        out.append(draft_cpu[i][: correct[i]] + [bonus[i]])
    return out
