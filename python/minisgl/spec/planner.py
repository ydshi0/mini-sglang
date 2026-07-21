"""Confidence-scheduled, load-aware verification planning — DSpark's signature
serving-side contribution.

A purely-static spec-decode engine always verifies the full ``gamma + 1`` window
for every request. Under high concurrency that wastes target FLOPs on deep block
positions that are very unlikely to be accepted. DSpark instead:

  1. reads a per-position acceptance probability from the draft's confidence head,
  2. turns it into a per-position *survival* probability (cumulative product), and
  3. spends a *global* verify-token budget on the positions most likely to pay off,
     where the budget itself is chosen to maximize throughput
     ``theta = expected_accepted_tokens / step_time`` against a profiled cost model.

This is a faithful, minimal port of the math in
``sglang/srt/speculative/dspark_components/dspark_planner.py``
(``compute_confidence`` -> ``compute_verify_token_budget`` ->
``schedule_verify_lens_topk_from_survival``). The heavy production machinery
(CUDA-graph tier alignment, cross-DP reduction, confidence-relay lag) is omitted;
the scheduling *decision* is identical. All tensor logic, CPU-unit-testable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Union

import torch


def compute_survival(confidence_probs: torch.Tensor) -> torch.Tensor:
    """Per-position survival = probability the block is still alive *and* accepted here.

    Args:
        confidence_probs: [bs, gamma] per-position acceptance probs in [0, 1]
            (the confidence head's ``apply_sts`` output).
    Returns:
        survival: [bs, gamma], monotonically non-increasing along dim 1.
    """
    return confidence_probs.clamp(0.0, 1.0).cumprod(dim=1)


@dataclass
class SpsCostModel:
    """Steps-per-second as a function of the total verify tokens in a batch.

    Models step time as affine in the number of verified tokens:
        step_time(n) = base_ms + per_token_ms * n     ->     sps(n) = 1000 / step_time(n)

    In production this comes from an offline profile
    (``benchmark/dspark_sps_profiler.py`` -> ``--speculative-dspark-sps-table-path``).
    A flat/degenerate model makes the throughput-optimal budget collapse to
    "verify everything", exactly matching the STATIC baseline.
    """

    base_ms: float = 6.0
    per_token_ms: float = 0.03

    def sps(self, num_tokens: Union[torch.Tensor, float]) -> Union[torch.Tensor, float]:
        if isinstance(num_tokens, torch.Tensor):
            return 1000.0 / (self.base_ms + self.per_token_ms * num_tokens.clamp_min(0))
        return 1000.0 / (self.base_ms + self.per_token_ms * max(num_tokens, 0.0))


def compute_verify_token_budget(
    survival: torch.Tensor,
    cost: SpsCostModel,
    *,
    base_tokens: int = 0,
) -> int:
    """Choose the number of *extra* draft-verify tokens that maximizes throughput.

    Sorting all ``[bs, gamma]`` survival probs descending, the expected number of
    accepted tokens after granting the top ``B`` positions is
        tau_star(B) = num_requests + cumsum(sorted_survival)[B]
    (each request always yields its guaranteed bonus token, hence ``num_requests``).
    Throughput ``theta(B) = tau_star(B) * sps(base_tokens + num_requests + B)``.
    We return ``argmax_B theta(B)``.

    ``base_tokens`` accounts for the committed-prefix KV already in the batch.
    """
    bs, gamma = survival.shape
    num_requests = bs
    sorted_surv, _ = torch.sort(survival.reshape(-1), descending=True)
    # tau_star[B] for B = 0..bs*gamma
    cumsum = torch.cat([torch.zeros(1, device=survival.device), sorted_surv.cumsum(0)])
    tau_star = num_requests + cumsum  # [N+1]
    n_extra = torch.arange(tau_star.numel(), device=survival.device, dtype=torch.float32)
    # anchor token per request is always verified -> +num_requests fixed cost
    total_tokens = base_tokens + num_requests + n_extra
    theta = tau_star * cost.sps(total_tokens)
    return int(torch.argmax(theta).item())


def schedule_verify_lens(
    survival: torch.Tensor,
    budget: int,
    *,
    min_verify_len: int = 1,
) -> torch.Tensor:
    """Distribute a global verify-token ``budget`` across requests by top-k survival.

    Because survival is monotonically non-increasing within each request, taking
    the globally-highest ``budget`` survival entries automatically yields a
    *contiguous prefix* per request — so per-request verify length is just the
    count of that request's chosen positions, clamped to ``[min_verify_len, gamma]``.

    Returns:
        verify_lens: [bs] int64, number of draft positions each request verifies.
    """
    bs, gamma = survival.shape
    budget = max(0, min(budget, bs * gamma))
    flat = survival.reshape(-1)
    if budget == 0:
        counts = torch.zeros(bs, dtype=torch.int64, device=survival.device)
    else:
        topk_idx = torch.topk(flat, budget, largest=True, sorted=False).indices
        rows = topk_idx // gamma
        counts = torch.bincount(rows, minlength=bs).to(torch.int64)
    return counts.clamp_(min=min_verify_len, max=gamma)


@dataclass
class VerifyPlan:
    verify_lens: torch.Tensor  # [bs] number of draft positions verified per request
    budget: int                # total extra verify tokens granted
    survival: torch.Tensor     # [bs, gamma] (for observability)

    @property
    def max_verify_len(self) -> int:
        return int(self.verify_lens.max().item()) if self.verify_lens.numel() else 0


def plan_verify(
    confidence_probs: torch.Tensor,
    cost: SpsCostModel,
    *,
    base_tokens: int = 0,
    min_verify_len: int = 1,
) -> VerifyPlan:
    """End-to-end confidence-scheduled plan for one decode step (COMPACT mode)."""
    survival = compute_survival(confidence_probs)
    budget = compute_verify_token_budget(survival, cost, base_tokens=base_tokens)
    verify_lens = schedule_verify_lens(survival, budget, min_verify_len=min_verify_len)
    return VerifyPlan(verify_lens=verify_lens, budget=budget, survival=survival)
