"""DSpark speculative-decoding worker: the host-side draft -> verify -> accept ->
commit loop for one decode step.

This orchestrator is deliberately engine-agnostic. The two GPU-heavy operations —
running the parallel draft backbone, and running the target over the verify window —
are injected as callables (:data:`DraftBackboneFn`, :data:`TargetVerifyFn`). This
keeps the *algorithm* (which is the interesting, DSpark-specific part) concrete and
testable with fakes, while the mini-sglang engine wiring that implements those two
callables lives behind a clear interface (see ``docs/dspark_speculative_decoding.md``
section "Engine integration", which is the part still pending on-GPU validation).

Ports the control flow of ``DSparkWorkerV2._forward_decode`` in
``sglang/srt/speculative/dspark_components/dspark_worker_v2.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, NamedTuple, Optional

import torch

from .config import DSparkConfig, RaggedVerifyMode
from .planner import SpsCostModel, plan_verify
from .proposer import (
    build_mask_block,
    make_greedy_sampler,
    make_temperature_sampler,
    sample_draft_block,
)
from .verify import accept_greedy, accept_sampling, gather_committed_tokens


class TargetVerifyOut(NamedTuple):
    logits: torch.Tensor  # [bs, W, vocab]
    hidden: torch.Tensor  # [bs, W, K*hidden]  (context features for next draft KV)


# Runs the parallel draft backbone over a [bs, gamma] block of token ids at the
# given [bs, gamma] positions, returning [bs, gamma, hidden] block hidden states.
DraftBackboneFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]

# Runs the target over the [bs, W] verify window at [bs, W] positions.
TargetVerifyFn = Callable[[torch.Tensor, torch.Tensor], TargetVerifyOut]


@dataclass
class SpecStepOutput:
    committed_tokens: List[List[int]]  # per request: accepted prefix + bonus
    accept_lengths: torch.Tensor       # [bs] correct_len (drafts accepted, excl. bonus)
    bonus_tokens: torch.Tensor         # [bs] next anchor token
    mean_accept_len: float             # scalar: mean committed tokens this step
    verify_lens: Optional[torch.Tensor] = None  # [bs] COMPACT per-req verify length


@dataclass
class DSparkWorker:
    spec_config: DSparkConfig
    draft_model: object  # DSparkDraftModel
    run_draft_backbone: DraftBackboneFn
    run_target_verify: TargetVerifyFn
    greedy: bool = True
    temperature: float = 0.0
    cost_model: SpsCostModel = field(default_factory=SpsCostModel)
    min_verify_len: int = 1

    @torch.inference_mode()
    def step(
        self,
        anchor_tokens: torch.Tensor,     # [bs] previous bonus tokens (the block anchors)
        block_positions: torch.Tensor,   # [bs, gamma] positions of the draft block
        base_tokens: int = 0,            # committed KV tokens already in batch (planner)
    ) -> SpecStepOutput:
        """One speculative decode step over a batch of ``bs`` requests."""
        cfg = self.spec_config
        gamma = cfg.gamma

        # -- 1. propose: mask block -> parallel backbone -> serial Markov sampling --
        block_ids = build_mask_block(anchor_tokens, gamma, cfg.mask_token_id)
        block_hidden = self.run_draft_backbone(block_ids, block_positions)  # [bs, gamma, h]
        sampler = (
            make_greedy_sampler() if self.greedy else make_temperature_sampler(self.temperature)
        )
        proposal = sample_draft_block(self.draft_model, block_hidden, anchor_tokens, sampler)

        # -- 2. (optional) confidence-scheduled verify length (COMPACT mode) --
        verify_lens = None
        if cfg.verify_mode is RaggedVerifyMode.COMPACT and proposal.confidence_probs is not None:
            plan = plan_verify(
                proposal.confidence_probs,
                self.cost_model,
                base_tokens=base_tokens,
                min_verify_len=self.min_verify_len,
            )
            verify_lens = plan.verify_lens  # [bs] (ragged packing is a documented TODO)

        # -- 3. verify: one target forward over the [anchor, draft_1..gamma] window --
        verify_ids = torch.cat([anchor_tokens.view(-1, 1), proposal.draft_tokens], dim=1)  # [bs,W]
        verify_positions = torch.cat(
            [block_positions[:, :1] - 1, block_positions], dim=1
        )  # anchor sits one position before the block
        target_out = self.run_target_verify(verify_ids, verify_positions)
        target_predict = target_out.logits.argmax(dim=-1)  # [bs, W]

        # -- 4. accept: longest correct prefix (greedy) or chain speculative sampling --
        if self.greedy:
            accept = accept_greedy(proposal.draft_tokens, target_predict)
        else:
            draft_probs = torch.softmax(proposal.corrected_logits.float(), dim=-1)
            target_probs = torch.softmax(target_out.logits.float(), dim=-1)
            accept = accept_sampling(proposal.draft_tokens, draft_probs, target_probs)

        committed = gather_committed_tokens(proposal.draft_tokens, accept)
        mean_accept = float(accept.commit_len.float().mean().item())

        # -- 5. KV commit (performed by the caller / scheduler) --
        # The target verify forward already wrote target KV for the whole window;
        # positions beyond commit_len per request are rejected and their cache slots
        # are freed / overwritten next step (see docs "KV rollback"). The *draft* KV
        # for committed positions is (re)seeded from target hidden by the caller via
        # ``draft_model.seed_draft_kv(target_out.hidden[committed], positions, out_loc)``.

        return SpecStepOutput(
            committed_tokens=committed,
            accept_lengths=accept.correct_len,
            bonus_tokens=accept.bonus,
            mean_accept_len=mean_accept,
            verify_lens=verify_lens,
        )


def compute_speedup_estimate(mean_accept_len: float, draft_cost_ratio: float) -> float:
    """Rough end-to-end speedup estimate for a chain (topk=1) speculative decoder.

    ``speedup ~= mean_accept_len / (1 + draft_cost_ratio)``, where
    ``draft_cost_ratio`` is the draft forward's cost relative to one target forward.
    A sanity check against measured numbers, not a substitute for benchmarking.
    """
    return mean_accept_len / (1.0 + max(draft_cost_ratio, 0.0))
