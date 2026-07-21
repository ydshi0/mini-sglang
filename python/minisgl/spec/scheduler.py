"""Continuous-batching scheduler for DSpark speculative decoding.

This is the serving layer requested on top of :class:`~minisgl.spec.worker.DSparkWorker`.
It manages the request lifecycle so that a *dynamic* batch of requests is decoded
together, one speculative step at a time:

    admit prompts (prefill + seed draft KV)  ->
    loop:
        one batched spec step over ALL running requests
        commit each request's variable number of accepted tokens (1 .. gamma+1)
        roll back the KV of rejected draft positions
        evict finished requests, admit waiting ones   (continuous batching)

Why speculative decoding makes batching non-trivial: every request commits a
*different* number of tokens per step, and finishes at a different step, so the
batch must be recomposed continuously and the per-request sequence length advances
by a ragged amount. That bookkeeping is what this module implements, reusing
mini-sglang's real :class:`CacheManager` / :class:`TableManager`.

STATUS / SEAMS (pending on-GPU validation; see docs/dspark_speculative_decoding.md §7):
  * ``_run_draft_backbone`` / ``_run_target_verify`` are the two GPU forwards. They
    must (a) swap the global context between the draft and target KV pools/attention
    backends (``_activate`` below) and (b) return **full-window** per-position logits
    for verify — mini-sglang's ``ParallelLMHead`` slices to the last token in prefill,
    so verify needs a raw ``hidden @ lm_head.Wᵀ`` over all W positions. Both are marked
    TODO(gpu-validate) inline.
  * The KV-window position math in ``_commit_and_rollback`` follows the DSpark
    convention (anchor at ``device_len-1`` reuses its slot; the gamma drafts take new
    slots ``[device_len, device_len+gamma)``); the exact off-by-one must be confirmed
    on GPU against a losslessness test.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import List, Optional

import torch
from minisgl.core import SamplingParams, get_global_ctx

from .config import DSparkConfig
from .worker import DSparkWorker, TargetVerifyOut


@dataclass
class SpecReq:
    """A request in flight through the speculative decoder."""

    uid: int
    table_idx: int
    input_ids: torch.Tensor          # cpu int32: prompt + all committed tokens
    prompt_len: int
    sampling_params: SamplingParams
    cache_handle: object = None      # BaseCacheHandle for the committed prefix
    device_len: int = 0              # #tokens with committed KV in cache
    anchor_token: int = 0            # token that anchors the next draft block (last bonus)
    output_ids: List[int] = field(default_factory=list)
    finished: bool = False

    @property
    def anchor_pos(self) -> int:
        """Position of the anchor token (last committed) = device_len - 1."""
        return self.device_len - 1

    def can_decode(self, max_seq_len: int) -> bool:
        return (
            not self.finished
            and len(self.output_ids) < self.sampling_params.max_tokens
            and self.device_len < max_seq_len
        )


@dataclass
class SpecBatchScheduler:
    """Continuous-batching loop around a DSpark worker.

    ``cache_manager`` / ``table_manager`` are the *target* engine's managers (the
    committed KV lives in the target pool). The draft KV pool is owned by the draft
    context and is (re)seeded from target hidden each step.
    """

    spec_config: DSparkConfig
    worker: DSparkWorker
    cache_manager: object            # minisgl.scheduler.cache.CacheManager
    table_manager: object            # minisgl.scheduler.table.TableManager
    eos_token_id: int
    max_seq_len: int
    max_running: int = 64

    running: List[SpecReq] = field(default_factory=list)
    waiting: List[SpecReq] = field(default_factory=list)
    device: torch.device = field(default_factory=lambda: torch.device("cuda"))

    # ------------------------------------------------------------------ admission
    def add_request(self, uid: int, input_ids: List[int], sp: SamplingParams) -> None:
        """Queue a request; it is admitted (prefilled) when there is capacity."""
        self.waiting.append(
            SpecReq(
                uid=uid,
                table_idx=-1,
                input_ids=torch.tensor(input_ids, dtype=torch.int32),
                prompt_len=len(input_ids),
                sampling_params=sp,
            )
        )

    def _admit_new(self) -> None:
        """Continuous batching: pull waiting requests into the running batch while
        table/KV capacity allows, prefilling each and seeding its draft KV."""
        while self.waiting and len(self.running) < self.max_running:
            if self.table_manager.available_size == 0:
                break
            req = self.waiting.pop(0)
            req.table_idx = self.table_manager.allocate()
            first_token, _last_hidden = self._prefill_and_seed(req)
            req.anchor_token = first_token
            req.output_ids.append(first_token)
            self.running.append(req)

    # ------------------------------------------------------------------ main loop
    @torch.inference_mode()
    def step(self) -> None:
        """Run one speculative step over the whole running batch."""
        self._admit_new()
        if not self.running:
            return

        batch = self.running
        bs = len(batch)
        gamma = self.spec_config.gamma

        anchor_tokens = torch.tensor(
            [r.anchor_token for r in batch], dtype=torch.int64, device=self.device
        )
        # block positions per request: [device_len, .., device_len+gamma-1]
        block_positions = torch.stack(
            [
                torch.arange(r.device_len, r.device_len + gamma, device=self.device)
                for r in batch
            ]
        )
        base_tokens = sum(r.device_len for r in batch)

        out = self.worker.step(anchor_tokens, block_positions, base_tokens=base_tokens)

        # per-request ragged commit + KV rollback + finish detection
        self._commit_and_rollback(batch, out.committed_tokens)
        self._evict_finished()

    def _commit_and_rollback(self, batch: List[SpecReq], committed: List[List[int]]) -> None:
        """Append each request's committed tokens, advance KV, free rejected slots.

        TODO(gpu-validate): the accepted drafts' KV occupies slots
        ``[old_device_len, old_device_len + correct_len)``; the rejected tail
        ``[old_device_len + correct_len, old_device_len + gamma)`` is freed back to
        the cache manager. ``device_len`` advances by ``correct_len`` (the bonus
        becomes next step's anchor and gets its KV as the anchor query next step).
        The precise slot indices come from ``page_table[table_idx, pos]``; confirm
        against a losslessness test on GPU.
        """
        page_table = self.cache_manager.page_table
        gamma = self.spec_config.gamma
        with self.cache_manager.lazy_free_region():
            for req, tokens in zip(batch, committed):
                correct_len = len(tokens) - 1  # committed = accepted drafts + bonus
                old_device_len = req.device_len

                # rejected draft KV slots to free: [old+correct_len, old+gamma)
                reject_start = old_device_len + correct_len
                reject_end = old_device_len + gamma
                if reject_end > reject_start:
                    slots = page_table[req.table_idx, reject_start:reject_end]
                    self.cache_manager._free(slots)  # noqa: SLF001 (lazy free region)

                # append committed tokens (accepted drafts + bonus)
                req.output_ids.extend(tokens)
                req.input_ids = torch.cat(
                    [req.input_ids, torch.tensor(tokens, dtype=torch.int32)]
                )
                # KV of accepted drafts is valid; bonus is next anchor (KV seeded next step)
                req.device_len = old_device_len + correct_len
                req.anchor_token = tokens[-1]

                # finish on EOS or max_tokens
                if not req.sampling_params.ignore_eos and self.eos_token_id in tokens:
                    req.finished = True
                elif len(req.output_ids) >= req.sampling_params.max_tokens:
                    req.finished = True
                elif req.device_len >= self.max_seq_len:
                    req.finished = True

    def _evict_finished(self) -> None:
        still_running: List[SpecReq] = []
        for req in self.running:
            if req.finished:
                self._free_req(req)
            else:
                still_running.append(req)
        self.running = still_running

    def _free_req(self, req: SpecReq) -> None:
        self.table_manager.free(req.table_idx)
        if req.cache_handle is not None:
            self.cache_manager.cache_req(req, finished=True)  # type: ignore[arg-type]

    @property
    def has_work(self) -> bool:
        return bool(self.running or self.waiting)

    # ------------------------------------------------------- GPU forwards (seams)
    def _prefill_and_seed(self, req: SpecReq):
        """Prefill the target over the prompt; return (first_token, last_hidden).

        TODO(gpu-validate): run the target engine's prefill over ``req.input_ids``,
        capture the final-layer hidden (CaptureHiddenMode.FULL), sample the first
        token from the last logits, set ``req.device_len = prompt_len``, and seed the
        draft KV with ``worker.draft_model.seed_draft_kv(last_hidden, positions,
        out_loc)``. Requires the target forward to expose per-token hidden states.
        """
        raise NotImplementedError(
            "prefill+seed needs the target engine prefill hook (see docstring / docs §7)"
        )

    @contextmanager
    def _activate(self, which: str):
        """Temporarily point the global context at the draft or target KV pool + attn.

        Minimal two-context seam without editing core.py: the worker/engine holds a
        ``draft_kv`` / ``draft_attn`` alongside the target's, and we swap the active
        ones around a forward. TODO(gpu-validate).
        """
        ctx = get_global_ctx()
        saved = (ctx.kv_cache, ctx.attn_backend)
        try:
            # engine is expected to attach ctx.draft_kv / ctx.draft_attn at init
            if which == "draft":
                ctx.kv_cache = getattr(ctx, "draft_kv", ctx.kv_cache)
                ctx.attn_backend = getattr(ctx, "draft_attn", ctx.attn_backend)
            yield
        finally:
            ctx.kv_cache, ctx.attn_backend = saved


def build_greedy_scheduler(
    spec_config: DSparkConfig,
    draft_model,
    run_draft_backbone,
    run_target_verify,
    *,
    cache_manager,
    table_manager,
    eos_token_id: int,
    max_seq_len: int,
    max_running: int = 64,
    device: Optional[torch.device] = None,
) -> SpecBatchScheduler:
    """Convenience factory wiring a greedy DSpark worker into the batch scheduler."""
    worker = DSparkWorker(
        spec_config=spec_config,
        draft_model=draft_model,
        run_draft_backbone=run_draft_backbone,
        run_target_verify=run_target_verify,
        greedy=True,
    )
    return SpecBatchScheduler(
        spec_config=spec_config,
        worker=worker,
        cache_manager=cache_manager,
        table_manager=table_manager,
        eos_token_id=eos_token_id,
        max_seq_len=max_seq_len,
        max_running=max_running,
        device=device or torch.device("cuda"),
    )


# TargetVerifyOut re-exported for callers implementing run_target_verify.
__all__ = ["SpecReq", "SpecBatchScheduler", "build_greedy_scheduler", "TargetVerifyOut"]
