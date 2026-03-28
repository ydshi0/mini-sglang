"""
Speculative LLM: end-to-end speculative decoding on Mini-SGLang.

Both target (Qwen3-8B) and draft (Qwen3-1.7B) use Mini-SGLang components.
CUDA graphs are disabled; paged attention is preserved.

Key insight: Mini-SGLang's ParallelLMHead extracts only the LAST hidden
state per request during prefill (an optimization for normal generation).
For speculative verification we need logits at ALL positions, so we call
the transformer backbone + LM head weight projection separately.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from minisgl.core import Batch, Req, SamplingParams
from minisgl.llm import LLM
from minisgl.scheduler.scheduler import _make_input_tuple, _make_positions
from minisgl.utils import init_logger, is_sm90_supported, is_sm100_supported

from .draft_engine import DraftEngine, DraftOutput
from .verify import VerifyResult, verify_greedy, verify_stochastic

logger = init_logger(__name__)


def _auto_attn_backend() -> str:
    if is_sm100_supported():
        return "trtllm"
    elif is_sm90_supported():
        return "fa,fi"
    else:
        return "fi"


@dataclass
class SpecStats:
    total_draft_tokens: int = 0
    total_accepted: int = 0
    total_bonus: int = 0
    total_target_forwards: int = 0
    total_tokens_generated: int = 0
    total_time: float = 0.0

    @property
    def acceptance_rate(self) -> float:
        return self.total_accepted / max(1, self.total_draft_tokens)

    @property
    def tokens_per_forward(self) -> float:
        return self.total_tokens_generated / max(1, self.total_target_forwards)

    @property
    def tokens_per_second(self) -> float:
        return self.total_tokens_generated / max(1e-6, self.total_time)

    def __repr__(self) -> str:
        return (
            f"SpecStats("
            f"accept_rate={self.acceptance_rate:.1%}, "
            f"tok/fwd={self.tokens_per_forward:.2f}, "
            f"tok/s={self.tokens_per_second:.1f}, "
            f"generated={self.total_tokens_generated}, "
            f"target_fwds={self.total_target_forwards}, "
            f"drafted={self.total_draft_tokens}, "
            f"accepted={self.total_accepted}, "
            f"bonus={self.total_bonus})"
        )


class SpeculativeLLM(LLM):
    """
    LLM with speculative decoding.

    Inherits Mini-SGLang's LLM for the target model. Creates a separate
    DraftEngine for the draft model. CUDA graphs disabled (cuda_graph_max_bs=0).
    """

    def __init__(
        self,
        target_model_path: str,
        draft_model_path: str,
        num_spec_tokens: int = 5,
        dtype: torch.dtype = torch.bfloat16,
        use_stochastic: bool = False,
        attention_backend: str = "auto",
        **kwargs,
    ):
        kwargs["cuda_graph_max_bs"] = 0
        if attention_backend != "auto":
            kwargs["attention_backend"] = attention_backend
        super().__init__(target_model_path, dtype=dtype, **kwargs)

        self.num_spec_tokens = num_spec_tokens
        self.use_stochastic = use_stochastic

        attn_str = attention_backend if attention_backend != "auto" else _auto_attn_backend()
        self.draft_engine = DraftEngine(
            model_path=draft_model_path,
            device=self.engine.device,
            dtype=dtype,
            attn_backend_str=attn_str,
            max_seq_len=self.engine.max_seq_len,
        )

        logger.info(
            f"SpeculativeLLM ready: "
            f"target={target_model_path}, draft={draft_model_path}, "
            f"K={num_spec_tokens}, stochastic={use_stochastic}"
        )

    # ═══════════════════════════════════════════════════
    #  Public API
    # ═══════════════════════════════════════════════════

    @torch.inference_mode()
    def generate(
        self,
        prompts: List[str] | List[List[int]],
        sampling_params: List[SamplingParams] | SamplingParams,
    ) -> List[Dict]:
        if isinstance(sampling_params, SamplingParams):
            sampling_params = [sampling_params] * len(prompts)
        return [self._generate_one(p, sp) for p, sp in zip(prompts, sampling_params)]

    # ═══════════════════════════════════════════════════
    #  Core speculative decoding loop
    # ═══════════════════════════════════════════════════

    def _generate_one(self, prompt: str | List[int], sp: SamplingParams) -> Dict:
        stats = SpecStats()
        t0 = time.time()

        input_ids = self._tokenize_one(prompt)
        prompt_len = len(input_ids)
        max_new = min(sp.max_tokens, self.engine.max_seq_len - prompt_len)
        if max_new <= 0:
            return {"text": "", "token_ids": [], "spec_stats": stats}

        temperature = sp.temperature
        is_greedy = sp.is_greedy

        # ── Prefill both models ──
        target_req, prefill_logits = self._target_prefill(input_ids, sp)
        self.draft_engine.prefill(input_ids)
        stats.total_target_forwards += 1

        # ── First token from target's prefill ──
        # prefill uses normal model.forward() which returns only last logit
        first_logit = prefill_logits[-1]
        if is_greedy:
            first_token = torch.argmax(first_logit).item()
        else:
            p = torch.softmax(first_logit / max(temperature, 1e-6), dim=-1)
            first_token = torch.multinomial(p, num_samples=1).item()
        output_ids: List[int] = [first_token]

        # ── Speculative decode loop ──
        while len(output_ids) < max_new:
            if not sp.ignore_eos and output_ids[-1] == self.eos_token_id:
                break

            remaining = max_new - len(output_ids)
            K = min(self.num_spec_tokens, remaining - 1)

            if K <= 0:
                tok = self._target_decode_one(target_req, prompt_len, output_ids)
                if tok is not None:
                    output_ids.append(tok)
                    stats.total_target_forwards += 1
                break

            # ── Draft K tokens ──
            draft_out = self.draft_engine.draft_k_tokens(
                last_token=output_ids[-1], K=K, temperature=temperature,
            )
            stats.total_draft_tokens += K

            # ── Verify with target (returns K+1 logits for ALL positions) ──
            verify_logits = self._target_verify(
                target_req, prompt_len, output_ids, draft_out.draft_tokens,
            )
            stats.total_target_forwards += 1

            # ── Accept / reject ──
            if is_greedy or not self.use_stochastic:
                vr = verify_greedy(verify_logits, draft_out.draft_tokens)
            else:
                vr = verify_stochastic(
                    verify_logits, draft_out.draft_tokens,
                    draft_out.draft_probs, temperature,
                )

            output_ids.extend(vr.accepted_tokens[: vr.total_accepted])
            stats.total_accepted += vr.num_draft_accepted
            if vr.bonus_token is not None:
                stats.total_bonus += 1

            # ── Rollback ──
            new_total = prompt_len + len(output_ids)
            self._target_rollback(target_req, new_total)
            self.draft_engine.rollback(new_total)

            logger.debug(
                f"round K={K} accepted={vr.total_accepted} output={len(output_ids)}"
            )

        # ── Cleanup ──
        self.draft_engine.reset()
        self._target_free(target_req)

        stats.total_tokens_generated = len(output_ids)
        stats.total_time = time.time() - t0

        if not sp.ignore_eos and output_ids and output_ids[-1] == self.eos_token_id:
            output_ids = output_ids[:-1]

        text = self.tokenizer.decode(output_ids, skip_special_tokens=True)
        return {"text": text, "token_ids": output_ids, "spec_stats": stats}

    # ═══════════════════════════════════════════════════
    #  Batch preparation & forward
    # ═══════════════════════════════════════════════════

    def _prepare_batch(self, batch: Batch) -> None:
        """
        Prepare batch metadata (positions, out_loc, input_ids, attn_metadata).
        Reuses the exact scheduler helper functions for correctness.
        allocate_paged is called ONCE here; callers must NOT call it separately.
        """
        dev = self.engine.device
        self.engine.graph_runner.pad_batch(batch)
        self.cache_manager.allocate_paged(batch.reqs)
        batch.positions = _make_positions(batch, dev)
        input_mapping = _make_input_tuple(batch, dev)
        batch.out_loc = self.engine.page_table[input_mapping]
        self.engine.attn_backend.prepare_metadata(batch)
        batch.input_ids = self.token_pool[input_mapping]

    def _run_forward(self, batch: Batch) -> torch.Tensor:
        """
        Run model.forward() on the engine's CUDA stream.
        Returns logits from the standard LM head (last-index-only for prefill).
        """
        with torch.cuda.stream(self.engine.stream):
            self.engine.stream.wait_stream(torch.cuda.current_stream())
            with self.engine.ctx.forward_batch(batch):
                logits = self.engine.model.forward()
        torch.cuda.current_stream().wait_stream(self.engine.stream)
        return logits

    def _run_forward_all_logits(self, batch: Batch) -> torch.Tensor:
        """
        Run model forward but return logits for ALL positions, bypassing
        the LM head's last-index extraction that normally happens in prefill.

        This calls the transformer backbone directly, then applies the
        LM head weight projection manually with correct TP handling.
        """
        with torch.cuda.stream(self.engine.stream):
            self.engine.stream.wait_stream(torch.cuda.current_stream())
            with self.engine.ctx.forward_batch(batch):
                # 1. Get ALL hidden states from the transformer backbone
                hidden = self.engine.model.model.forward(batch.input_ids)

                # 2. Apply LM head weight to ALL hidden states (no last-index extraction)
                lm_head = self.engine.model.lm_head
                module = lm_head.tied_embedding or lm_head
                logits = F.linear(hidden, module.weight, lm_head.bias)

                # 3. Handle tensor parallelism (all_gather + reshape)
                if lm_head.tp_size > 1:
                    num_tokens = logits.shape[0]
                    gathered = lm_head._comm.all_gather(logits)  # (tp * N, vocab_tp)
                    gathered = gathered.view(lm_head.tp_size, num_tokens, -1)
                    gathered = gathered.permute(1, 0, 2).contiguous()  # (N, tp, vocab_tp)
                    logits = gathered.reshape(num_tokens, -1)  # (N, tp * vocab_tp)
                    logits = logits[:, : lm_head.num_embeddings]  # (N, vocab_full)

        torch.cuda.current_stream().wait_stream(self.engine.stream)
        return logits

    # ═══════════════════════════════════════════════════
    #  Target model operations
    # ═══════════════════════════════════════════════════

    def _target_prefill(self, input_ids: torch.Tensor, sp: SamplingParams) -> Tuple[Req, torch.Tensor]:
        """Prefill target model. Returns (Req, last_logit)."""
        from minisgl.scheduler.utils import PendingReq

        table_idx = self.table_manager.allocate()
        pending = PendingReq(uid=0, input_ids=input_ids, sampling_params=sp)
        match = self.cache_manager.match_req(pending)
        handle = match.cuda_handle
        cached_len = handle.cached_len
        self.cache_manager.lock(handle)

        if cached_len > 0:
            self.table_manager.token_pool[table_idx, :cached_len].copy_(
                input_ids[:cached_len].pin_memory(), non_blocking=True,
            )
            self.engine.page_table[table_idx, :cached_len].copy_(
                handle.get_matched_indices(),
            )

        req = Req(
            input_ids=input_ids,
            table_idx=table_idx,
            cached_len=cached_len,
            output_len=sp.max_tokens,
            uid=0,
            sampling_params=sp,
            cache_handle=handle,
        )

        _s = slice(cached_len, len(input_ids))
        self.table_manager.token_pool[table_idx, _s].copy_(
            input_ids[_s].pin_memory(), non_blocking=True,
        )

        # Normal prefill: model.forward() returns last logit only (1, vocab)
        batch = Batch(reqs=[req], phase="prefill")
        self._prepare_batch(batch)
        logits = self._run_forward(batch)

        req.cached_len = req.device_len
        req.device_len += 1
        return req, logits

    def _target_verify(
        self,
        req: Req,
        prompt_len: int,
        output_ids: List[int],
        draft_tokens: List[int],
    ) -> torch.Tensor:
        """
        Verify K draft tokens. Returns (K+1, vocab_size) logits for ALL positions.

        Extend region = [last_accepted, d0, …, d_{K-1}] = K+1 tokens.
        Uses _run_forward_all_logits to bypass LM head's last-index extraction.
        """
        K = len(draft_tokens)
        current_total = prompt_len + len(output_ids)

        req.cached_len = current_total - 1
        req.device_len = current_total + K
        # extend_len = K + 1

        # Write [last_accepted, d0, …, d_{K-1}] into token pool
        verify_ids = torch.tensor(
            [output_ids[-1]] + draft_tokens, dtype=torch.int32,
        )
        lo = current_total - 1
        hi = current_total + K
        self.table_manager.token_pool[req.table_idx, lo:hi].copy_(
            verify_ids.pin_memory(), non_blocking=True,
        )

        # Extend input_ids on CPU
        req.input_ids = torch.cat([
            req.input_ids[:current_total],
            torch.tensor(draft_tokens, dtype=torch.int32),
        ])

        # Prepare batch as "prefill" for correct attention metadata (multi-token),
        # but use _run_forward_all_logits to get ALL logits, not just the last one
        batch = Batch(reqs=[req], phase="prefill")
        self._prepare_batch(batch)
        logits = self._run_forward_all_logits(batch)  # (K+1, vocab_size)

        return logits

    def _target_decode_one(
        self,
        req: Req,
        prompt_len: int,
        output_ids: List[int],
    ) -> Optional[int]:
        """Single-token decode fallback."""
        total = prompt_len + len(output_ids)
        req.cached_len = total - 1
        req.device_len = total
        self.table_manager.token_pool[req.table_idx, total - 1] = output_ids[-1]

        batch = Batch(reqs=[req], phase="decode")
        self._prepare_batch(batch)
        logits = self._run_forward(batch)

        token = torch.argmax(logits[0]).item()
        req.cached_len = req.device_len
        req.device_len += 1
        return token

    def _target_rollback(self, req: Req, new_total: int) -> None:
        req.cached_len = new_total - 1
        req.device_len = new_total

    def _target_free(self, req: Req) -> None:
        self.table_manager.free(req.table_idx)
        self.cache_manager.cache_req(req, finished=True)