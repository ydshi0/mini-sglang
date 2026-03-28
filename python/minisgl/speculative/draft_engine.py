"""
Draft Engine for Speculative Decoding.

A lightweight Mini-SGLang engine for the draft (small) model. It reuses
the same core components as the target engine:
    - Model (Qwen3ForCausalLM loaded via Mini-SGLang's create_model + load_weight)
    - Paged KV Cache (MHAKVCache)
    - Attention Backend (FlashInfer / FlashAttention / TRT-LLM)

But it does NOT use:
    - CUDA Graphs (unnecessary for the small model)
    - Scheduler / CacheManager / TableManager (we manage one request manually)
    - torch.distributed init (shares the process group with the target engine)

Context switching: Mini-SGLang stores a global Context that the model's
forward pass reads via get_global_ctx(). The DraftEngine swaps to its own
Context before every forward pass, then swaps back immediately.

    ┌─────────────────────────────────────────────────┐
    │                 DraftEngine                       │
    │                                                   │
    │  Model (Qwen3-1.7B)   ←── Mini-SGLang layers     │
    │  KV Cache Pool         ←── MHAKVCache             │
    │  Page Table            ←── Identity mapping       │
    │  Token Pool            ←── stores token IDs       │
    │  Attention Backend     ←── same type as target     │
    │  Context               ←── private, swapped in    │
    │                                                   │
    │  No CUDA Graph, No Scheduler, No CacheManager     │
    └─────────────────────────────────────────────────┘
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import List, Tuple

import torch
from minisgl.attention import create_attention_backend
from minisgl.core import Batch, Context, Req, SamplingParams, get_global_ctx
from minisgl.kvcache import create_kvcache_pool
from minisgl.layers import set_rope_device
from minisgl.models import ModelConfig, create_model, load_weight
from minisgl.utils import init_logger, torch_dtype

# swap_global_ctx was added to core.py for speculative decoding.
# If the user hasn't patched core.py yet, define it here as a fallback.
try:
    from minisgl.core import swap_global_ctx
except ImportError:
    import minisgl.core as _core

    def swap_global_ctx(new_ctx: Context) -> Context:
        old_ctx = _core._GLOBAL_CTX
        _core._GLOBAL_CTX = new_ctx
        return old_ctx  # type: ignore

logger = init_logger(__name__)


@dataclass
class DraftOutput:
    """Output from one draft generation round."""
    draft_tokens: List[int]       # K drafted token ids
    draft_probs: List[torch.Tensor]  # K probability distributions (on CPU)


class DraftEngine:
    """
    Lightweight Mini-SGLang engine for the draft model.

    Manages a single request at a time with a simple sequential page layout
    (position i → physical KV slot i). This avoids all the complexity of
    CacheManager / RadixCache while still using paged attention.
    """

    def __init__(
        self,
        model_path: str,
        device: torch.device,
        dtype: torch.dtype,
        attn_backend_str: str,
        max_seq_len: int,
    ):
        self.device = device
        self.dtype = dtype
        self.max_seq_len = max_seq_len

        # ── Load model config ──
        from minisgl.utils import cached_load_hf_config
        hf_config = cached_load_hf_config(model_path)
        self.model_config = ModelConfig.from_hf(hf_config)

        logger.info(
            f"Draft model: {self.model_config.num_layers} layers, "
            f"hidden={self.model_config.hidden_size}, "
            f"heads={self.model_config.num_qo_heads}/{self.model_config.num_kv_heads}"
        )

        # ── Create model on meta device, then load real weights ──
        set_rope_device(device)
        with torch.device("meta"), torch_dtype(dtype):
            self.model = create_model(self.model_config)
        state_dict = {k: v.to(dtype) for k, v in load_weight(model_path, device)}
        self.model.load_state_dict(state_dict)

        # ── KV Cache Pool ──
        page_size = 1
        num_pages = max_seq_len + 1  # +1 for dummy page
        self.kv_cache = create_kvcache_pool(
            model_config=self.model_config,
            num_pages=num_pages + 1,
            page_size=page_size,
            dtype=dtype,
            device=device,
        )

        # ── Page Table: identity mapping (position i → slot i) ──
        # Shape: (2, max_seq_len) — slot 0 = active request, slot 1 = dummy
        self.page_table = torch.zeros((2, max_seq_len), dtype=torch.int32, device=device)
        self.page_table[0] = torch.arange(max_seq_len, dtype=torch.int32, device=device)
        self.page_table[1].fill_(num_pages)  # dummy → last page

        # ── Token Pool: stores token IDs ──
        self.token_pool = torch.zeros((2, max_seq_len), dtype=torch.int32, device=device)

        # ── Private Context (NOT set as global yet) ──
        self.ctx = Context(page_size)
        self.ctx.kv_cache = self.kv_cache
        self.ctx.page_table = self.page_table

        # ── Attention Backend ──
        # CRITICAL: FA/FI backends capture `get_global_ctx().kv_cache` in __init__.
        # We must swap to draft context first so the backend binds to the DRAFT's
        # KV cache, not the target's.
        old_ctx = swap_global_ctx(self.ctx)
        self.attn_backend = create_attention_backend(attn_backend_str, self.model_config)
        swap_global_ctx(old_ctx)

        self.ctx.attn_backend = self.attn_backend

        # ── Pre-allocate a CPU buffer for Req.input_ids (avoids repeated alloc) ──
        self._input_ids_buf = torch.zeros(max_seq_len, dtype=torch.int32)

        # ── Sequence state ──
        self.cached_len = 0  # positions [0, cached_len) have valid KV
        self.device_len = 0  # total tokens written so far

        logger.info(f"DraftEngine ready: {model_path}, max_seq_len={max_seq_len}")

    # ──────────────────────────────────────────────────
    # Context switching
    # ──────────────────────────────────────────────────

    @contextmanager
    def _use_draft_ctx(self):
        """Temporarily swap the global context to the draft engine's context."""
        old_ctx = swap_global_ctx(self.ctx)
        try:
            yield
        finally:
            swap_global_ctx(old_ctx)

    # ──────────────────────────────────────────────────
    # Forward pass helpers
    # ──────────────────────────────────────────────────

    def _make_batch(self, cached_len: int, device_len: int, phase: str) -> Batch:
        """Build a Batch for the given extend region [cached_len, device_len)."""
        extend_len = device_len - cached_len

        # Create a Req (content of input_ids doesn't matter, only its length)
        req = Req(
            input_ids=self._input_ids_buf[:device_len],
            table_idx=0,
            cached_len=cached_len,
            output_len=self.max_seq_len - device_len,
            uid=0,
            sampling_params=SamplingParams(),
            cache_handle=None,  # type: ignore  # no CacheManager
        )

        batch = Batch(reqs=[req], phase=phase)
        batch.padded_reqs = batch.reqs  # no CUDA graph padding

        # Positions
        batch.positions = torch.arange(
            cached_len, device_len, dtype=torch.int32, device=self.device
        )

        # Index mapping: (table_idx=0, offset) for each token in extend region
        zeros = torch.zeros(extend_len, dtype=torch.int64, device=self.device)
        offsets = torch.arange(cached_len, device_len, dtype=torch.int64, device=self.device)
        mapping = (zeros, offsets)

        batch.out_loc = self.page_table[mapping]
        batch.input_ids = self.token_pool[mapping]

        return batch

    def _forward(self, batch: Batch) -> torch.Tensor:
        """Run model forward with context swap. Returns logits (extend_len, vocab)."""
        with self._use_draft_ctx():
            self.attn_backend.prepare_metadata(batch)
            with self.ctx.forward_batch(batch):
                logits = self.model.forward()
        return logits

    # ──────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────

    def prefill(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Prefill the draft model with prompt tokens.

        Args:
            input_ids: 1D CPU tensor of prompt token ids.

        Returns:
            logits: (prompt_len, vocab_size) logits for all prompt positions.
        """
        prompt_len = len(input_ids)
        assert prompt_len < self.max_seq_len

        # Write prompt tokens to token pool
        self.token_pool[0, :prompt_len].copy_(
            input_ids.to(self.device), non_blocking=True
        )

        # Build and run batch
        batch = self._make_batch(cached_len=0, device_len=prompt_len, phase="prefill")
        logits = self._forward(batch)

        # Update state
        self.cached_len = prompt_len
        self.device_len = prompt_len

        return logits

    def decode_one(self, token_id: int) -> torch.Tensor:
        """
        Decode one token. Writes token to the token pool, runs forward,
        updates KV cache, and returns logit for the next position.

        Args:
            token_id: Token to process at the current position.

        Returns:
            logits: (1, vocab_size) logit predicting the next token.
        """
        pos = self.device_len
        assert pos < self.max_seq_len

        # Write token
        self.token_pool[0, pos] = token_id
        self.device_len = pos + 1

        # Forward: extend from [cached_len, device_len)
        batch = self._make_batch(
            cached_len=self.cached_len,
            device_len=self.device_len,
            phase="decode",
        )
        logits = self._forward(batch)

        # Advance
        self.cached_len = self.device_len

        return logits

    def draft_k_tokens(
        self,
        last_token: int,
        K: int,
        temperature: float = 0.0,
    ) -> DraftOutput:
        """
        Autoregressively draft K tokens.

        Args:
            last_token: The last accepted token (starting point for drafting).
            K: Number of tokens to draft.
            temperature: Sampling temperature (0.0 = greedy).

        Returns:
            DraftOutput with K tokens and their probability distributions.
        """
        draft_tokens: List[int] = []
        draft_probs: List[torch.Tensor] = []
        current_token = last_token

        for _ in range(K):
            logits = self.decode_one(current_token)  # (1, vocab_size)
            logit_1d = logits[0]  # (vocab_size,)

            if temperature <= 0.0:
                probs = torch.softmax(logit_1d, dim=-1)
                next_token = torch.argmax(logit_1d).item()
            else:
                probs = torch.softmax(logit_1d / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).item()

            draft_tokens.append(next_token)
            draft_probs.append(probs.cpu())
            current_token = next_token

        return DraftOutput(draft_tokens=draft_tokens, draft_probs=draft_probs)

    def rollback(self, keep_len: int) -> None:
        """
        Roll back state so that only positions [0, keep_len) are valid.

        The KV cache entries beyond keep_len are stale but harmless —
        the attention kernel only attends to [0, cached_len).

        We set cached_len = keep_len - 1 because the last position may
        hold a corrected/bonus token from the target model that differs
        from what the draft computed. The next decode_one() call will
        reprocess that position with the correct token, updating the KV.

        Args:
            keep_len: Number of valid tokens (prompt + accepted output).
        """
        self.cached_len = keep_len - 1
        self.device_len = keep_len - 1
        # On the next decode_one(correct_token), it will:
        #   - write correct_token at position keep_len - 1
        #   - device_len becomes keep_len
        #   - forward processes [cached_len=keep_len-1, device_len=keep_len)
        #   - KV at position keep_len-1 is recomputed with the correct token ✓

    def reset(self) -> None:
        """Reset all state for a new request."""
        self.cached_len = 0
        self.device_len = 0