"""DSpark draft model for mini-sglang = DFlash parallel backbone + serial Markov head
+ confidence head, sharing (and freezing) the target's ``embed_tokens`` / ``lm_head``.

Faithful port of ``sglang/srt/models/dflash.py::DFlashDraftModel`` +
``sglang/srt/models/dspark.py::DSparkDraftMixin``, re-expressed with mini-sglang's
``BaseOP`` layer primitives so it plugs into the same weight loader and forward
machinery as the target model.

The checkpoint (e.g. ``deepseek-ai/dspark_qwen3_8b_block7``) ships ONLY:
    layers.*            transformer backbone (Qwen3-style: qk-norm, RoPE, SiLU MLP)
    fc.weight           Linear(K*hidden -> hidden), projects target context features
    hidden_norm.weight  RMSNorm after fc
    norm.weight         final RMSNorm
    markov_head.*        the serial semi-AR head (see spec/heads.py)
    confidence_head.*    per-position acceptance-prob head (see spec/heads.py)
It does NOT ship embed_tokens / lm_head / rotary caches — those are taken from the
target and skipped on load (``_SKIP_PREFIXES``).

INTEGRATION NOTE (validation-pending, see docs/dspark_speculative_decoding.md):
the backbone attention (`Qwen3DecoderLayer`) reads the *global* forward context
(`get_global_ctx().batch`) for positions / KV pool / attention metadata, exactly
like the target. The worker is therefore responsible for installing a *draft*
context (separate KV pool + attention backend) around ``forward_backbone`` and for
seeding the draft KV from target hidden via ``seed_draft_kv`` before decoding.
Those engine seams need on-GPU validation; the module structure, math and weight
mapping below are complete.
"""

from __future__ import annotations

from typing import Dict, Iterator, Optional, Tuple

import torch
import torch.nn.functional as F
from minisgl.core import get_global_ctx
from minisgl.layers import LinearReplicated, OPList, RMSNormFused
from minisgl.layers.base import BaseOP
from minisgl.models.config import ModelConfig
from minisgl.models.qwen3 import Qwen3DecoderLayer

from .config import DSparkConfig
from .heads import build_confidence_head, build_markov_head


class DSparkDraftModel(BaseOP):
    """Embed-less semi-autoregressive draft model."""

    def __init__(self, model_config: ModelConfig, spec_config: DSparkConfig) -> None:
        super().__init__()
        self.model_config = model_config
        self.spec_config = spec_config
        hidden = model_config.hidden_size

        # --- DFlash parallel backbone (Qwen3-style layers, no embed/lm_head) ---
        self.layers = OPList(
            [Qwen3DecoderLayer(model_config, i) for i in range(model_config.num_layers)]
        )
        self.norm = RMSNormFused(size=hidden, eps=model_config.rms_norm_eps)

        # --- target-context projection: concat(K * hidden) -> hidden ---
        num_context_features = len(spec_config.target_layer_ids)
        self.fc = LinearReplicated(num_context_features * hidden, hidden, has_bias=False)
        self.hidden_norm = RMSNormFused(size=hidden, eps=model_config.rms_norm_eps)

        # --- DSpark heads (serial Markov + confidence) ---
        self.markov_head = build_markov_head(spec_config)
        self.confidence_head = build_confidence_head(spec_config)

        # --- shared modules from the target (underscored => excluded from state_dict) ---
        self._lm_head = None  # ParallelLMHead of the target
        self._embed_tokens = None  # VocabParallelEmbedding of the target

    # ------------------------------------------------------------------ sharing
    def attach_shared_modules(self, *, embed_tokens, lm_head) -> None:
        """Wire in the target's (frozen) embedding and lm_head. Called once at init."""
        self._embed_tokens = embed_tokens
        self._lm_head = lm_head

    def embed(self, input_ids: torch.Tensor) -> torch.Tensor:
        assert self._embed_tokens is not None, "call attach_shared_modules first"
        return self._embed_tokens.forward(input_ids)

    # -------------------------------------------------------------- projections
    def project_target_hidden(self, target_hidden: torch.Tensor) -> torch.Tensor:
        """hidden_norm(fc(concat target-layer features)) -> draft context hidden.

        ``target_hidden`` is ``[N, K*hidden]`` (K = num_context_features). Mirrors
        ``DFlashDraftModel.project_target_hidden``.
        """
        return self.hidden_norm.forward(self.fc.forward(target_hidden), None)[0]

    def forward_backbone(self, input_embeds: torch.Tensor) -> torch.Tensor:
        """Run the parallel backbone over token embeddings, return final hidden.

        Relies on the worker having installed the *draft* forward context
        (positions / KV pool / attn metadata) into the global context, identical
        to how the target's ``Qwen3Model.forward`` consumes it.
        """
        x = input_embeds
        residual: Optional[torch.Tensor] = None
        for layer in self.layers.op_list:
            x, residual = layer.forward(x, residual)
        return self.norm.forward(x, residual)[0]

    def compute_base_logits(self, hidden: torch.Tensor) -> torch.Tensor:
        """base logits = hidden @ target_lm_head.weight.T (shared, cropped to vocab).

        Mirrors ``DSparkDraftMixin.compute_base_logits`` (the draft has no lm_head
        of its own; it reuses the target's).
        """
        assert self._lm_head is not None, "call attach_shared_modules first"
        module = self._lm_head.tied_embedding or self._lm_head
        logits = F.linear(hidden, module.weight)
        return logits[..., : self.spec_config.vocab_size]

    def confidence_probs(
        self, hidden: torch.Tensor, markov_embed_stack: Optional[torch.Tensor]
    ) -> Optional[torch.Tensor]:
        """Per-position acceptance probability (STS-calibrated), or None if disabled."""
        if self.confidence_head is None:
            return None
        raw = self.confidence_head.forward(hidden, markov_embed_stack)
        return self.confidence_head.apply_sts(raw)

    # ---------------------------------------------------------------- KV inject
    def seed_draft_kv(
        self,
        target_hidden: torch.Tensor,
        positions: torch.Tensor,
        out_loc: torch.Tensor,
    ) -> None:
        """Write the committed prefix's draft KV from *target* hidden states.

        Faithful to ``DSparkDraftMixin.write_target_hidden_kv``: for each draft
        layer, project target hidden -> K/V (skip Q) -> k-norm -> k-rope -> store
        into the draft KV pool at ``out_loc``. The draft never runs a full forward
        over the prefix; its cache is *derived* from the target's hidden, which is
        what keeps the drafter both cheap and aligned with the target.

        INTEGRATION NOTE: uses the draft KV pool held in the global context; the
        worker must install the draft context before calling. Needs GPU validation.
        """
        ctx = get_global_ctx()
        pool = ctx.kv_cache
        ctx_hidden = self.project_target_hidden(target_hidden)
        for layer_id, layer in enumerate(self.layers.op_list):
            attn = layer.self_attn  # RopeAttn
            k, v = _kv_proj_only(attn, ctx_hidden)
            k = _apply_k_norm(attn, k)
            k = _apply_k_rope(attn, positions, k)
            pool.store_kv(k, v, out_loc, layer_id)

    # ------------------------------------------------------------ weight loading
    def load_weights(self, weights: Iterator[Tuple[str, torch.Tensor]]) -> None:
        """Consume a stream of (name, tensor) from the draft checkpoint.

        We reuse mini-sglang's :func:`minisgl.models.weight.load_weight`, which
        already fuses ``q/k/v_proj`` -> ``qkv_proj`` and ``gate/up_proj`` ->
        ``gate_up_proj`` for the backbone, then remap the DSpark-specific keys and
        drop the shared modules. See ``remap_draft_key``.
        """
        state: Dict[str, torch.Tensor] = {}
        for name, tensor in weights:
            mapped = remap_draft_key(name)
            if mapped is None:
                continue  # shared/rotary/unused
            state[mapped] = tensor
        self.load_state_dict(state)


# Keys the draft checkpoint ships but that live on the (shared, frozen) target.
_SKIP_PREFIXES = ("embed_tokens.", "lm_head.", "rotary_emb.", "model.embed_tokens.")


def remap_draft_key(name: str) -> Optional[str]:
    """Map a DSpark draft checkpoint key to a mini-sglang ``DSparkDraftModel`` key.

    Returns None for keys that should be skipped (shared/rotary).

    NOTE: exact source key spellings (esp. ``markov_head.*`` / ``confidence_head.*``)
    must be reconciled against the real checkpoint — see the runbook's
    ``inspect_checkpoint`` step. The mapping below matches the sglang reference
    module attribute names.
    """
    name = name.removeprefix("model.")
    if any(name.startswith(p) for p in _SKIP_PREFIXES):
        return None
    # backbone layers.* / fc.weight / hidden_norm.weight / norm.weight map 1:1
    if name.startswith(("layers.", "fc.", "hidden_norm.", "norm.")):
        return name
    # Markov head: nn.Embedding/nn.Linear .weight -> bare tensors w1/w2
    if name.startswith("markov_head."):
        return (
            name.replace("markov_head.markov_w1.weight", "markov_head.w1")
            .replace("markov_head.markov_w2.weight", "markov_head.w2")
            .replace("markov_head.gate_proj.weight", "markov_head.gate_proj_w")
            .replace("markov_head.gate_proj.bias", "markov_head.gate_proj_b")
        )
    # Confidence head: proj.{weight,bias} -> proj_w/proj_b
    if name.startswith("confidence_head."):
        return name.replace("confidence_head.proj.weight", "confidence_head.proj_w").replace(
            "confidence_head.proj.bias", "confidence_head.proj_b"
        )
    return name


# --------------------------------------------------------------------------- #
# KV-projection helpers (slice the fused qkv weight of a mini-sglang RopeAttn). #
# Mirror DFlashAttention.kv_proj_only / apply_k_norm / apply_k_rope.           #
# --------------------------------------------------------------------------- #
def _kv_proj_only(attn, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    qkv = attn.qkv_proj  # LinearQKVMerged, fused [q|k|v] on dim 0
    q_size = attn.attn.qo_attn_dim
    kv_size = attn.attn.kv_attn_dim
    weight = qkv.weight[q_size : q_size + 2 * kv_size]
    bias = qkv.bias[q_size : q_size + 2 * kv_size] if qkv.bias is not None else None
    kv = F.linear(hidden_states, weight, bias)
    k, v = kv.split([kv_size, kv_size], dim=-1)
    return k, v


def _apply_k_norm(attn, k: torch.Tensor) -> torch.Tensor:
    if attn.k_norm is None:
        return k
    head_dim = attn.attn.head_dim
    k_view = k.view(-1, attn.attn.num_kv_heads, head_dim)
    attn.k_norm.forward_inplace(k_view)
    return k


def _apply_k_rope(attn, positions: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    dummy_q = torch.empty_like(k)
    _, k = attn.attn.rotary.forward(positions, dummy_q, k)
    return k
