from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, List

# Default block size (gamma) used by DeepSeek's released dspark_qwen3_*_block7 checkpoints.
DEFAULT_DSPARK_GAMMA = 7


class RaggedVerifyMode(str, Enum):
    """How many draft tokens each request verifies per step.

    * STATIC   : every request always verifies the full ``gamma + 1`` window.
                 The simplest, always-correct baseline. This is our default.
    * COMPACT  : per-request variable verify length chosen by the confidence
                 planner (DSpark's confidence-scheduled verification), packed
                 into a ragged layout to save target FLOPs under load.

    Mirrors ``sglang/srt/speculative/ragged_verify.py::RaggedVerifyMode``.
    """

    STATIC = "static"
    COMPACT = "compact"


@dataclass(frozen=True)
class DSparkConfig:
    """Parsed DSpark draft configuration.

    The fields mirror ``DSparkDraftConfig`` in
    ``sglang/srt/speculative/dspark_components/dspark_config.py``.

    A DSpark drafter is a *semi-autoregressive* model:
      - a parallel DFlash backbone proposes the hidden states of a whole
        ``gamma``-wide block in a single forward pass, and
      - a lightweight serial Markov head walks the block token-by-token to
        inject intra-block dependency (this is what mitigates the acceptance
        decay of purely-parallel drafters).
      - an optional confidence head predicts a per-position acceptance
        probability, consumed by the verify planner.

    The draft *shares and freezes* the target's ``embed_tokens`` and ``lm_head``;
    the checkpoint therefore only ships the backbone + the two heads.
    """

    # ---- backbone ----
    num_hidden_layers: int
    hidden_size: int
    vocab_size: int
    # which target layers feed the draft (context projection). Usually a single
    # late layer; kept for parity with the reference config.
    target_layer_ids: List[int] = field(default_factory=lambda: [-1])

    # ---- block / verify geometry ----
    gamma: int = DEFAULT_DSPARK_GAMMA
    # the "mask"/noise token id filling draft-block positions 1..gamma before
    # the parallel backbone fills them in. Slot 0 is the previous bonus token.
    mask_token_id: int = 0

    # ---- Markov head (the serial, semi-AR component) ----
    markov_rank: int = 0
    markov_head_type: str = "vanilla"  # {"vanilla", "gated", "rnn"}

    # ---- confidence head (drives confidence-scheduled verification) ----
    enable_confidence_head: bool = True
    confidence_head_with_markov: bool = True

    # ---- runtime ----
    verify_mode: RaggedVerifyMode = RaggedVerifyMode.STATIC

    def __post_init__(self) -> None:
        if self.markov_rank <= 0:
            raise ValueError(
                "DSpark requires markov_rank > 0 (the Markov head is the core of "
                f"the semi-AR draft); got markov_rank={self.markov_rank}."
            )
        if self.markov_head_type not in ("vanilla", "gated", "rnn"):
            raise ValueError(f"Unsupported markov_head_type={self.markov_head_type!r}.")

    @property
    def verify_window(self) -> int:
        """Number of tokens verified per request per step: anchor + gamma drafts."""
        return self.gamma + 1

    @property
    def num_draft_tokens(self) -> int:
        """Alias used by the CLI (``--speculative-num-draft-tokens`` == gamma + 1)."""
        return self.gamma + 1

    def require_confidence(self) -> bool:
        return self.enable_confidence_head and self.verify_mode is not RaggedVerifyMode.STATIC


def _get(hf: Any, *names: str, default: Any = None) -> Any:
    """Read the first present attribute among ``names`` from an HF config-like object.

    DSpark checkpoints expose their keys either directly (``markov_rank``) or
    behind a ``dspark_`` prefix on the *target* config (``dspark_markov_rank``),
    or nested under a ``dspark_config`` sub-object. We look in all of them.
    """
    nested = getattr(hf, "dspark_config", None)
    text = getattr(hf, "text_config", None)
    for src in (hf, nested, text):
        if src is None:
            continue
        for name in names:
            if hasattr(src, name) and getattr(src, name) is not None:
                return getattr(src, name)
            if isinstance(src, dict) and src.get(name) is not None:
                return src[name]
    return default


def parse_dspark_config(draft_hf_config: Any, verify_mode: RaggedVerifyMode) -> DSparkConfig:
    """Build a :class:`DSparkConfig` from a draft checkpoint's HF config.

    Mirrors ``parse_dspark_draft_config`` in the sglang reference.
    """
    gamma = int(_get(draft_hf_config, "dspark_block_size", "block_size", "gamma",
                     default=DEFAULT_DSPARK_GAMMA))
    markov_rank = int(_get(draft_hf_config, "dspark_markov_rank", "markov_rank", default=0))
    markov_head_type = str(
        _get(draft_hf_config, "dspark_markov_head_type", "markov_head_type", default="vanilla")
    ).lower()
    mask_token_id = int(
        _get(draft_hf_config, "dspark_noise_token_id", "mask_token_id", default=0)
    )
    hidden_size = int(_get(draft_hf_config, "hidden_size"))
    vocab_size = int(_get(draft_hf_config, "vocab_size"))
    num_hidden_layers = int(_get(draft_hf_config, "num_hidden_layers", default=1))
    target_layer_ids = _get(draft_hf_config, "dspark_target_layer_ids", "target_layer_ids",
                            default=[-1])
    if isinstance(target_layer_ids, int):
        target_layer_ids = [target_layer_ids]
    enable_conf = bool(_get(draft_hf_config, "enable_confidence_head", default=True))
    conf_with_markov = bool(
        _get(draft_hf_config, "confidence_head_with_markov", default=markov_rank > 0)
    )

    return DSparkConfig(
        num_hidden_layers=num_hidden_layers,
        hidden_size=hidden_size,
        vocab_size=vocab_size,
        target_layer_ids=list(target_layer_ids),
        gamma=gamma,
        mask_token_id=mask_token_id,
        markov_rank=markov_rank,
        markov_head_type=markov_head_type,
        enable_confidence_head=enable_conf,
        confidence_head_with_markov=conf_with_markov,
        verify_mode=verify_mode,
    )
