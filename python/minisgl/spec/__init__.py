"""DSpark speculative decoding for mini-sglang.

DSpark (DeepSeek, 2026) = a *semi-autoregressive* draft (parallel DFlash backbone +
serial Markov head) + *confidence-scheduled* verification. This package implements
it on top of mini-sglang:

    config.py       DSparkConfig + HF-config parsing
    heads.py        serial Markov head + confidence head (semi-AR core)
    verify.py       longest-correct-prefix accept (greedy) + chain spec sampling
    planner.py      confidence -> survival -> throughput-optimal verify budget
    draft_model.py  DFlash backbone sharing target embed/lm_head + weight loading
    proposer.py     mask-block build + serial Markov block sampling
    worker.py       host-side draft -> verify -> accept -> commit orchestration

The pure-logic modules (heads/verify/planner/proposer) are unit-tested on CPU
(see tests/spec/). The engine wiring that drives the two GPU forwards is documented
in docs/dspark_speculative_decoding.md and is pending on-GPU validation.
"""

from __future__ import annotations

from .config import DEFAULT_DSPARK_GAMMA, DSparkConfig, RaggedVerifyMode, parse_dspark_config
from .planner import SpsCostModel, VerifyPlan, plan_verify
from .scheduler import SpecBatchScheduler, SpecReq, build_greedy_scheduler
from .verify import AcceptResult, accept_greedy, accept_sampling
from .worker import DSparkWorker, SpecStepOutput, TargetVerifyOut

__all__ = [
    "DEFAULT_DSPARK_GAMMA",
    "DSparkConfig",
    "RaggedVerifyMode",
    "parse_dspark_config",
    "SpsCostModel",
    "VerifyPlan",
    "plan_verify",
    "AcceptResult",
    "accept_greedy",
    "accept_sampling",
    "DSparkWorker",
    "SpecStepOutput",
    "TargetVerifyOut",
    "SpecBatchScheduler",
    "SpecReq",
    "build_greedy_scheduler",
]
