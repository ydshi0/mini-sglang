# DSpark Speculative Decoding on Mini-SGLang

An implementation of **DSpark** (DeepSeek, June 2026 — *"DSpark: Confidence-Scheduled
Speculative Decoding with Semi-Autoregressive Generation"*) inside mini-sglang.

DSpark accelerates LLM inference **losslessly** (byte-identical output) by pairing a
cheap *semi-autoregressive* drafter with the target model, and by spending the
target's verification budget where it pays off. Reported production numbers: **+57-85%
per-user generation speed** on DeepSeek-V4, and released Qwen3 drafter checkpoints
(`deepseek-ai/dspark_qwen3_{4,8,14}b_block7`).

This document is the design record for the port. Code lives in
[`python/minisgl/spec/`](../python/minisgl/spec/).

> **Status.** The DSpark *algorithm* (semi-AR drafting, longest-prefix / chain
> acceptance, confidence-scheduled verification) is implemented and CPU-unit-tested
> ([`tests/spec/test_dspark_logic.py`](../tests/spec/test_dspark_logic.py)). The
> *engine wiring* that drives the two GPU forwards (separate draft KV pool, verify
> attention with `extend_len > 1`, KV commit) is implemented against a clear
> interface but is **pending on-GPU validation** — the machine this was written on
> has no CUDA/torch env. See [section 7](#7-engine-integration-validation-pending) and
> the [runbook](./dspark_runbook.md).

---

## 1. Why DSpark (vs. a plain two-model drafter)

A classic drafter proposes tokens **autoregressively** - as accurate as it is slow,
because it is itself serial. A *parallel* drafter (EAGLE-3 / DFlash) emits a whole
block in one forward, but its positions are conditionally independent, so acceptance
**decays sharply with depth** (measured per-position accept ~ 60/34/19/10/5/3/1%).

DSpark keeps the parallel backbone's speed but bolts on **one lightweight serial
head** so each block position can glance at the previous token - just enough
autoregressivity to kill suffix decay. Then, instead of always verifying the whole
block, it uses a **confidence head + a load-aware scheduler** to verify only as deep
as is throughput-optimal. Two ideas, stacked:

| Component | What it fixes | Where |
|---|---|---|
| Semi-autoregressive draft (parallel backbone + serial Markov head) | acceptance decay of parallel drafters | [`heads.py`](../python/minisgl/spec/heads.py), [`draft_model.py`](../python/minisgl/spec/draft_model.py) |
| Confidence-scheduled verification | wasted target FLOPs on doomed tail tokens under load | [`planner.py`](../python/minisgl/spec/planner.py) |

---

## 2. Vocabulary & block geometry

```
gamma (block size)  : number of draft tokens proposed per step (checkpoint = 7)
W = gamma + 1        : verify window per request = anchor + gamma drafts
anchor               : the previous step's *bonus* token (already accepted)
bonus                : the free token the target contributes each step
```

Per step, every request commits **>=1 and <= gamma+1** tokens (worst case: only the
bonus; best case: all gamma drafts + bonus). Mean committed tokens per step is the
**acceptance length** - the single number that drives speedup.

---

## 3. The draft model - [`draft_model.py`](../python/minisgl/spec/draft_model.py)

`DSparkDraftModel` = **DFlash backbone** + **Markov head** + **confidence head**,
sharing the target's `embed_tokens` and `lm_head`. Faithful port of
`sglang/srt/models/dflash.py::DFlashDraftModel` + `dspark.py::DSparkDraftMixin`,
rewritten with mini-sglang's `BaseOP` primitives (reuses `Qwen3DecoderLayer`).

The checkpoint ships **only** `layers.*`, `fc.weight`, `hidden_norm.weight`,
`norm.weight`, `markov_head.*`, `confidence_head.*` - no `embed_tokens` / `lm_head`
(those come from the target and are skipped on load, `remap_draft_key`).

- **Context projection.** `project_target_hidden(h) = hidden_norm(fc(h))` maps the
  target's last-layer hidden into draft space. `fc` is `Linear(K*hidden -> hidden)`,
  `K = len(target_layer_ids)` (1 for the dense Qwen3 drafter).
- **Base logits.** The draft has no lm_head: `compute_base_logits(h) = h @ W^T` using
  the **target's** `lm_head.weight`, cropped to vocab (`DSparkDraftMixin.compute_base_logits`).
- **KV seeding.** `seed_draft_kv` derives the draft's prefix KV from *target* hidden
  states - for each draft layer it does `kv_proj_only(ctx_hidden)` -> k-norm -> k-rope
  -> store. The draft never runs a full forward over the committed prefix; its cache is
  *derived* from the target, which is what keeps it cheap **and** aligned. Faithful to
  `DSparkDraftMixin.write_target_hidden_kv`.

---

## 4. The semi-autoregressive heads - [`heads.py`](../python/minisgl/spec/heads.py)

**Markov head (the "semi-AR" bit).** The backbone produces `base_logits[bs, gamma, vocab]`
in one parallel pass. The Markov head then walks the block **serially**
(`run_markov_block`), adding a previous-token-dependent bias at each step and feeding
the sampled token into the next:

```
bias(prev_token) = W2 @ (W1[prev_token])          # VanillaMarkov
step_logits[k]   = base_logits[k] + bias(token[k-1])
token[k]         = sample(step_logits[k])          # conditions step k+1
```

That single serial dependency recovers most of a full-AR drafter's accuracy at a
fraction of the cost (only a tiny `W1/W2` are serial; the transformer stays parallel).
Vanilla / gated variants are implemented; RNN is left as a documented extension.

**Confidence head.** `proj(cat[hidden, markov_embed]) -> 1`, then a
temperature-scaled sigmoid (`apply_sts`) -> a **per-position acceptance probability**.
This is the signal the verify planner consumes.

---

## 5. Verification & acceptance - [`verify.py`](../python/minisgl/spec/verify.py)

One target forward over the `[anchor, draft_1..draft_gamma]` window yields
`target_predict[:, k]` = the token the target itself would emit after window prefix
`k`. DSpark uses a **linear chain** (top-k = 1), not a branching tree, so the mask is
just a causal extend - no bitmask.

- **Greedy** (`accept_greedy`): the *longest correct prefix* rule, exactly the
  reference's `cumprod`-of-matches:
  ```
  matches     = (draft == target_predict[:, :gamma])
  correct_len = matches.int().cumprod(1).sum(1)     # leading run length
  bonus       = target_predict[arange, correct_len]  # first correction / trailing token
  ```
- **Sampling** (`accept_sampling`): lossless chain speculative sampling (accept prob
  `min(1, p/q)`; residual `normalize(relu(p-q))` on first rejection). This is what
  keeps output distribution identical to plain target sampling.

Both return `(correct_len, bonus, commit_len = correct_len + 1)`.

---

## 6. Confidence-scheduled verification - [`planner.py`](../python/minisgl/spec/planner.py)

DSpark's serving-side contribution, faithful to
`dspark_planner.py::{compute_confidence, compute_verify_token_budget, schedule_verify_lens_topk}`:

1. **Survival** = `cumprod(confidence)` per position - prob a block is still alive and
   accepted at depth `k` (monotonically non-increasing).
2. **Budget** = the number of extra verify tokens that maximizes throughput
   `theta(B) = tau*(B) * sps(tokens(B))`, where `tau*(B) = num_requests + cumsum(sorted survival)[B]`
   is expected accepted tokens and `sps(.)` is a profiled steps-per-second cost model
   ([`SpsCostModel`](../python/minisgl/spec/planner.py)). A flat cost table degenerates
   to "verify everything" - i.e. the STATIC baseline.
3. **Per-request lengths** = a *global* top-k over all `[bs, gamma]` survival entries.
   Because survival is monotone per row, the top-k automatically forms **contiguous
   prefixes**, so each request's verify length is just its count, clamped to
   `[min_verify_len, gamma]`.

Under low concurrency the optimum is "verify deep" (latency-bound -> speculation pays);
under high concurrency it shrinks the window to protect throughput. That adaptive
trade-off is the headline result to benchmark ([runbook section 5](./dspark_runbook.md)).

Two modes ([`RaggedVerifyMode`](../python/minisgl/spec/config.py)): `STATIC` (verify
all, default, simplest-correct) and `COMPACT` (planner-driven variable length).

---

## 7. Engine integration (validation-pending)

The host-side loop is [`worker.py::DSparkWorker.step`](../python/minisgl/spec/worker.py)
(one batched step) and [`scheduler.py::SpecBatchScheduler`](../python/minisgl/spec/scheduler.py)
(the **continuous-batching** loop around it: admit / prefill+seed / step / ragged commit
/ KV rollback / evict / re-admit). The batching *bookkeeping* — variable per-request
commit length, KV rollback of rejected drafts, request lifecycle — is implemented
against the real `CacheManager`/`TableManager`. The two GPU forwards remain injected
callables (`run_draft_backbone`, `run_target_verify`). Wiring these into mini-sglang
requires the following, none of which change the algorithm above:

1. **Two model contexts.** mini-sglang uses a single global `Context`
   ([`core.py`](../python/minisgl/core.py)) with one KV pool + attention backend.
   Spec decode needs a **draft KV pool** (draft `num_layers` != target) and a draft
   attention backend. Plan: make the worker swap `get_global_ctx().kv_cache` /
   `attn_backend` / `batch` between a *target* context and a *draft* context around
   each forward (or generalize `Context` to hold both). The draft shares the target's
   `embed_tokens`/`lm_head` via `attach_shared_modules`.
2. **Verify forward with `extend_len = W`.** Today the decode path assumes
   `extend_len == 1` per request ([`scheduler.py`](../python/minisgl/scheduler/scheduler.py),
   `_make_positions`). Verification runs `W = gamma+1` tokens per request - this is an
   *extend*, which the FlashInfer **prefill** wrapper already supports
   ([`attention/fi.py`](../python/minisgl/attention/fi.py) handles `max_seqlen_q > 1`).
   Route verify batches through the prefill/extend wrapper with per-request
   `qo_indptr = W`.
3. **KV rollback.** Allocate `W` cache slots per request for the verify window
   ([`cache.py::CacheManager.allocate_paged`](../python/minisgl/scheduler/cache.py)),
   run verify, then **free the rejected tail** `[commit_len, W)` back to `free_slots`
   and advance `device_len` by `commit_len` only. Because `page_size = 1`, freeing is
   just returning slot ids - this matches the "KV Cache rollback, no explicit release"
   pattern. `lazy_free_region` already batches frees.
4. **Draft KV re-seed.** After accept, call `draft_model.seed_draft_kv` with the
   target hidden of the committed positions to refresh the draft cache for the next
   block.

CUDA-graph capture of the fixed-`W` verify batch and overlap scheduling are
performance layers to add after correctness (mirrors sglang's `DsparkVerifyEpilogue`).

### Data flow (one decode step)

```
anchor(bonus)  --> build_mask_block --> [draft] backbone forward (parallel, 1 pass)
                                          |
                                          v
                                   serial Markov sampling --> draft_tokens[bs,gamma]
                                          |                    confidence[bs,gamma]
   verify window [anchor,drafts] <--------+                        |
        |                                                          v (COMPACT)
        v                                                    plan_verify -> verify_lens
   [target] verify forward (1 pass, extend W) --> target_predict[bs,W], target_hidden
        |
        v
   accept_greedy / accept_sampling --> correct_len, bonus
        |
        +--> commit accepted prefix + bonus to each request
        +--> free rejected KV slots (rollback)
        +--> seed_draft_kv(target_hidden[committed]) --> next step's anchor = bonus
```

---

## 8. Mapping to the sglang reference

| mini-sglang (this port) | sglang reference |
|---|---|
| `spec/config.py::parse_dspark_config` | `dspark_components/dspark_config.py::parse_dspark_draft_config` |
| `spec/heads.py::VanillaMarkov, run_markov_block` | `models/dspark.py::VanillaMarkov, run_markov_block` |
| `spec/heads.py::DSparkConfidenceHead` | `models/dspark.py::DSparkConfidenceHead` |
| `spec/draft_model.py::DSparkDraftModel` | `models/dspark.py::Qwen3DSparkModel` (+ `dflash.py::DFlashDraftModel`) |
| `spec/draft_model.py::seed_draft_kv` | `models/dspark.py::DSparkDraftMixin.write_target_hidden_kv` |
| `spec/proposer.py::sample_draft_block` | `dspark_components/dspark_draft.py::DraftBlockProposer` |
| `spec/verify.py::accept_greedy` | `kernels/dspark_accept.py::AcceptGreedy` (`compute_dflash_correct_drafts_and_bonus`) |
| `spec/verify.py::accept_sampling` | `kernels/dspark_accept.py::AcceptSampling` |
| `spec/planner.py` | `dspark_components/dspark_planner.py::DSparkVerifyPlanner` |
| `spec/worker.py::DSparkWorker.step` | `dspark_components/dspark_worker_v2.py::DSparkWorkerV2._forward_decode` |

**Deliberately omitted** (production infra with little value on a single-GPU demo):
CUDA-graph tier alignment, cross-DP verify-tier reduction, confidence-relay ring lag,
SPS/STS offline profiling harness, observability/metrics. All are noted at their call
sites.

---

## 9. What to measure

- **Acceptance length** (mean committed tokens/step) - target ~ 3.4 for the fully
  trained `block7` checkpoint (~2.3 for the partially-trained community one).
- **Single-stream speedup** - reference ~ 1.76x (Qwen3-8B).
- **Throughput-latency Pareto** across concurrency 1->32, STATIC vs COMPACT - the plot
  that demonstrates the confidence scheduler's value.

See the [runbook](./dspark_runbook.md) for exact commands.
