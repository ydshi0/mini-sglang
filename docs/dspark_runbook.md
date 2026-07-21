# DSpark on Mini-SGLang - Runbook (how to set up, test, benchmark)

Concrete commands to set up, validate, and benchmark the DSpark port
([design doc](./dspark_speculative_decoding.md)). Written for the L20D box
(8x GPU, HuggingFace unreachable, ModelScope reachable).

> The pure-logic core is CPU-unit-tested and ready to run. Steps 3-6 (weights, GPU
> validation, benchmarking) are what remains to turn this into measured numbers.
> Nothing here has been executed on this machine yet (no torch/CUDA env installed);
> these are the exact commands to run once the env is up.

## 0. TL;DR

```bash
# env
curl -LsSf https://astral.sh/uv/install.sh | sh    # if uv missing; else use pip/conda
cd ~/mini-sglang && uv venv --python=3.12 && source .venv/bin/activate
uv pip install -e .

# pure-logic tests (no GPU / no weights needed)
pytest tests/spec/test_dspark_logic.py -q

# weights (ModelScope, since HF is blocked here)
pip install modelscope
python docs/scripts/fetch_weights.py    # downloads Qwen3-8B + dspark_qwen3_8b_block7

# inspect the draft checkpoint's real key names (reconcile spec/draft_model.py::remap_draft_key)
python docs/scripts/inspect_checkpoint.py models/dspark_qwen3_8b
```

## 1. Environment

Mini-sglang needs CUDA torch + `flashinfer` + `sgl-kernel` (JIT-compiled on first
run). On this box there is no torch/conda/uv yet:

```bash
# Option A: uv (recommended by the repo)
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
cd ~/mini-sglang && uv venv --python=3.12 && source .venv/bin/activate
uv pip install -e .

# Option B: existing conda/python with a matching CUDA toolkit
pip install -e .
```

Sanity: `python -c "import torch; print(torch.cuda.is_available())"` -> `True`.

## 2. Run the CPU unit tests (pure logic - run this first)

These cover acceptance (greedy longest-prefix + chain speculative sampling), the
confidence planner (survival / throughput-optimal budget / top-k schedule), and the
Markov head math. No GPU, no weights:

```bash
pytest tests/spec/test_dspark_logic.py -q
# or a single case:
pytest tests/spec/test_dspark_logic.py::test_accept_greedy_partial_prefix -q
```

Expected: all pass. This is the correctness gate for the DSpark *algorithm* before any
GPU work.

## 3. Get the weights (ModelScope)

HuggingFace is unreachable on this host; ModelScope mirrors both models.

```python
# docs/scripts/fetch_weights.py
from modelscope import snapshot_download
# target
snapshot_download("Qwen/Qwen3-8B", local_dir="models/Qwen3-8B")
# DSpark drafter for Qwen3-8B (block size 7). If the deepseek-ai org is not mirrored
# on ModelScope, use the community port Dogacel/Qwen3-8B-DSpark, or copy the HF repo
# deepseek-ai/dspark_qwen3_8b_block7 via an allowed transfer.
snapshot_download("deepseek-ai/dspark_qwen3_8b_block7", local_dir="models/dspark_qwen3_8b")
```

## 4. Reconcile checkpoint key names

`spec/draft_model.py::remap_draft_key` maps checkpoint keys to module params. The
Markov/confidence spellings are taken from the sglang reference; verify them against
the actual file and adjust the two `replace(...)` lines if needed:

```python
# docs/scripts/inspect_checkpoint.py
import sys, glob, safetensors.torch as st
for f in glob.glob(f"{sys.argv[1]}/*.safetensors"):
    for k, v in st.load_file(f).items():
        print(tuple(v.shape), k)
```

Expected families: `layers.*`, `fc.weight`, `hidden_norm.weight`, `norm.weight`,
`markov_head.*`, `confidence_head.*`; NO `embed_tokens`/`lm_head`.

## 5. GPU validation & benchmarking (what to run, in order)

Bring up correctness first, then measure. Order matches design-doc section 7:

1. **Draft loads + proposes.** Load target + draft, `attach_shared_modules`, seed
   draft KV from a prefill, run one `DSparkWorker.step` on a single prompt; assert the
   committed tokens are non-empty and decode to sane text.
2. **Losslessness (key gate).** With greedy decoding, DSpark output must be
   **token-identical** to plain Qwen3-8B greedy decoding on the same prompts:
   ```bash
   # baseline
   python -m minisgl --model models/Qwen3-8B --shell         # or offline bench
   # dspark - outputs must match token-for-token
   python -m minisgl --model models/Qwen3-8B --speculative-algorithm dspark \
       --speculative-draft-model models/dspark_qwen3_8b --spec-verify-mode static
   ```
3. **Acceptance length.** Log `mean_accept_len` over a dataset; expect ~3.4 (fully
   trained block7) / ~2.3 (partial). Low numbers => weight-mapping or KV-seeding bug.
4. **Single-stream speedup.** Compare tokens/s vs baseline (spec off). Reference ~1.76x
   on Qwen3-8B. Run:
   ```bash
   python benchmark/offline/bench.py --model models/Qwen3-8B                       # baseline
   python benchmark/offline/bench.py --model models/Qwen3-8B --speculative-algorithm dspark \
       --speculative-draft-model models/dspark_qwen3_8b
   ```
5. **Throughput-latency Pareto.** Sweep concurrency 1,2,4,8,16,32; plot tokens/s vs
   TPOT for `--spec-verify-mode static` vs `compact`. COMPACT (confidence-scheduled)
   should push the frontier out at higher concurrency - the headline plot:
   ```bash
   for c in 1 2 4 8 16 32; do
     python benchmark/online/bench_qwen.py --model models/Qwen3-8B \
        --speculative-algorithm dspark --speculative-draft-model models/dspark_qwen3_8b \
        --spec-verify-mode static  --max-concurrency $c
     python benchmark/online/bench_qwen.py --model models/Qwen3-8B \
        --speculative-algorithm dspark --speculative-draft-model models/dspark_qwen3_8b \
        --spec-verify-mode compact --max-concurrency $c
   done
   ```

Datasets: GSM8K / HumanEval (structured tasks accept longer blocks than open chat).

> The `--speculative-*` / `--spec-verify-mode` flags are added when the scheduler
> wiring in design-doc section 7 lands (add to `server/args.py` + `engine/config.py`).

## 6. Common failure modes

| Symptom | Likely cause |
|---|---|
| `Unexpected keys in state_dict` on draft load | `remap_draft_key` spelling vs real checkpoint (section 4) |
| accept length ~= 1.0 | draft KV not seeded / wrong positions in `seed_draft_kv` |
| output differs from baseline greedy | acceptance rule or verify-window alignment bug |
| `fc.weight shape mismatch` | `target_layer_ids` (K) != checkpoint's context features |
| OOM at verify | verify allocates `W` slots/req; lower max batch or `--num-pages` |
