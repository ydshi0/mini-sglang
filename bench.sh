CUDA_VISIBLE_DEVICES=7  python benchmark/offline/bench_spec1.py \
        --target-model /shared_LLM_model/Qwen/Qwen3-8B \
        --draft-model /shared_LLM_model/Qwen/Qwen3-1.7B \
        --num-spec-tokens 5