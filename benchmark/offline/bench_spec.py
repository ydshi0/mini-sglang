#!/usr/bin/env python3
"""
Speculative Decoding Example for Mini-SGLang.

Both target (Qwen3-8B) and draft (Qwen3-1.7B) models run on
Mini-SGLang's inference stack with paged attention.

"""

import argparse
import time

import torch
from transformers import AutoTokenizer

from minisgl.core import SamplingParams
from minisgl.speculative import SpeculativeLLM
from minisgl.llm import LLM

def load_questions() -> list[str]:
    data_path = "/workspace/ydshi/dataset/gsm8k/gsm8k_256.txt"
    with open(data_path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f.readlines() if line.strip()]

def run(args):


    print("Speculative Decoding (Mini-SGLang)")
    print(f"  Target: {args.target_model}")
    print(f"  Draft:  {args.draft_model}")
    print(f"  K:      {args.num_spec_tokens}")
    print("=" * 60)

    kwargs = {}
    if args.dummy_weight:
        kwargs["use_dummy_weight"] = True
    if args.max_seq_len:
        kwargs["max_seq_len_override"] = args.max_seq_len
    kwargs["memory_ratio"] = args.memory_ratio



    sp = SamplingParams(temperature=0.0, max_tokens=args.max_tokens)

    tokenizer = AutoTokenizer.from_pretrained(args.target_model)
    questions = load_questions()
    prompt_token_ids = []
    for q in questions:
        ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": q}],
            tokenize=True,
            add_generation_prompt=True,
        )
        prompt_token_ids.append(ids)

    spec = True
    if(spec):
        spec_llm = SpeculativeLLM(
            target_model_path=args.target_model,
            draft_model_path=args.draft_model,
            num_spec_tokens=args.num_spec_tokens,
            dtype=torch.bfloat16,
            **kwargs,
            attention_backend="fa"
        )

        print("\n--- Generating ---\n")
        bench_results = []
        t = time.time()
        for i in range(len(prompt_token_ids)):
            result = spec_llm.generate([prompt_token_ids[i]], sp)
            bench_results.extend(result)
        t = time.time() - t

        output_lens = []
        for res in bench_results:
            text = res["token_ids"]
            output_lens.append(len(text))
            print(f"Output len {len(text)} preview: '{res['text'].replace('\n', ' ')[:100]}'")
        total_output_tokens = sum(output_lens)

        print("\n--- Benchmark Results ---")
        throughput = total_output_tokens / t if t > 0 else 0.0
        print(f"Total Output: {total_output_tokens}tok, Time: {t:.2f}s, Throughput: {throughput:.2f} tok/s")



    else:
        # ── Normal decoding ──
        normal_llm = LLM(args.target_model, dtype=torch.bfloat16, **kwargs, cuda_graph_max_bs=0, attention_backend="fa")
        
        bench_results = []
        t = time.time()
        for i in range(len(prompt_token_ids)):
            result = normal_llm.generate([prompt_token_ids[i]], sp)
            bench_results.extend(result)
        t = time.time() - t

        output_lens = []
        for res in bench_results:
            text = res["token_ids"]
            output_lens.append(len(text))
            print(f"Output len {len(text)} preview: '{res['text'].replace('\n', ' ')[:100]}'")
        total_output_tokens = sum(output_lens)

        print("\n--- Benchmark Results ---")
        throughput = total_output_tokens / t if t > 0 else 0.0
        print(f"Total Output: {total_output_tokens}tok, Time: {t:.2f}s, Throughput: {throughput:.2f} tok/s")





def main():
    parser = argparse.ArgumentParser(description="Speculative Decoding Example")
    parser.add_argument("--target-model", default="Qwen/Qwen3-8B")
    parser.add_argument("--draft-model", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--num-spec-tokens", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--max-seq-len", type=int, default=300)
    parser.add_argument("--memory-ratio", type=float, default=0.7,
                        help="GPU memory ratio for target KV cache (lower to "
                             "leave room for draft model, default 0.7)")
    parser.add_argument("--dummy-weight", action="store_true")
    args = parser.parse_args()


    run(args)
    


if __name__ == "__main__":
    main()