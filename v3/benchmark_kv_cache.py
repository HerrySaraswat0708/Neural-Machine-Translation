"""Phase 7a/7b benchmark: naive full-recompute decoding (greedy_decode,
what the project has always used) vs. KV-cached decoding
(greedy_decode_cached). Reports tokens/sec and per-sentence latency for
both, plus the speedup -- the control-group number everything else in
Phase 7 gets compared against.
"""
import argparse
import random
import time

import torch

import manifest
from decoding import greedy_decode, greedy_decode_cached
from model import load_model
from tokenization import TOKENIZER_REGISTRY
from tokenization.base import EOS_ID


def run(model, src_tok, sentences, decode_fn, warmup=5):
    src_ids_list = [src_tok.encode(s) + [EOS_ID] for s in sentences]

    for src_ids in src_ids_list[:warmup]:
        decode_fn(model, src_ids)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    per_sentence_latency_ms = []
    total_tokens = 0
    t_start = time.perf_counter()
    for src_ids in src_ids_list:
        t0 = time.perf_counter()
        out_ids, _ = decode_fn(model, src_ids)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        per_sentence_latency_ms.append((t1 - t0) * 1000)
        total_tokens += len(out_ids)
    total_time = time.perf_counter() - t_start

    per_sentence_latency_ms.sort()
    n = len(per_sentence_latency_ms)
    return {
        "n_sentences": n,
        "total_tokens": total_tokens,
        "total_time_sec": total_time,
        "tokens_per_sec": total_tokens / total_time,
        "avg_latency_ms": sum(per_sentence_latency_ms) / n,
        "p50_latency_ms": per_sentence_latency_ms[n // 2],
        "p95_latency_ms": per_sentence_latency_ms[int(n * 0.95)],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", default="bpe_joint_rope_mha")
    parser.add_argument("--n", type=int, default=300)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    entry = manifest.load()[args.variant]
    model, _ = load_model(entry["checkpoint_path"], device=device)
    tok_cls = TOKENIZER_REGISTRY[entry["tokenizer_type"]]
    src_tok = tok_cls.load(entry["vocab_dir"] + "/src")

    pairs = []
    with open("../data/splits/test.tsv", encoding="utf-8") as f:
        for line in f:
            eng, _ = line.rstrip("\n").split("\t")
            pairs.append(eng)
    random.seed(42)
    sentences = random.sample(pairs, args.n)

    print(f"variant={args.variant} device={device} n_sentences={args.n}")
    print()

    naive = run(model, src_tok, sentences, greedy_decode)
    print("=== 7a: naive (full recompute every step) ===")
    for k, v in naive.items():
        print(f"  {k}: {v:.2f}" if isinstance(v, float) else f"  {k}: {v}")

    cached = run(model, src_tok, sentences, greedy_decode_cached)
    print("\n=== 7b: KV-cached ===")
    for k, v in cached.items():
        print(f"  {k}: {v:.2f}" if isinstance(v, float) else f"  {k}: {v}")

    print(f"\n=== Speedup ===")
    print(f"  tokens/sec: {cached['tokens_per_sec'] / naive['tokens_per_sec']:.2f}x")
    print(f"  avg latency: {naive['avg_latency_ms'] / cached['avg_latency_ms']:.2f}x faster")
    print(f"  p95 latency: {naive['p95_latency_ms'] / cached['p95_latency_ms']:.2f}x faster")


if __name__ == "__main__":
    main()
