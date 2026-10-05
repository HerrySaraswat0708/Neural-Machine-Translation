"""Phase 7e benchmark: naive greedy decode (one target-model call per
token) vs. speculative decoding (one target-model call per round of up
to k draft-proposed tokens). Reports wall-clock tokens/sec, target-model
call counts, and draft acceptance rate.
"""
import argparse
import random
import time

import torch

import manifest
from decoding import greedy_decode
from model import load_model
from serving.speculative import speculative_decode
from tokenization import TOKENIZER_REGISTRY
from tokenization.base import EOS_ID


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_variant", default="bpe_joint_rope_mha")
    parser.add_argument("--draft_variant", default="bpe_joint_rope_mha_draft")
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--n", type=int, default=300)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    entries = manifest.load()
    target_model, _ = load_model(entries[args.target_variant]["checkpoint_path"], device=device)
    draft_model, _ = load_model(entries[args.draft_variant]["checkpoint_path"], device=device)
    src_tok = TOKENIZER_REGISTRY["bpe_joint"].load(entries[args.target_variant]["vocab_dir"] + "/src")

    pairs = []
    with open("../data/splits/test.tsv", encoding="utf-8") as f:
        for line in f:
            eng, _ = line.rstrip("\n").split("\t")
            pairs.append(eng)
    random.seed(42)
    sentences = random.sample(pairs, args.n)
    src_ids_list = [src_tok.encode(s) + [EOS_ID] for s in sentences]

    # warmup
    for src_ids in src_ids_list[:5]:
        greedy_decode(target_model, src_ids)
        speculative_decode(draft_model, target_model, src_ids, src_ids, k=args.k)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    total_tokens_naive = 0
    t0 = time.perf_counter()
    for src_ids in src_ids_list:
        out_ids, _ = greedy_decode(target_model, src_ids)
        total_tokens_naive += len(out_ids)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t_naive = time.perf_counter() - t0

    total_tokens_spec = 0
    agg_stats = {"proposed": 0, "accepted": 0, "rounds": 0, "target_calls": 0}
    t0 = time.perf_counter()
    for src_ids in src_ids_list:
        out_ids, stats = speculative_decode(draft_model, target_model, src_ids, src_ids, k=args.k)
        total_tokens_spec += len(out_ids)
        for key in agg_stats:
            agg_stats[key] += stats[key]
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t_spec = time.perf_counter() - t0

    print(f"n_sentences={args.n} k={args.k} target={args.target_variant} draft={args.draft_variant}")
    print()
    print("=== Naive greedy (target model only) ===")
    print(f"  total_tokens: {total_tokens_naive}  target_calls: {total_tokens_naive} (1 per token)")
    print(f"  wall time: {t_naive:.2f}s   tokens/sec: {total_tokens_naive / t_naive:.2f}")
    print()
    print("=== Speculative decoding ===")
    print(f"  total_tokens: {total_tokens_spec}  target_calls: {agg_stats['target_calls']}")
    print(f"  draft tokens proposed: {agg_stats['proposed']}  accepted: {agg_stats['accepted']}  "
          f"acceptance_rate: {100*agg_stats['accepted']/agg_stats['proposed']:.1f}%")
    print(f"  wall time: {t_spec:.2f}s   tokens/sec: {total_tokens_spec / t_spec:.2f}")
    print()
    print("=== Comparison ===")
    print(f"  target-model call reduction: {total_tokens_naive / agg_stats['target_calls']:.2f}x "
          f"fewer target calls ({total_tokens_naive} -> {agg_stats['target_calls']})")
    print(f"  wall-clock speedup: {t_naive / t_spec:.2f}x")


if __name__ == "__main__":
    main()
