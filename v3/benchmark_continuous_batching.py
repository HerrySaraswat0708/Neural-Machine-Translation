"""Phase 7d benchmark: static batching (a new batch only starts once the
previous one has fully drained) vs. continuous batching (evict/admit
every tick) under the same simulated stream of staggered-arrival
requests. Reports per-request latency (arrival -> finish) percentiles,
where continuous batching's scheduling-policy advantage should show up
directly: static batching makes every request in a cohort wait behind
that cohort's slowest member, continuous batching doesn't.
"""
import argparse
import random
import time

import torch

import manifest
from model import load_model
from serving.batching import run_continuous, run_static
from serving.kv_cache_naive import NaiveContiguousCache
from tokenization import TOKENIZER_REGISTRY
from tokenization.base import EOS_ID


def percentile(sorted_vals, p):
    idx = min(int(len(sorted_vals) * p), len(sorted_vals) - 1)
    return sorted_vals[idx]


def summarize(name, completed, wall_time):
    latencies = sorted(r.finish_tick - r.arrival_tick for r in completed)
    total_tokens = sum(r.step_count for r in completed)
    print(f"=== {name} ===")
    print(f"  requests completed: {len(completed)}")
    print(f"  wall-clock time: {wall_time:.2f}s   tokens/sec: {total_tokens / wall_time:.2f}")
    print(f"  latency (ticks) -- mean: {sum(latencies) / len(latencies):.1f}  "
          f"P50: {percentile(latencies, 0.50)}  P95: {percentile(latencies, 0.95)}  "
          f"P99: {percentile(latencies, 0.99)}  max: {latencies[-1]}")
    return latencies


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", default="bpe_joint_rope_mha")
    parser.add_argument("--n_requests", type=int, default=40)
    parser.add_argument("--max_batch_size", type=int, default=8)
    parser.add_argument("--stagger_ticks", type=int, default=3)
    parser.add_argument("--seed", type=int, default=11)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    entry = manifest.load()[args.variant]
    model, cfg = load_model(entry["checkpoint_path"], device=device)
    src_tok = TOKENIZER_REGISTRY[entry["tokenizer_type"]].load(entry["vocab_dir"] + "/src")

    num_layers, num_kv_heads = cfg["num_layers"], model.num_kv_heads
    head_dim = cfg["emb_dim"] // cfg["num_heads"]

    pairs = []
    with open("../data/splits/test.tsv", encoding="utf-8") as f:
        for line in f:
            eng, _ = line.rstrip("\n").split("\t")
            pairs.append(eng)
    random.seed(args.seed)
    sentences = random.sample(pairs, args.n_requests)

    # Staggered arrivals: a Poisson-ish stream, requests arriving faster
    # than a slow batch could clear, so the two policies actually diverge.
    rng = random.Random(args.seed)
    arrivals = []
    tick = 0
    for i, s in enumerate(sentences):
        tick += rng.randint(1, args.stagger_ticks * 2)
        src_ids = src_tok.encode(s) + [EOS_ID]
        arrivals.append((i, src_ids, tick))

    print(f"variant={args.variant} n_requests={args.n_requests} max_batch_size={args.max_batch_size} "
          f"stagger_ticks={args.stagger_ticks} last_arrival_tick={arrivals[-1][2]}")
    print()

    backend_static = NaiveContiguousCache(num_layers, num_kv_heads, head_dim, model.max_seq_len, device)
    t0 = time.perf_counter()
    completed_static = run_static(model, backend_static, list(arrivals), args.max_batch_size, model.max_seq_len)
    t_static = time.perf_counter() - t0
    lat_static = summarize("Static batching", completed_static, t_static)

    print()
    backend_cont = NaiveContiguousCache(num_layers, num_kv_heads, head_dim, model.max_seq_len, device)
    t0 = time.perf_counter()
    completed_cont = run_continuous(model, backend_cont, list(arrivals), args.max_batch_size, model.max_seq_len)
    t_cont = time.perf_counter() - t0
    lat_cont = summarize("Continuous batching", completed_cont, t_cont)

    print()
    print("=== Comparison ===")
    print(f"  P50 latency: {lat_static[len(lat_static)//2] / max(lat_cont[len(lat_cont)//2], 1):.2f}x lower with continuous batching")
    p95_static = percentile(lat_static, 0.95)
    p95_cont = percentile(lat_cont, 0.95)
    print(f"  P95 latency: static={p95_static} ticks, continuous={p95_cont} ticks "
          f"({p95_static / max(p95_cont, 1):.2f}x lower with continuous batching)")
    print(f"  wall-clock speedup: {t_static / t_cont:.2f}x")


if __name__ == "__main__":
    main()
