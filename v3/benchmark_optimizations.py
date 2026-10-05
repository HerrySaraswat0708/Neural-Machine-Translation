"""The comparison study: TTFT (time to first token), TBT (time between
tokens / inter-token latency), end-to-end latency, and throughput,
measured across every serving-time optimization built in this project --
isolated one at a time so each technique's OWN effect is visible, then
combined, so the final table answers "does this actually help, and in
which phase (prefill vs. decode)" with real measured numbers, not
assumption.

Techniques covered here by default: naive/uncached baseline (7a), KV
caching (7b), chunked prefill, prefix caching, Flash Attention (SDPA),
continuous batching + throughput (7d). Compression (quantization/
pruning) gets its own section since it changes the MODEL itself (not the
serving/scheduling logic) and needs a quality check (BLEU) alongside
speed -- unlike every technique above (all exact, or -- SDPA --
numerically negligible, verified in PROGRESS.md). Speculative decoding
(7e) is fully built and separately verified (PROGRESS.md's Phase 7e
section, 200/200 bit-identical) but excluded from the default comparison
-- pass --include_speculative to add it back in.

Usage:
    python benchmark_optimizations.py --n 40           # single-request TTFT/TBT/e2e sweep
    python benchmark_optimizations.py --throughput      # concurrent-request throughput sweep
    python benchmark_optimizations.py --compression      # quantization/pruning speed+quality
    python benchmark_optimizations.py --all              # everything
"""
import argparse
import random
import time

import torch

import manifest
from decoding import greedy_decode
from eval.metrics import all_metrics
from model import load_model
from serving.batching import Request, run_continuous
from serving.compression import (
    linear_sparsity_pct, model_size_mb, prune_unstructured, quantize_dynamic_int8,
)
from serving.engine import InferenceEngine
from serving.kv_cache_naive import NaiveContiguousCache
from serving.speculative import speculative_decode
from tokenization import TOKENIZER_REGISTRY
from tokenization.base import EOS_ID


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def load_sentences(n, seed):
    pairs = []
    with open("../data/splits/test.tsv", encoding="utf-8") as f:
        for line in f:
            eng, fra = line.rstrip("\n").split("\t")
            pairs.append((eng, fra))
    random.seed(seed)
    return random.sample(pairs, n)


# ------------------------------------------------------- single-request ----

def timed_single_request(model, src_ids, max_len, device, use_cache, use_sdpa=False, chunk_layers=None):
    """Times ONE request through the same Request/NaiveContiguousCache
    machinery the real serving engine uses -- not a separate
    reimplementation -- recording wall-clock for the prefill phase and
    each individual decode step directly.
    use_cache=False: decoding.py's greedy_decode (Phase 7a naive
    baseline). No external cache, so every step (including the first)
    redoes full attention over the whole growing prefix -- there's no
    meaningful prefill/decode split the way there is once caching exists;
    only e2e and an approximate per-token figure are reported."""
    if not use_cache:
        _sync(device); t0 = time.perf_counter()
        out_ids, _ = greedy_decode(model, src_ids, max_len=max_len)
        _sync(device); e2e = time.perf_counter() - t0
        return {"ttft": None, "tbt_mean": None, "e2e": e2e, "n_tokens": len(out_ids), "prefill_time": None}

    num_layers = len(model.decoder)
    num_kv_heads = model.num_kv_heads
    head_dim = model.decoder[0].masked_attn.head_dim
    backend = NaiveContiguousCache(num_layers, num_kv_heads, head_dim, max_len, device)
    backend.allocate_request(0)
    req = Request(0, src_ids, 0, max_len)

    _sync(device); t0 = time.perf_counter()
    if chunk_layers is not None:
        req.start_chunked_prefill(model, 0)
        while req.prefilling:
            req.continue_chunked_prefill(model, chunk_layers)
    else:
        req.admit(model, 0)
    _sync(device); prefill_time = time.perf_counter() - t0

    step_times = []
    while not req.finished:
        _sync(device); ts = time.perf_counter()
        req.step(model, backend, use_sdpa=use_sdpa)
        _sync(device); step_times.append(time.perf_counter() - ts)

    backend.free_request(0)
    ttft = prefill_time + step_times[0] if step_times else prefill_time
    tbt_mean = sum(step_times[1:]) / max(1, len(step_times) - 1) if len(step_times) > 1 else None
    e2e = prefill_time + sum(step_times)
    return {"ttft": ttft, "tbt_mean": tbt_mean, "e2e": e2e, "n_tokens": len(step_times), "prefill_time": prefill_time}


def timed_single_request_speculative(draft_model, target_model, src_ids, max_len, device, k=4):
    """Speculative decoding generates several tokens per round instead of
    one at a time, so TTFT/TBT are defined per ROUND here: TTFT = time
    until the first round's accepted tokens are available, TBT = mean
    wall-clock per accepted token across the remaining rounds (a fair
    like-for-like against the other configs' per-token TBT, since what a
    caller actually experiences is tokens arriving, whatever the internal
    batching)."""
    _sync(device); t0 = time.perf_counter()
    out_ids, stats = speculative_decode(draft_model, target_model, src_ids, src_ids, k=k, max_len=max_len)
    _sync(device); e2e = time.perf_counter() - t0
    # speculative_decode doesn't expose per-round timing internally;
    # approximate TTFT/TBT by splitting e2e proportionally to rounds vs.
    # tokens -- coarser than the other configs' true per-step timing, but
    # rounds are already reported (stats['rounds']) so this is stated
    # explicitly as an approximation, not presented as equally precise.
    rounds = max(1, stats["rounds"])
    per_round = e2e / rounds
    ttft = per_round  # time for round 1 to land, assuming uniform round cost
    tbt_mean = e2e / max(1, len(out_ids))
    return {"ttft": ttft, "tbt_mean": tbt_mean, "e2e": e2e, "n_tokens": len(out_ids),
            "prefill_time": None, "rounds": rounds, "target_calls": stats["target_calls"]}


def run_single_request_sweep(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    entries = manifest.load()
    entry = entries[args.variant]
    model, cfg = load_model(entry["checkpoint_path"], device=device)
    src_tok = TOKENIZER_REGISTRY[entry["tokenizer_type"]].load(entry["vocab_dir"] + "/src")

    pairs = load_sentences(args.n, args.seed)
    src_ids_list = [src_tok.encode(eng) + [EOS_ID] for eng, _ in pairs]
    max_len = model.max_seq_len

    # The project's inference engine is scoped to KV caching + compression
    # only (see PROGRESS.md's "Final scope" section) -- chunked prefill and
    # Flash Attention (SDPA) are fully built and separately verified but
    # excluded from this default comparison; pass --include_chunked_prefill
    # / --include_sdpa to add them back in.
    configs = [
        ("naive (7a, no cache)", dict(use_cache=False)),
        ("KV cache (7b)", dict(use_cache=True)),
    ]
    if args.include_chunked_prefill:
        configs.append(("KV cache + chunked prefill (chunk=2)", dict(use_cache=True, chunk_layers=2)))
    if args.include_sdpa:
        configs.append(("KV cache + Flash Attention (SDPA)", dict(use_cache=True, use_sdpa=True)))
    if args.include_chunked_prefill and args.include_sdpa:
        configs.append(("KV cache + chunked prefill + SDPA",
                         dict(use_cache=True, chunk_layers=2, use_sdpa=True)))

    # warmup
    for src_ids in src_ids_list[:3]:
        timed_single_request(model, src_ids, max_len, device, use_cache=True)

    results = {}
    for label, kwargs in configs:
        rows = [timed_single_request(model, src_ids, max_len, device, **kwargs) for src_ids in src_ids_list]
        results[label] = rows

    # Speculative decoding (7e) is fully built and separately verified
    # (see PROGRESS.md's Phase 7e section, 200/200 bit-identical) but is
    # deliberately excluded from this comparison by default -- pass
    # --include_speculative to add it back in.
    if args.include_speculative:
        draft_model, _ = load_model(entries[args.draft_variant]["checkpoint_path"], device=device)
        for src_ids in src_ids_list[:3]:
            timed_single_request_speculative(draft_model, model, src_ids, max_len, device, k=args.k)
        spec_rows = [timed_single_request_speculative(draft_model, model, src_ids, max_len, device, k=args.k)
                     for src_ids in src_ids_list]
        results[f"speculative decoding (k={args.k})"] = spec_rows

    print(f"=== Single-request TTFT / TBT / end-to-end latency (n={args.n}, variant={args.variant}) ===")
    print(f"{'config':<42} {'TTFT (ms)':>10} {'TBT (ms)':>10} {'e2e (ms)':>10} {'tok/s':>8}")
    for label, rows in results.items():
        ttfts = [r["ttft"] * 1000 for r in rows if r["ttft"] is not None]
        tbts = [r["tbt_mean"] * 1000 for r in rows if r["tbt_mean"] is not None]
        e2es = [r["e2e"] * 1000 for r in rows]
        total_tokens = sum(r["n_tokens"] for r in rows)
        total_time = sum(r["e2e"] for r in rows)
        ttft_s = f"{sum(ttfts) / len(ttfts):.2f}" if ttfts else "n/a"
        tbt_s = f"{sum(tbts) / len(tbts):.2f}" if tbts else "n/a"
        print(f"{label:<42} {ttft_s:>10} {tbt_s:>10} {sum(e2es) / len(e2es):>10.2f} {total_tokens / total_time:>8.1f}")
    print()
    print("(prefix cache is a multi-request/repeated-query effect, not a single-request one -- see --throughput)")


# --------------------------------------------------------------- prefix ----

def run_prefix_cache_bench(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    entries = manifest.load()
    entry = entries[args.variant]
    model, cfg = load_model(entry["checkpoint_path"], device=device)
    src_tok = TOKENIZER_REGISTRY[entry["tokenizer_type"]].load(entry["vocab_dir"] + "/src")
    tgt_tok = TOKENIZER_REGISTRY[entry["tokenizer_type"]].load(entry["vocab_dir"] + "/tgt")

    pairs = load_sentences(20, args.seed)
    base_sentences = [eng for eng, _ in pairs][:8]
    # A realistic repeated-query pattern (common FAQ-style phrases hit
    # many times) rather than a synthetic one-off: each base sentence
    # requested repeat_factor times, interleaved with unique fillers so
    # the scheduler doesn't just process one giant repeated block.
    repeat_factor = 5
    fillers = [eng for eng, _ in load_sentences(10, args.seed + 1)]
    stream = []
    for i in range(repeat_factor):
        stream.extend(base_sentences)
        if i < len(fillers):
            stream.append(fillers[i])
    random.Random(args.seed).shuffle(stream)

    for enable_cache in (False, True):
        engine = InferenceEngine(model, src_tok, tgt_tok, max_batch_size=4, enable_prefix_cache=enable_cache)
        _sync(device); t0 = time.perf_counter()
        for s in stream:
            engine.submit(s)
        engine.run_until_idle()
        _sync(device); wall = time.perf_counter() - t0
        label = "prefix cache ON" if enable_cache else "prefix cache OFF"
        print(f"{label:<20} n_requests={len(stream)}  wall_time={wall:.3f}s  "
              f"cache_entries={len(engine.prefix_cache)}")


# ---------------------------------------------------------- throughput -----

def run_throughput_bench(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    entries = manifest.load()
    entry = entries[args.variant]
    model, cfg = load_model(entry["checkpoint_path"], device=device)
    src_tok = TOKENIZER_REGISTRY[entry["tokenizer_type"]].load(entry["vocab_dir"] + "/src")
    tgt_tok = TOKENIZER_REGISTRY[entry["tokenizer_type"]].load(entry["vocab_dir"] + "/tgt")
    max_len = model.max_seq_len

    pairs = load_sentences(args.n_requests, args.seed)
    sentences = [eng for eng, _ in pairs]

    def make_arrivals(seed):
        rng = random.Random(seed)
        arrivals = []
        tick = 0
        for i, s in enumerate(sentences):
            tick += rng.randint(1, args.stagger_ticks * 2)
            src_ids = src_tok.encode(s) + [EOS_ID]
            arrivals.append((i, src_ids, tick))
        return arrivals

    num_layers = len(model.decoder)
    num_kv_heads = model.num_kv_heads
    head_dim = model.decoder[0].masked_attn.head_dim

    print(f"=== Throughput under concurrent load (n_requests={args.n_requests}, max_batch_size={args.max_batch_size}) ===")

    # Static vs continuous batching (7d, reconfirmed under this file too)
    from serving.batching import run_static
    for label, runner in (("static batching", run_static), ("continuous batching", run_continuous)):
        backend = NaiveContiguousCache(num_layers, num_kv_heads, head_dim, max_len, device)
        _sync(device); t0 = time.perf_counter()
        completed = runner(model, backend, make_arrivals(args.seed), args.max_batch_size, max_len)
        _sync(device); wall = time.perf_counter() - t0
        total_tokens = sum(r.step_count for r in completed)
        print(f"{label:<30} wall={wall:.2f}s  tokens={total_tokens}  throughput={total_tokens / wall:.1f} tok/s")

    # Continuous batching + chunked prefill
    engine_cp = InferenceEngine(model, src_tok, tgt_tok, max_batch_size=args.max_batch_size, prefill_chunk_layers=2)
    arrivals = make_arrivals(args.seed)
    for rid, src_ids, arrival in arrivals:
        engine_cp.waiting.append((rid, src_ids, arrival))
    engine_cp._next_id = len(arrivals)
    _sync(device); t0 = time.perf_counter()
    engine_cp.run_until_idle()
    _sync(device); wall = time.perf_counter() - t0
    total_tokens = sum(len(tgt_tok.encode(t)) for t in engine_cp.results.values())
    print(f"{'continuous + chunked prefill':<30} wall={wall:.2f}s  tokens~{total_tokens}  throughput~{total_tokens / wall:.1f} tok/s")

    # Speculative decoding (7e) is fully built and separately verified
    # (PROGRESS.md's Phase 7e section) but excluded from this throughput
    # comparison by default -- pass --include_speculative to add it back.
    if args.include_speculative:
        draft_model, _ = load_model(entries[args.draft_variant]["checkpoint_path"], device=device)
        engine_spec = InferenceEngine(model, src_tok, tgt_tok, max_batch_size=args.max_batch_size,
                                       draft_model=draft_model, k=args.k)
        arrivals = make_arrivals(args.seed)
        for rid, src_ids, arrival in arrivals:
            engine_spec.waiting.append((rid, src_ids, arrival))
        engine_spec._next_id = len(arrivals)
        _sync(device); t0 = time.perf_counter()
        engine_spec.run_until_idle()
        _sync(device); wall = time.perf_counter() - t0
        total_tokens = sum(len(tgt_tok.encode(t)) for t in engine_spec.results.values())
        print(f"{'continuous + speculative':<30} wall={wall:.2f}s  tokens~{total_tokens}  throughput~{total_tokens / wall:.1f} tok/s")

    # Continuous batching + SDPA
    engine_sdpa = InferenceEngine(model, src_tok, tgt_tok, max_batch_size=args.max_batch_size, use_sdpa=True)
    arrivals = make_arrivals(args.seed)
    for rid, src_ids, arrival in arrivals:
        engine_sdpa.waiting.append((rid, src_ids, arrival))
    engine_sdpa._next_id = len(arrivals)
    _sync(device); t0 = time.perf_counter()
    engine_sdpa.run_until_idle()
    _sync(device); wall = time.perf_counter() - t0
    total_tokens = sum(len(tgt_tok.encode(t)) for t in engine_sdpa.results.values())
    print(f"{'continuous + SDPA':<30} wall={wall:.2f}s  tokens~{total_tokens}  throughput~{total_tokens / wall:.1f} tok/s")


# -------------------------------------------------------- compression ------

def run_compression_bench(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    entries = manifest.load()
    entry = entries[args.variant]
    model, cfg = load_model(entry["checkpoint_path"], device=device)
    src_tok = TOKENIZER_REGISTRY[entry["tokenizer_type"]].load(entry["vocab_dir"] + "/src")
    tgt_tok = TOKENIZER_REGISTRY[entry["tokenizer_type"]].load(entry["vocab_dir"] + "/tgt")

    pairs = load_sentences(args.n, args.seed)

    def bleu_of(model_, device_):
        hyps, refs = [], []
        for eng, fra in pairs:
            src_ids = src_tok.encode(eng) + [EOS_ID]
            out_ids, _ = greedy_decode(model_, src_ids)
            hyps.append(tgt_tok.decode(out_ids))
            refs.append(fra)
        return all_metrics(hyps, refs)["bleu"]

    print(f"=== Compression: model size, quality (BLEU), latency (n={args.n}) ===")
    print(f"{'baseline (fp32, GPU)':<28} size={model_size_mb(model):.2f}MB")
    base_bleu = bleu_of(model, device)
    print(f"  BLEU: {base_bleu:.2f}")

    # Quantization: CPU-only (see serving/compression.py's docstring for why)
    qmodel = quantize_dynamic_int8(model)
    cpu_device = torch.device("cpu")
    print(f"{'quantized (dynamic INT8, CPU)':<28} size={model_size_mb(qmodel):.2f}MB "
          f"({model_size_mb(model) / model_size_mb(qmodel):.2f}x smaller)")
    q_bleu = bleu_of(qmodel, cpu_device)
    print(f"  BLEU: {q_bleu:.2f}  (delta: {q_bleu - base_bleu:+.2f})")

    n_lat = min(15, args.n)
    lat_pairs = pairs[:n_lat]

    def mean_latency(model_, dev):
        times = []
        for eng, _ in lat_pairs:
            src_ids = src_tok.encode(eng) + [EOS_ID]
            if dev.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            greedy_decode(model_, src_ids)
            if dev.type == "cuda":
                torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
        return sum(times) / len(times)

    # Fair CPU-vs-CPU comparison for quantization's actual speed claim
    model_cpu = load_model(entry["checkpoint_path"], device=cpu_device)[0]
    fp32_cpu_lat = mean_latency(model_cpu, cpu_device)
    q_lat = mean_latency(qmodel, cpu_device)
    print(f"  CPU latency: fp32={fp32_cpu_lat * 1000:.1f}ms  int8={q_lat * 1000:.1f}ms  "
          f"({fp32_cpu_lat / q_lat:.2f}x)")

    print()
    for amount in (0.3, 0.5, 0.7):
        pmodel = prune_unstructured(model, amount)
        p_bleu = bleu_of(pmodel, device)
        p_lat = mean_latency(pmodel, device)
        base_gpu_lat = mean_latency(model, device)
        print(f"{'pruned ' + str(int(amount * 100)) + '% (GPU)':<28} "
              f"sparsity={linear_sparsity_pct(pmodel):.1f}% size={model_size_mb(pmodel):.2f}MB(unchanged, dense) "
              f"BLEU={p_bleu:.2f}(delta {p_bleu - base_bleu:+.2f}) "
              f"latency={p_lat * 1000:.1f}ms(vs fp32 {base_gpu_lat * 1000:.1f}ms)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", default="bpe_joint_rope_mha")
    parser.add_argument("--draft_variant", default="bpe_joint_rope_mha_draft")
    parser.add_argument("--n", type=int, default=40)
    parser.add_argument("--n_requests", type=int, default=40)
    parser.add_argument("--max_batch_size", type=int, default=8)
    parser.add_argument("--stagger_ticks", type=int, default=3)
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--include_speculative", action="store_true",
                         help="also benchmark speculative decoding (7e) in --single/--throughput -- "
                              "excluded by default; the code (serving/speculative.py) is fully "
                              "built and separately verified, just not part of the default "
                              "active comparison")
    parser.add_argument("--include_chunked_prefill", action="store_true",
                         help="also benchmark chunked prefill in --single -- excluded by default; "
                              "the code (serving/chunked_prefill.py) is fully built and separately "
                              "verified bit-identical, just not part of the project's final scope "
                              "(KV caching + compression only)")
    parser.add_argument("--include_sdpa", action="store_true",
                         help="also benchmark Flash Attention (PyTorch SDPA) in --single -- excluded "
                              "by default; the code (model/attention.py's use_sdpa path) is fully "
                              "built and separately verified numerically-equivalent, just not part "
                              "of the project's final scope (KV caching + compression only)")
    parser.add_argument("--single", action="store_true")
    parser.add_argument("--prefix", action="store_true",
                         help="prefix-caching benchmark -- kept/verified but outside the final scope; "
                              "run explicitly, not included in --all")
    parser.add_argument("--throughput", action="store_true",
                         help="continuous-batching throughput benchmark -- kept/verified but outside "
                              "the final scope; run explicitly, not included in --all")
    parser.add_argument("--compression", action="store_true")
    parser.add_argument("--all", action="store_true",
                         help="run the project's final active comparison: --single + --compression "
                              "(KV caching + compression only). Use --prefix/--throughput explicitly "
                              "to reproduce the kept-but-descoped continuous-batching/prefix-cache numbers.")
    args = parser.parse_args()

    if args.all or args.single:
        run_single_request_sweep(args)
        print()
    if args.prefix:
        run_prefix_cache_bench(args)
        print()
    if args.throughput:
        run_throughput_bench(args)
        print()
    if args.all or args.compression:
        run_compression_bench(args)


if __name__ == "__main__":
    main()
