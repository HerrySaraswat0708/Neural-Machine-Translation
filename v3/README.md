# v3: Production-Grade NMT Engine (English -> French)

## Why this exists

v1 already proved out tokenization x positional-encoding x decoding as a
controlled comparison. That's a solid *modeling* project. It does not, on
its own, read as "understands how LLMs actually get served in production" --
and that's the gap this version closes. v3 keeps the single eng-fra pair
(no multilingual scope creep -- that's v2's job) and goes deep on two
things a translation-app author usually skips: (1) the attention-mechanism
axis (MHA/MQA/GQA), and (2) a hand-built inference-serving stack. The
**actively demoed and benchmarked** serving engine is scoped to **KV
caching + post-training compression** (quantization + pruning) -- narrow
on purpose, so every number shown is one that can be defended in depth.
Continuous batching, chunked prefill, prefix caching, Flash Attention
(SDPA), and speculative decoding were all also built and independently
verified (bit-identical or numerically equivalent to a naive baseline)
and are kept in the codebase, reproducible via `benchmark_optimizations.py`
flags -- see `PROGRESS.md`'s "Final scope" section -- but excluded from
the live demo and headline comparison table. The throughline for all of
it: MQA/GQA exist *because* they shrink the KV cache, which is *why*
paged/contiguous KV caching, batching, and speculative decoding are all
worth doing in the first place. Framing the whole project around that
causal chain is what makes it read as systems understanding rather than
"trained a transformer."

## Relationship to v1 and v2

- **v1** (repo root): word/BPE/char x sinusoidal/RoPE x greedy/beam/top-k/
  top-p, 6 trained variants, Streamlit app. Stays exactly as-is -- it's a
  finished, working reference and a fallback demo if v3 runs out of runway.
- **v2** (`v2/`): multilingual BPCC, DDP, 4x A6000 cluster scale. Separate
  effort, separate goal (breadth across languages). Not touched by this
  plan.
- **v3** (`v3/`, this doc): single pair, `data/eng-fra.txt` (reused from
  v1 via the same `data/prepare_data.py`-style split, copied in Phase 0),
  maximum depth on architecture breadth + inference-serving depth.

## The axes

| Axis | v1 had | v3 adds | Status |
|---|---|---|---|
| Tokenization | word, BPE, char | **WordPiece** | new |
| Positional encoding | sinusoidal, RoPE | **ALiBi** | new |
| Attention mechanism | MHA only | **MQA, GQA** | new |
| Decoding | greedy, beam, top-k, top-p | -- | reuse v1's `decoding.py` as-is |
| Evaluation | BLEU (sacrebleu) | **chrF++, TER**, latency/throughput/memory benchmarks | new |
| **Inference optimization** | none | **Active/demoed: KV cache, quantization, pruning.** Also built + verified, kept in codebase, not in the active demo: continuous batching, chunked prefill, prefix caching, Flash Attention (SDPA), speculative decoding | new -- this is the main focus (PagedAttention was built, verified, and later removed entirely for scope; the rest were built, verified, and kept but excluded from the active showcase -- see PROGRESS.md's "Scope change" and "Final scope" sections) |

## Directory layout (target)

```
v3/
  README.md              this plan (becomes the project write-up as it fills in)
  PROGRESS.md             session-to-session implementation log (created once work starts)
  data/                    eng-fra split, reused from v1 (copy of data/prepare_data.py output)
  tokenization/            word.py, bpe.py, char.py (ported from v1), wordpiece.py (new)
  model/
    positional.py           sinusoidal, rope, alibi
    attention.py             generalized MHA/MQA/GQA (num_kv_heads param)
    transformer.py
  decoding.py               greedy/beam/top-k/top-p, ported from v1 decoding.py
  train.py                  config-driven, one variant per run (same pattern as v1/v2)
  eval/
    metrics.py               BLEU + chrF++ + TER
    benchmark.py              latency/throughput/memory harness (the numbers backing Phase 7)
  serving/                  the inference engine -- new subsystem, no v1/v2 equivalent
    kv_cache_naive.py         naive per-request contiguous KV cache (the engine's backend)
    chunked_prefill.py         layer-chunked encoder pass (spreads prefill across scheduler ticks)
    compression.py              post-training quantization + pruning
    flash_attention.py          SDPA/Flash-Attention design notes (mechanism lives in model/attention.py)
    batching.py                continuous-batching scheduler (admit/evict mid-generation) + prefix cache
    speculative.py             draft-model propose + target-model verify
    engine.py                  ties cache+batching+speculative+chunked-prefill+prefix-cache into one serving loop
    server.py                  thin async front-end (FastAPI) over engine.py
  configs/
  checkpoints/              gitignored, reproducible via train.py
  results/                  BLEU/chrF/TER tables + benchmark plots
```

## Training strategy: breadth vs. depth, on purpose

Tokenizer (4) x PE (3) x attention (3) is 36 combinations -- training all
of them is busywork, not signal. Split into two matrices instead:

1. **Breadth matrix (12 runs):** tokenizer x PE, attention fixed at MHA,
   same light training budget v1 used. This is the direct v1 upgrade --
   proves WordPiece and ALiBi were actually implemented and measured, not
   just claimed.
2. **Attention matrix (3 runs):** MHA vs MQA vs GQA, tokenizer+PE fixed at
   whatever wins the breadth matrix (prior: BPE + RoPE, the modern-LLM
   default). Report both BLEU delta *and* KV-cache-bytes-per-token delta --
   the second number is what actually motivates Phase 7.

Winner of the attention matrix becomes the one **production model**,
retrained at larger scale (bigger `emb_dim`/layers, full epochs, ideally on
the A6000 cluster). Every Phase 7 serving benchmark targets this one model
-- the serving stack doesn't need to be genericized across 15 checkpoints,
just correct and fast for the one that matters.

## Phases

**Phase 0 -- Foundation.** Scaffold the directories above; copy/adapt v1's
data split and word/BPE/char tokenizers as a starting point rather than
rewriting from scratch.

**Phase 1 -- WordPiece tokenizer.** Same shared tokenizer interface v1
already defines (`tokenization/base.py`); add `wordpiece.py` using HF
`tokenizers`' WordPiece trainer (already a dependency).

**Phase 2 -- ALiBi.** Add as a third `pe_type` alongside sinusoidal/RoPE:
linear bias on attention scores by head-specific slope x relative
distance, no learned/cached embedding table. Same "why RoPE skips
cross-attention" reasoning from v1's README applies to ALiBi's bias term
-- document it the same way.

**Phase 3 -- MQA/GQA.** Generalize v1's `MultiHeadAttention` with a
`num_kv_heads` parameter: `num_kv_heads == num_heads` is MHA (today's
behavior), `num_kv_heads == 1` is MQA, `1 < num_kv_heads < num_heads` is
GQA (K/V heads repeat-interleaved up to `num_heads` before the QK^T
matmul). This one parameter is also exactly what `serving/kv_cache.py`
needs to compute cache size, so get the shape contract right here.

**Phase 4 -- Breadth matrix.** Train the 12 tokenizer x PE variants, eval
BLEU + chrF++ + TER on the held-out split, results table in
`results/breadth_matrix.md`.

**Phase 5 -- Attention matrix.** Train MHA/MQA/GQA on the winning tok+PE
combo. Report BLEU delta and KV-cache-bytes-per-token for each (the latter
is a closed-form function of `num_kv_heads`, `head_dim`, `num_layers` --
no need to measure it empirically, just compute and state it).

**Phase 6 -- Production model.** Retrain the overall winner at a scale
that makes the Phase 7 benchmarks mean something (a tiny model's
"speedup" numbers are noise). Bigger `emb_dim`/`ff_dim`/layers, full data,
full epochs.

**Phase 7 -- Inference engine (main focus).** Built incrementally so each
step has its own before/after benchmark instead of one big-bang rewrite:

- **7a. Naive baseline.** Recompute full K/V every decode step (this is
  what v1's `decoding.py` already does). Benchmark: tokens/sec, latency,
  as the control group everything else is measured against.
- **7b. KV cache.** Cache and incrementally extend K/V per request instead
  of recomputing. Benchmark speedup vs. 7a; verify output is bit-identical
  (this is a pure speed optimization, not an approximation).
- **7c. PagedAttention -- built, verified, then removed.** Block-based KV
  cache allocator (fixed-size pages, non-contiguous, referenced via a
  per-request page table -- the vLLM mechanism) instead of one contiguous
  pre-allocated buffer per request. Built and benchmarked (83.3%->23.1%
  fragmentation, ~4.6x more concurrent requests at fixed memory -- see
  PROGRESS.md's Phase 7c section for the historical record), then
  removed in favor of a broader technique set: chunked prefill, prefix
  caching, Flash Attention (SDPA), and post-training compression
  (quantization + pruning) -- see PROGRESS.md's "Scope change" section.
- **7d. Continuous batching.** Replace static batching with a scheduler
  that admits new requests and evicts finished ones between decode steps
  rather than waiting for the whole batch to finish. Benchmark throughput
  and P50/P95/P99 latency under a simulated stream of concurrent requests
  with staggered arrival times.
- **7e. Speculative decoding.** Train a small draft model (2-layer variant
  of the same architecture, or the char-tokenizer variant from Phase 4 if
  its speed/quality tradeoff fits), draft-then-verify against the
  production model. Benchmark acceptance rate and effective speedup net
  of draft-model overhead.
- **7f. Assemble the engine.** `serving/engine.py` combines paged cache +
  continuous batching + speculative decoding into one serving loop;
  `serving/server.py` puts a thin async API in front of it. Final
  benchmark sweep: every combination of optimizations on/off, one table.

**Phase 8 -- Evaluation report.** Consolidate Phase 4/5 quality tables and
Phase 7 performance tables into `results/`, with plots (throughput vs.
concurrency, latency percentiles, memory vs. batch size).

**Phase 9 -- Demo + packaging.** A small demo (Streamlit, or FastAPI +
curl/simple UI) that hits `serving/server.py` so the optimizations are
visibly running, not just benchmarked in isolation. Rewrite this README
into a project write-up once the numbers exist; draft resume bullets from
it (see below).

## What gets measured, precisely (so the numbers are defensible)

- **Throughput:** tokens/sec at increasing concurrency levels.
- **Latency:** time-to-first-token, inter-token latency, end-to-end
  P50/P95/P99.
- **Memory:** KV cache bytes/request (closed-form from Phase 5), peak
  allocator memory, fragmentation % -- naive-contiguous vs. paged.
- **Batching efficiency:** requests/sec vs. concurrent-request count.
- **Speculative decoding:** draft acceptance rate, net speedup after
  subtracting draft-model cost.
- **Correctness:** 7b/7c/7d must reproduce 7a's greedy-decode output
  exactly (they're speed/memory optimizations, not approximations); only
  7e (speculative decoding) needs a distributional-equivalence argument
  instead of exact match, since it changes *which* tokens get proposed
  before verification.

## Open dependency: GPU access

This machine currently reports `torch.cuda.is_available() == False`.
`v2/PROGRESS.md` records a working 4x A6000 cluster used for v2 -- confirm
whether that's reachable for v3 too. It matters more here than it did for
v1: Phase 6's production model and essentially all of Phase 7's
throughput/batching/memory numbers are only convincing measured on a real
GPU. CPU-only numbers are fine for Phases 0-5 (correctness, small-scale
training) but won't support the inference-optimization story this project
is centered on.

## Resume bullets this is designed to produce

**Superseded -- see `interview.md` Section 8 for the final, numbers-filled-in
bullets** (post scope-narrowing to KV caching + compression as the active
engine, with the rest kept-but-not-demoed). Original planning-stage bullets,
kept for history:

- "Implemented MHA, MQA, and GQA from scratch in a custom Transformer,
  quantifying the KV-cache memory/quality tradeoff across attention
  variants."
- "Implemented a continuous-batching request scheduler, improving
  throughput [X]x over static batching under concurrent load." (built +
  verified; kept in codebase, not part of the active demo)
- "Built chunked prefill and exact prefix caching for an encoder-decoder
  Transformer, correctly adapting both techniques (normally designed for
  causal decoder-only LLMs) to a bidirectional encoder without breaking
  exactness." (built + verified; kept in codebase, not part of the active
  demo)
- "Integrated Flash Attention via PyTorch SDPA into a from-scratch
  Transformer's decode path, verified numerically equivalent to the
  manual attention implementation." (built + verified; kept in codebase,
  not part of the active demo)
- "Applied post-training INT8 quantization and magnitude pruning,
  measuring real size/quality/latency tradeoffs (including where each
  technique does and doesn't transfer to GPU serving)." (active/demoed)
- "Compared 4 tokenization schemes (word, BPE, WordPiece, character) and
  3 positional encodings (sinusoidal, RoPE, ALiBi) across 12 trained
  model variants, evaluated on BLEU/chrF++/TER."

## Next step

Phase 0: scaffold `v3/` (directories above), port v1's data split +
word/BPE/char tokenizer modules in as a base. Nothing has been written
yet beyond this plan.
