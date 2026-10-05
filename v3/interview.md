# Neural Machine Translation — From-Scratch Transformer with LLM Inference Engine

**Resume-ready overview + interview prep.** Everything below is grounded in this
repo — `PROGRESS.md` and `results/manifest.json` are the receipts if anyone
asks to see it live.

---

## 1. Problem Statement

Build an English→French machine translation system, and — the actual center
of gravity of this project — a **from-scratch LLM inference-serving engine**
implementing the techniques production systems like vLLM use to serve large
language models efficiently: KV caching, continuous batching, chunked
prefill, prefix caching, Flash Attention, and post-training compression.

Two problems, stated precisely:

1. **Translation quality**: given an English sentence, produce a fluent,
   accurate French translation, using a Transformer encoder-decoder trained
   entirely from scratch (no pretrained weights, no HuggingFace `transformers`
   library) — including implementing and empirically comparing every major
   architectural choice (tokenization strategy, positional encoding scheme,
   attention variant) rather than assuming textbook defaults are correct for
   this data.
2. **Inference efficiency**: naive autoregressive decoding recomputes full
   self-attention over the entire generated prefix at every single output
   token — wasteful, and the exact bottleneck real LLM-serving systems are
   built to eliminate. Implement, from first principles, the techniques
   that solve this at production scale, and prove each one is either a
   **pure speed optimization** (bit-for-bit identical output to naive
   decoding) or, where that's not possible (Flash Attention), *numerically
   equivalent* — before trusting any benchmark number.

---

## 2. Motivation

- **Why build it from scratch instead of fine-tuning a pretrained model?**
  Fine-tuning `MarianMT` or calling the OpenAI API would produce a "working
  translator" in an afternoon, but would demonstrate nothing about how
  Transformers, attention, or LLM inference actually work internally — which
  is precisely the depth an interviewer is trying to probe for. Every
  component here (tokenizer, positional encoding, attention, decoding
  strategy, KV cache, chunked-prefill/prefix-cache scheduler) is
  hand-implemented in PyTorch.
- **Why translation as the task?** It's a clean, standard encoder-decoder
  problem with unambiguous, industry-standard quality metrics (BLEU, chrF++,
  TER) — which frees up the actual project time budget for the harder part:
  the inference engine, rather than debating how to evaluate output quality.
- **Why go deep on inference optimization specifically?** This is the part of
  the NMT/LLM stack that most student projects skip entirely (most stop at
  "the model translates a sentence"). It's also exactly the skill set behind
  real production LLM serving (vLLM, TensorRT-LLM, TGI) — so it's the single
  highest-leverage differentiator to invest project time in.

---

## 3. Technology Stack

**Language / framework:** Python, PyTorch (raw `nn.Module`s — no
`transformers`, no pretrained checkpoints)

**Core architecture:** Transformer encoder-decoder (Vaswani et al., *Attention
Is All You Need*), built layer-by-layer: multi-head self-attention,
cross-attention, position-wise feed-forward networks, residual connections +
LayerNorm, label-smoothed cross-entropy loss, Noam learning-rate schedule
(warmup + inverse-square-root decay), gradient clipping

**Tokenization:** Byte-Pair Encoding (BPE) via HuggingFace `tokenizers`
(Rust-backed, not the `transformers` model library), WordPiece, word-level,
character-level, plus a **joint/shared BPE vocabulary** across source and
target languages (the technique used by mBART/NLLB/M2M-100)

**Positional encoding:** Sinusoidal (original Transformer), RoPE (Rotary
Positional Embeddings — the scheme used in LLaMA/Mistral/Qwen), ALiBi
(Attention with Linear Biases — used in BLOOM/MPT)

**Attention variants:** Multi-Head Attention (MHA), Grouped-Query Attention
(GQA — Llama 2/Mistral's choice), Multi-Query Attention (MQA)

**Decoding strategies:** Greedy decoding, beam search (with tunable beam size
and length penalty), top-k sampling, top-p / nucleus sampling

**Evaluation:** `sacrebleu` — BLEU, chrF++, TER (corpus-level and
sentence-level)

**Inference-serving engine (the differentiator):** **active/demoed:** KV
cache, post-training compression (INT8 dynamic quantization, magnitude
pruning). **Also built and independently verified, kept in the
codebase, not part of the current live demo:** continuous batching
(dynamic request scheduling), chunked prefill (layer-chunked encoder
pass spread across scheduler ticks), prefix caching (exact
full-sentence encoder-output reuse), Flash Attention (via PyTorch's
fused `scaled_dot_product_attention`), speculative decoding, FastAPI-
based async serving layer

**NLP tooling:** spaCy (`en_core_web_sm`) for Named Entity Recognition

**Demo / visualization:** Streamlit (interactive live-translation UI,
architecture explorer, experiment-results dashboard), Plotly

**Infra:** multi-GPU cluster (8× NVIDIA A6000), tmux-based long-running job
management

---

## 4. Concepts Implemented (keyword bank)

Transformer · Self-Attention · Cross-Attention · Multi-Head Attention ·
Grouped-Query Attention · Multi-Query Attention · Scaled Dot-Product Attention
· Rotary Positional Embeddings (RoPE) · ALiBi · Sinusoidal Positional
Encoding · Byte-Pair Encoding (BPE) · WordPiece Tokenization · Shared/Joint
Vocabulary · Sequence-to-Sequence Learning · Encoder-Decoder Architecture ·
Autoregressive Decoding · Greedy Decoding · Beam Search · Top-k Sampling ·
Nucleus (Top-p) Sampling · Label Smoothing · Noam Learning Rate Schedule ·
Gradient Clipping · Teacher Forcing · BLEU Score · chrF++ · Translation Edit
Rate (TER) · KV Cache · Continuous Batching · Dynamic Batch Scheduling ·
Chunked Prefill · Prefix Caching · Flash Attention · Scaled Dot-Product Attention
(SDPA) · Post-Training Quantization (INT8, Dynamic Quantization) ·
Magnitude Pruning / Model Sparsity · Time to First Token (TTFT) · Time
Between Tokens (TBT) / Inter-Token Latency · Throughput vs. Latency
Tradeoff · LLM Inference Optimization · Model Serving · Memory
Fragmentation · Named Entity Recognition (NER) · Data Augmentation ·
Noise-Robust Training · Bidirectional Translation · Multilingual NMT
(target-tag trick, Johnson et al. 2017) · Ablation Study · Hyperparameter
Tuning · Distributed Training infrastructure

---

## 5. Step-by-Step: How the Project Was Built

### Step 1 — Baseline architecture attempt, and a real failure
Started with a "textbook" Transformer-base-scale config (`emb_dim=512,
num_layers=12`, ~100M params) — standard sizing for corpora with millions of
sentence pairs. **12 of 14 tokenizer/positional-encoding combinations
collapsed**: trained normally for one epoch, then degenerated into repeating
a single token for the rest of training (BLEU 0.00).

**Root-caused, not guessed at**: cross-referenced an earlier version of this
project that hit the identical failure on the identical data with the
identical oversized config — pointing straight at **capacity/dataset-size
mismatch** (~100M params on only 106K training pairs), not a code bug (RoPE
math, ALiBi construction, masking, and attention head-repeat logic were all
independently verified correct first). Fixed by shrinking to `emb_dim=128,
num_layers=6` (~10.6M params) — matching the scale actually supported by this
data.

### Step 2 — Clean comparison sweep: tokenizer × positional encoding
With capacity fixed, ran a controlled 3×2 sweep (BPE/word/char ×
sinusoidal/RoPE), all else held constant. **RoPE beat sinusoidal in every
tokenizer pairing, no exceptions.**

One combination — **char + sinusoidal** — still failed in a much subtler way:
healthy 69% teacher-forced token accuracy, but free-running generation
produced the *identical* French sentence regardless of input. Measured, not
assumed, the mechanism: encoder output pairwise cosine similarity across
positions was **1.0000** (full representational collapse) for sinusoidal vs.
**0.013** for RoPE on the identical architecture — because sinusoidal PE is
injected once at the embedding layer and never reinforced, so it has to
survive 6 rounds of self-attention (an averaging operation) purely through
the residual stream; RoPE re-derives relative position fresh at every
attention layer, so it can't wash out the same way.

### Step 3 — Completed the breadth matrix (WordPiece, ALiBi) and swept decoding strategies
Retrained WordPiece and ALiBi variants at the corrected capacity (both had
only been tested under the broken oversized config in Step 1). Then compared
greedy / beam search / top-k / top-p decoding on the leading checkpoint.

### Step 4 — Two rounds of architecture/hyperparameter tuning
Empirically tuned depth, embedding width, and feed-forward ratio rather than
guessing. Finding: depth consistently hurt at this data scale (106K pairs
isn't Transformer-base-scale data); reallocating the feed-forward ratio from
a non-standard 16× down to the textbook 4× — freeing capacity for a wider
embedding — plus doubling the training budget, compounded to the best result
across 14 configs.

### Step 5 — Attention-type comparison: MHA vs. GQA vs. MQA
Retrained all three attention types at matched capacity and epoch budget
(the earlier MQA run had only been tested under Step 1's broken config).
Measured the real tradeoff: MHA→GQA trades a **4× KV-cache-size reduction**
for only −0.2 to −0.4 BLEU; GQA→MQA trades a further 2× reduction for
−0.7 to −0.85 BLEU — roughly double the quality cost for half the
compression gain, the textbook argument for why production LLMs (Llama 2 70B,
Mistral) settled on GQA rather than MQA.

### Step 6 — Joint vocabulary + noise-augmentation experiments
Tried a **shared BPE vocabulary** across English and French (the mBART/
NLLB/M2M-100 technique) to address proper-noun/unseen-word mistranslation.
Result: improved overall quality (+1.15 BLEU) by letting rare shared surface
forms (cognates, numbers, punctuation) share statistical signal — but did
**not** fix the specific proper-noun problem it targeted (honestly reported,
not glossed over: shared vocabulary doesn't imply shared embedding weights
unless explicitly tied).

### Step 7 — NER + entity-masking pipeline (the production-pragmatic fix)
Since the architectural fix (Step 6) didn't solve proper-noun preservation,
built an explicit engineering solution instead: detect a PERSON entity with
spaCy → mask it with a placeholder the model has *proven* it reproduces
reliably → translate → restore the original text afterward. No retraining
needed. Found and fixed several real bugs along the way (NER false positives
on this corpus's space-split contractions, restoration collisions when
multiple entities reuse the same placeholder pool, spaCy's inconsistent
lowercase-name detection).

### Step 8 — Bidirectional translation, one model
Added English→French **and** French→English translation without training
two separate models, using the target-tag trick (Johnson et al. 2017,
Google's original multilingual NMT paper): one shared vocabulary, a
`tofr`/`toen` tag prepended to the source telling the single model which
direction to translate.

### Step 9 — Phase 7: the from-scratch LLM inference engine
See the dedicated deep-dive below — this is the part worth spending the most
interview time on.

### Step 10 — Live demo
Built a Streamlit app: pick any of the ~25 trained architecture variants,
compare two side by side, watch cross-attention heatmaps, try the entity
pipeline live, see real decode latency.

---

## 6. What Makes This Different From a Beginner NMT Project

Be ready to say this directly if asked, or if you sense the interviewer is
mentally filing this as "another seq2seq tutorial project":

> "A typical student NMT project fine-tunes a pretrained HuggingFace model,
> or follows the standard PyTorch seq2seq-with-attention tutorial, and stops
> once it translates a sentence. This project has two things that go well
> past that bar. First, every architectural choice — tokenizer, positional
> encoding, attention type — was independently implemented from scratch and
> empirically compared, including finding and root-causing two real training
> collapses with measured evidence, not guesswork. Second, and this is the
> part I'd point to as the actual differentiator: I built a working
> inference-serving engine — KV caching, continuous batching, chunked
> prefill, prefix caching, Flash Attention, and post-training compression
> — each one correctness-verified (bit-for-bit
> exact where the technique allows it, numerically verified where it
> doesn't) against a naive baseline before I trusted a single benchmark
> number, and each one measured on real LLM-serving metrics — time to
> first token, inter-token latency, throughput — not just a single
> latency number. That's the part of the LLM stack most people never
> touch, because it's systems engineering, not just model training."

If you only remember one line to say when someone's eyes start to glaze over
at "yet another translation project": **"the model is the least interesting
part — the interesting part is that I built the serving engine underneath
it."**

---

## 7. Phase 7 Deep-Dive — build real understanding here, not just talking points

**Final scope, read this first:** the project's *actively shown* inference
engine is just **KV caching + compression (quantization/pruning)** — 7a/7b
and the Compression subsection below. Everything else on this page
(PagedAttention, chunked prefill, prefix caching, Flash Attention,
continuous batching, speculative decoding) was fully built and
independently verified, then deliberately excluded from the live demo and
the headline comparison table — kept in the codebase, not deleted, and
each reproducible via its own `benchmark_optimizations.py` flag (see
`PROGRESS.md`'s "Final scope" section). **Interview strategy:** know 7a/7b
and Compression cold — that's what the demo shows and what you'll be asked
to walk through live. For the rest, one or two sentences ("I also built and
verified X, here's the one interesting thing I learned, but it's not part
of the core demo") is the right depth unless the interviewer explicitly
digs in — in which case the material below is still there and still
accurate.

### 7a/7b — The baseline problem, and KV caching

**The problem naive decoding has:** an autoregressive Transformer generates
one token at a time. To predict token *n*, self-attention needs to look at
tokens *1...n-1*. The naive way to do this is to **recompute self-attention
over the entire generated sequence so far, every single step** — so
generating a 20-token sentence means the 20th step redundantly recomputes
attention for tokens 1-19 all over again, even though their K (key) and V
(value) projections never change once computed. This is O(n²) work for an
n-token output.

**The fix:** cache each token's K/V projections the first time they're
computed, and on every subsequent step only compute Q/K/V for the *one new
token*, then attend against the cached K/V from all previous steps plus the
new one. Turns the per-step cost from O(n) into O(1) (amortized), the total
generation cost from O(n²) into O(n).

**How I verified it's correct, not just faster:** ran it against the naive
version on 200 test sentences and required **bit-for-bit identical
output** — since this is a pure caching optimization, not an approximation,
any mismatch would mean a real bug. Zero mismatches.

**Honest result:** only ~1.2-1.4× wall-clock speedup at this project's scale
(short sequences, a 10.85M-parameter model) — Python/kernel-launch overhead
dominates the savings at this size. This is expected, not a shortfall: the
O(n²)→O(n) improvement is a much bigger win as sequence length and model size
grow, which is exactly why it matters for real LLMs generating hundreds of
tokens.

### 7c — PagedAttention: built, verified, then deliberately removed

**Worth knowing even though it's not in the current build — this shows
scope judgment, not just implementation ability.** PagedAttention (Kwon
et al. 2023, the vLLM mechanism) solves KV-cache *memory fragmentation*:
the naive approach reserves a contiguous memory block per request sized
for the worst-case output length, wasting most of it whenever the actual
output is shorter. The fix, borrowed from OS virtual memory paging: split
GPU memory into small fixed-size blocks from one shared pool; each
request gets a block table mapping its logical sequence to (possibly
non-contiguous) physical blocks, allocated on demand. I built this,
verified it bit-identical to a naive baseline on 100/100 sentences, and
measured a real result — fragmentation dropped from ~83% to ~23%,
~4.6× more concurrent requests at the same memory budget. I later removed
it: at this project's actual demonstrated scale (tens of concurrent
requests, short MT outputs, not the thousands of long-running chat
sessions PagedAttention is built for), it was the least-visible win of
the original four Phase 7 techniques in my own numbers — so I swapped it
for four more directly comparable, broader techniques (chunked prefill,
prefix caching, Flash Attention, compression) that keep the story
anchored to what actually shows up at this project's scale, and that are
individually strong, common interview topics in their own right. The
mechanism worked; the decision to remove it was a scope call, and I can
walk through the tradeoff either way if asked.

### Chunked prefill: adapting a causal-LLM technique to a bidirectional encoder

**The problem in real serving:** a newly-admitted request's prefill (its
whole prompt, processed before generation starts) normally runs as one
atomic forward pass, which blocks every other in-flight request's decode
step for that whole call if the prompt is long — a real fairness/latency
problem in production LLM serving (vLLM, Sarathi-Serve built chunked
prefill specifically to fix this).

**The genuinely interesting architectural wrinkle, worth stating
directly if asked "how does this work":** the standard technique chunks
a prompt by *sequence position* — chunk 1 attends only to itself, chunk 2
attends to chunk 1 + itself, and so on — which is safe specifically
because a causal decoder-only LLM's attention already only lets a token
see *earlier* tokens. This project's **encoder is bidirectional** — every
source token legitimately depends on every *other* source token,
including later ones. Position-chunking a bidirectional encoder the
causal way would silently change what it computes — a genuinely
*different*, wrong result, not a pure speed optimization. I chunk by
**layer** instead: run a fixed number of encoder layers per scheduler
tick, carry the partial hidden state to the next tick. That preserves the
encoder's exact bidirectional computation (verified bit-identical across
every chunk size tested, `max_diff=0.0`) while still spreading its
wall-clock cost across ticks — the actual scheduling property chunked
prefill exists to provide.

### Prefix caching: the same causal-vs-bidirectional issue, from a different angle

**The problem:** repeated identical requests (a real pattern — FAQ-style
phrases, common short queries) redundantly re-run the entire encoder
every time.

**Why naive "cache a shared prefix across requests" (vLLM/SGLang's
automatic prefix caching) isn't safe to port directly here:** that
technique is exact for causal decoder-only LLMs specifically because a
token's representation only depends on itself and earlier tokens, so two
prompts sharing a leading phrase produce byte-identical KV state for that
shared part regardless of what follows. With a bidirectional encoder,
two sentences sharing a leading phrase ("tom said hello" / "tom said
goodbye") do *not* produce identical encoder states for "tom said",
because "hello" vs. "goodbye" bidirectionally influences it. Naively
porting partial-prefix caching here would be a silent approximation, not
exact.

**What I actually built, and why it's exact:** caching keyed by the
*full, exact* source sentence — the safe, degenerate case of prefix
caching, since identical input always produces identical output
regardless of bidirectionality. Verified bit-identical including genuine
concurrent cache hits (repeated sentences interleaved with unique ones,
all processed together); measured **1.10× wall-clock** on a realistic
repeated-query workload.

### Flash Attention: using a library, and knowing why it's not bit-identical

**What "using a library instead of hand-rolling it" means precisely:**
Flash Attention (Dao et al. 2022) is a fused, tiled, IO-aware CUDA
kernel that never materializes the full attention matrix in GPU memory —
reimplementing that by hand in pure PyTorch wouldn't actually be faster,
just a slower, less careful re-derivation of what a real kernel already
does. I call PyTorch's built-in `torch.nn.functional.scaled_dot_product_attention`
(SDPA) instead, which PyTorch itself dispatches to a real
FlashAttention-2 CUDA kernel automatically on Ampere+ GPUs, with zero
custom kernel code.

**Why this is the one Phase 7 technique that's *not* bit-identical to the
manual path, and why that's correct, not a bug:** SDPA computes the exact
same math (`softmax(QK^T/√d)V`) via a different algorithm — tiled, with
an online/running softmax. Floating-point addition isn't associative, so
summing in a different order produces tiny rounding differences. Measured
directly: max absolute logit difference **1.9×10⁻⁵**, and — the part that
actually matters — every argmax decision matched exactly, so the final
decoded text is identical on 40/40 test sentences even though the raw
numbers aren't bit-exact.

### Compression: quantization and pruning, measured honestly

**Quantization** (PyTorch's post-training dynamic INT8 quantization over
every Linear layer): **3.52× smaller model** (41.4MB → 11.8MB). The
honest scope limit, stated directly rather than found by a confusing
benchmark: PyTorch's stock quantized INT8 kernels are CPU-only — no CUDA
implementation — so this never touches the GPU serving path. Still a
real, useful technique (smaller weights to load/transfer regardless of
where the model runs); knowing *why* it doesn't transfer to GPU serving
is the more useful thing to be able to say than a blindly-benchmarked
number would be.

**Pruning** (magnitude-based unstructured pruning at 30/50/70% sparsity):
a prediction stated up front, then actually measured, not assumed —
unstructured sparsity (zeros scattered through an otherwise-dense
tensor) only pays off with a *sparse* tensor format and a sparse-aware
kernel; a dense tensor with zeros in it still occupies the same memory
and still gets multiplied elementwise by a standard GEMM kernel. Measured
directly: sparsity hit the exact target (30.00%/50.00%/70.00%) while
model size stayed byte-for-byte unchanged regardless of sparsity level —
confirming the prediction exactly, a clean example of understanding
*why* a technique won't help before ever running a benchmark, not
discovering it by surprise.

### 7d — Continuous batching: fixing the *scheduling*, not the memory

**The problem:** even with efficient memory, a naive scheduler processes
requests in **static batches** — it waits for the *entire* batch to finish
before admitting new requests. If one request in a batch of 8 needs 100
tokens and the rest need only 10, the 7 short ones sit idle (their GPU slot
wasted) for the full 100 steps, and new arrivals wait in a queue the whole
time even if slots are technically free.

**The fix (the Orca/vLLM scheduling innovation):** check *every single decode
step* whether any request in the batch has finished — if so, evict it
immediately and admit the next waiting request into that freed slot right
away, instead of waiting for the whole batch's slowest member.

**Measured result:** P50 latency dropped ~1.9× under a simulated stream of
staggered request arrivals, purely from the scheduling change (same memory
backend, same model).

### 7e — Speculative decoding: built and verified, not part of the current showcase

**Worth a one-line mention, not a deep-dive, unless specifically asked:**
also implemented — a small draft model (1.78M params, 16% of the target's
size, same shared vocabulary) proposes several tokens ahead, the target
model verifies all of them in one parallel forward pass, and because this
project's decoding is greedy (deterministic), the accept rule is exact
equality, making the whole thing bit-for-bit exact rather than merely
distributionally equivalent — a real, correct, fully-verified (200/200
bit-identical) piece of work. Measured up to 3.1× fewer target-model
calls, though wall-clock speedup stayed roughly flat at this project's
small scale (the draft model's own overhead cancels the savings — expected,
since speculative decoding's real payoff needs a much larger, more
expensive target model). Deliberately not part of the current live demo
or the main comparison study — a scope choice to keep the showcased
project tighter, not a failure or a deletion; the code is intact and
this is a fine thing to bring up if an interviewer asks "what else did
you build."

### 7f — Assembling the engine

Combined chunked prefill, prefix caching, and continuous batching into
one scheduler behind a request-in/translation-out API, with a thin
FastAPI layer on top — verified 20/20 bit-identical across every
combination of the three running together (including multiple requests
genuinely interleaved mid-prefill and mid-generation at once), confirming
the underlying request/backend abstraction holds up under composition,
not just one technique at a time.

---

## 8. Resume Bullet Points (ready to use)

- Built a from-scratch Transformer encoder-decoder (PyTorch, no pretrained
  weights) for English↔French NMT; ran a controlled 14+ configuration sweep
  across tokenization (BPE/WordPiece/word/char), positional encoding
  (RoPE/ALiBi/sinusoidal), and attention (MHA/GQA/MQA), root-causing two
  real training collapses via measured evidence (encoder representational
  collapse, capacity/data-scale mismatch) rather than trial-and-error.
- Implemented a from-scratch LLM inference engine centered on KV caching
  and post-training compression (INT8 dynamic quantization, magnitude
  pruning), measuring TTFT, inter-token latency, end-to-end latency, and
  throughput on real serving metrics; verified every optimization exact
  (bit-identical against a naive baseline) before trusting a benchmark
  number, and honestly reported where a technique didn't pay off at this
  project's scale (e.g. quantization is CPU-only and ~5.3 min/sentence
  slower there; unstructured pruning changes nothing about size/latency
  without a sparse kernel, but shows a sharp 30%→50% BLEU quality cliff).
- Also built and independently verified (bit-identical or numerically
  equivalent, kept in the codebase but scoped out of the active demo)
  continuous batching, chunked prefill, prefix caching, Flash Attention
  (PyTorch SDPA), and speculative decoding — including correctly
  adapting two causal-decoder-only LLM serving techniques (chunked
  prefill, prefix caching) to a bidirectional Transformer encoder by
  identifying and avoiding the exactness bug a naive port would
  introduce (position-chunking / partial-prefix caching both silently
  change output on non-causal attention).
- Designed and shipped a production-pragmatic NER + entity-masking pipeline
  (spaCy) to fix proper-noun mistranslation without retraining, after an
  architectural fix (joint BPE vocabulary) improved overall BLEU by 1.15 but
  didn't solve the targeted problem — an honestly-reported negative result
  that informed the final design choice.
- Built an interactive Streamlit demo exposing 25+ trained model variants,
  live translation with attention visualization, and side-by-side
  architecture comparison.

---

## 9. Interview Q&A

### Getting started

**Q: Walk me through this project.**
A: See Section 5 above, compressed to 30-60 seconds: "I built an
English-to-French Transformer from scratch, then spent most of the
project time building an LLM inference engine underneath it, centered on
KV caching and post-training compression — quantization and pruning —
with every optimization verified exact against a naive baseline before I
trusted a benchmark number, measured on real serving metrics like
time-to-first-token and throughput. I also built and verified continuous
batching, chunked prefill, prefix caching, Flash Attention, and
speculative decoding along the way — they're in the codebase and I can
go deep on any of them, but I scoped the live demo down to the two I
just mentioned so I can defend everything I show without hedging."

**Q: Why translation? Why not a chatbot / summarizer / something else?**
A: Translation is a clean, standard encoder-decoder task with unambiguous
metrics (BLEU/chrF++/TER), which let me spend the real project budget on the
inference engine rather than on evaluation methodology debates.

**Q: What was the hardest part?**
A: Two things, both explained in Section 5: root-causing the two training
collapses (Steps 1 and 2) with actual measured evidence rather than guessing
hyperparameters, and correctly adapting chunked prefill and prefix caching
— both originally designed for causal decoder-only LLMs — to this
project's bidirectional encoder, where the naive port of either technique
would have silently produced wrong output rather than just being slow.
Catching that *before* implementing, not after a benchmark looked wrong,
was the actual hard part.

### Architecture fundamentals

**Q: Why does RoPE outperform sinusoidal positional encoding?**
A: Sinusoidal PE is added once at the embedding layer and has to survive N
layers of self-attention (fundamentally an averaging operation) purely
through the residual stream. RoPE instead re-derives *relative* position
fresh at every layer's Q/K dot product, so it can't be washed out by depth or
sequence length — measured directly: encoder-output cosine similarity across
positions was 1.0000 (fully collapsed) for sinusoidal vs. 0.013 for RoPE on
an identical architecture.

**Q: What's the difference between MHA, GQA, and MQA, and why does it
matter for serving?**
A: MHA gives every attention head its own K/V projections. MQA shares one
K/V projection across *all* heads (queries stay per-head). GQA is the middle
ground — heads are split into groups, each group shares one K/V projection.
The practical reason this matters: KV-cache memory scales linearly with the
number of *KV heads*, not query heads — so MQA gives an 8× smaller cache,
GQA a 4× smaller cache, versus full MHA, at a real but much smaller quality
cost (measured: −0.7 to −0.85 BLEU for MQA, −0.2 to −0.4 for GQA). This is
exactly why Llama 2 70B and Mistral use GQA — it's near the knee of the
cache-size-vs-quality curve.

**Q: What's the actual formula for KV-cache size per token?**
A: `bytes/token = 2 × num_layers × num_kv_heads × head_dim × dtype_bytes`
(the 2 is for K and V separately). Only `num_kv_heads` changes between
MHA/GQA/MQA — everything else is fixed by the architecture.

**Q: Why greedy decoding instead of beam search as the default?**
A: Measured, not assumed: at this model's scale, a 150-sentence
sentence-level BLEU comparison showed the smaller/earlier model's apparent
"wins" over the tuned model were concentrated in rote memorization of short
template sentences (exact reference match), not genuine translation quality
— consistent with the model's output distribution not being sharp enough for
beam search's wider exploration to help. All four strategies are still
implemented and selectable; greedy is the empirically-justified default, not
a shortcut.

### Phase 7 — expect the most questions here

**Scope reminder:** the live demo and headline comparison table only show
KV caching and compression — know those two cold. The rest of this
subsection (PagedAttention, chunked prefill, prefix caching, Flash
Attention, continuous batching, speculative decoding) is real, verified,
kept-in-codebase work worth being able to discuss if asked, but it's
"also built," not "what I'll show you."

**Q: Explain KV caching like I don't know what it is.**
A: When generating text one token at a time, each new token's prediction
needs to "look back" at every previous token via attention. The naive way
recomputes that look-back from scratch at every step — step 20 redoes the
work step 19 already did, for tokens 1 through 19. KV caching just remembers
each token's key/value vectors the first time they're computed, so every
step only does the work for the *one new token*. It's a caching optimization,
not an approximation — I verified it against the uncached version and it
produces the exact same output, every time, on 200 test sentences.

**Q: What's PagedAttention, and why isn't it in your engine anymore?**
A: It's KV-cache memory management borrowed from OS virtual memory
paging. Before it, the standard approach reserves one big contiguous
memory block per request, sized for the worst-case output length — so a
request that ends up short still holds onto memory it never used, and
that waste compounds across many concurrent requests. I built it: memory
split into small fixed-size blocks from one shared pool, each request
getting a block table mapping logical positions to (possibly
non-contiguous) physical blocks, allocated on demand and returned to the
pool the instant a request finishes. Measured fragmentation drop from
~83% to ~23%, ~4.6× more concurrent requests at the same memory budget —
a real, working result. I removed it afterward as a scope decision: at
this project's actual scale (tens of concurrent requests, short MT
outputs), it was the least-visible win of the original four Phase 7
techniques, so I traded it for four more broadly useful, individually
strong techniques — chunked prefill, prefix caching, Flash Attention,
and compression — that together broaden what I can speak to in depth
rather than deepen one memory-management mechanism further.

**Q: How does chunked prefill work when your encoder isn't causal?**
A: The standard technique (from causal decoder-only LLM serving) chunks
a prompt by sequence position — safe there because a token can only ever
attend to earlier tokens anyway. My encoder is bidirectional: every
token depends on every other token, including later ones. Position-
chunking it the causal way would change what it computes — a different,
wrong result. So I chunk by *layer* instead: run a fixed number of
encoder layers per scheduler tick, carry the partial state forward. That
keeps the encoder's output exactly what a single one-shot call would
produce (verified bit-identical across every chunk size I tested) while
still spreading its cost across ticks so it doesn't block other
requests' decode steps.

**Q: Same question for prefix caching — how is that different from what
vLLM does?**
A: vLLM's automatic prefix caching reuses KV state for a shared *partial*
prompt prefix across different requests, and that's exact for causal
attention specifically. My encoder is bidirectional, so two sentences
sharing a leading phrase don't actually produce identical encoder states
for that shared part — whatever comes after still influences it. Porting
partial-prefix caching directly would be a silent approximation. What I
built instead is exact: cache keyed by the *entire* source sentence, so
a repeated identical request skips the encoder pass entirely. Less
general than vLLM's version, but provably correct for this architecture,
and it's still a real, useful pattern — a translation service sees a lot
of repeated common queries.

**Q: You used PyTorch's SDPA for Flash Attention instead of writing your
own kernel — isn't that against your "from scratch" approach?**
A: For everything else in this project, yes, from scratch is the point.
Flash Attention specifically needs a fused, tiled CUDA kernel to get its
actual benefit (avoiding materializing the full attention matrix in
memory) — reimplementing that in plain PyTorch wouldn't give me a real
Flash Attention kernel, it'd give me a slower, less numerically careful
copy of ordinary attention. Using the library *is* the correct
engineering decision here, and knowing when to reach for an existing,
battle-tested kernel instead of re-deriving one badly is itself part of
understanding the technique, not a shortcut around it.

**Q: Why isn't your Flash Attention output bit-identical to your manual
attention, when everything else in Phase 7 is?**
A: Because it's a genuinely different algorithm computing the same math,
not just a different schedule for the same arithmetic. Flash Attention
tiles the computation and uses a running softmax so it never holds the
full attention matrix in memory at once; floating-point addition isn't
associative, so summing in a different order gives tiny rounding
differences. I measured it directly — max logit difference of about
2×10⁻⁵ — and confirmed every argmax decision still matched, so the final
translated text is identical even though the raw numbers aren't
bit-exact. KV caching, continuous batching, chunked prefill, and prefix
caching are all different in kind — they only change what gets computed,
cached, or scheduled, never the actual
arithmetic — which is why those are held to bit-identical, and Flash
Attention is the one held to "numerically equivalent" instead.

**Q: What did you find with quantization and pruning?**
A: Quantization (PyTorch's dynamic INT8) gave a genuine 3.52× smaller
model — but it's CPU-only; PyTorch's quantized kernels have no CUDA
implementation, so it never touches my actual GPU serving path. That's
worth saying plainly rather than glossing over: it's still useful for
memory/transfer size regardless of where the model runs, but it's not a
GPU speed technique here. Pruning was the more interesting negative
result: I predicted *before* measuring that unstructured pruning
(zeroing individual weights without a sparse tensor format) wouldn't
change model size or speed, since a dense tensor with zeros in it still
occupies the same memory and still runs through the same dense GEMM
kernel. Measured it directly at 30/50/70% sparsity — exactly as
predicted, size stayed byte-for-byte identical regardless of sparsity.
Knowing *why* before running the benchmark, and having the benchmark
confirm it, is a better answer than either skipping the measurement or
being surprised by it.

**Q: What's continuous batching, and how is it different from just
batching requests together?**
A: Regular ("static") batching still waits for the *whole* batch to finish
before accepting new requests — so a fast request sits idle wasting its slot
until the slowest one in its batch is done. Continuous batching checks every
single decode step for finished requests, evicts them immediately, and
slots in the next waiting request right away. Same underlying model, same
memory backend — purely a scheduling change — and it dropped P50 latency
about 1.9× in my staggered-arrival test.

**Q: Did you look at speculative decoding at all?**
A: Yes — built and fully verified (200/200 bit-identical; because this
project's decoding is greedy/deterministic, the accept rule is exact
equality against the target model's own argmax, which makes it
bit-for-bit exact rather than merely distributionally equivalent, a
stronger guarantee than the general stochastic-sampling case in the
literature). Measured up to 3.1× fewer target-model calls, though
wall-clock speedup stayed flat at this project's small scale since the
draft model's own overhead cancels the savings — its real payoff needs a
much larger, more expensive target model. It's a deliberate scope
decision to keep it out of the current demo and comparison study, not a
failure or something I couldn't get working — happy to go deeper on it
if useful.

**Q: How did you verify any of Phase 7 was actually correct, not just
"probably faster"?**
A: Held every single optimization to the same bar: run it against the
naive/uncached baseline on a batch of real test sentences and require
bit-for-bit identical token-level output (or, for Flash Attention, the
weaker but still strict bar of numerically-equivalent logits and
identical argmax decisions). Not "looks similar," not "BLEU is close" —
exact match, every token. That standard is what caught a real bug
integrating Flash Attention: PyTorch's SDPA rejected my attention mask
with "last dimension must be contiguous" — the mask's last dimension was
a broadcast placeholder (size 1, stride 0) that the manual attention
path happily broadcasts through `masked_fill`, but SDPA's fused kernel
needs a real, materialized dimension matching the actual key length.
Fixed by explicitly expanding the mask to its full shape before making it
contiguous — invisible until I actually ran the correctness check against
the manual baseline, not something I'd have caught by eyeballing the
diff.

**Q: If you had more time, what would you add to Phase 7?**
A: Scale the whole engine to a genuinely large target model, where Flash
Attention's and continuous batching's wall-clock wins should show up far
more strongly than at this project's small scale (their benefit scales
with sequence length and model size, both of which this project
deliberately kept modest); batch the plain decode path itself into one
padded tensor op across requests of different lengths instead of
looping per request each scheduling tick; and extend prefix caching from
exact full-sentence matching to genuine partial-prefix reuse for any
future decoder-only extension of this work, where it would be exact
rather than the approximation it would be for this project's
bidirectional encoder.

### Scope and honesty questions

**Q: This seems like a lot for one project — is all of it really yours /
did you really understand all of it?**
A: Everything in `PROGRESS.md` has a matching benchmark number in
`results/manifest.json`, and I can walk through the mechanism of any piece
of it in detail, which is what Section 7 above is for. I deliberately scoped
to three focus areas instead of the 6-7 shallow projects some candidates list
— I'd rather defend three things deeply than be unable to answer a follow-up
on any of seven.

**Q: What would you NOT put on your resume from this project, or what's
still incomplete?**
A: Some pieces (ALiBi's full sweep, the bidirectional model, the standalone
draft model) are built, benchmarked, and documented in `PROGRESS.md`, but
deliberately trimmed out of the live demo's navigation to keep a five-minute
walkthrough focused on the core story — worth saying plainly if asked,
rather than implying the demo shows everything that exists.
