# NMT Project — Interview Prep

A from-scratch English→French neural machine translation system: every
component (tokenizers, positional encodings, attention variants, decoding
strategies, and a full inference-serving engine with KV cache /
PagedAttention / continuous batching / speculative decoding) implemented
by hand in PyTorch — no HuggingFace `transformers`, no pretrained
checkpoints. Numbers below are pulled directly from `results/manifest.json`
and `PROGRESS.md`; you can `cat` either file live in an interview if asked
to back anything up.

---

## 1. "What's the project?" (30-second version)

> "I built an English-to-French translation system from scratch — a
> Transformer encoder-decoder in raw PyTorch, no pretrained weights. The
> project has two halves. First, a controlled experiment comparing
> tokenization strategies (BPE, WordPiece, word, char), positional
> encodings (sinusoidal, RoPE, ALiBi), and attention variants (MHA, GQA,
> MQA) — about 30 trained checkpoints, benchmarked on BLEU/chrF++/TER, to
> find the best config empirically instead of guessing. Second, and this
> is the part I actually care about most: a from-scratch LLM inference
> engine — KV caching, PagedAttention, continuous batching, and
> speculative decoding, the same techniques vLLM and TGI use in
> production — built and correctness-verified bit-for-bit against a naive
> baseline before I trusted any speed number."

**If they ask "why translation specifically":** it's a clean, well-scoped
domain to build a *real* encoder-decoder Transformer end-to-end (not just
a decoder-only toy), with an unambiguous, standard way to measure quality
(BLEU/chrF++/TER) — which freed me up to spend most of my actual effort on
the harder, more interesting part: the inference engine.

**One-line summary of the winning config:** BPE tokenizer with a joint
8,000-token vocabulary shared between English and French, RoPE positional
encoding, standard multi-head attention, 10.85M parameters, greedy
decoding. BLEU 43.75 / chrF++ 63.12 / TER 37.53 on the held-out test set
(`bpe_joint_rope_mha` in `results/manifest.json`).

---

## 2. Problems faced, and how I resolved them

Lead with these — they're the actual substance of the project. Each one
follows the same shape: **observed a wrong result → didn't guess, measured
→ found the real mechanism → fixed it → verified the fix.** That loop,
repeated, is the thing to sell in an interview, more than any single bug.

### 2.1 The first training run collapsed — 12 of 14 variants

**What happened:** first pass used a Transformer-base-scale config
(`emb_dim=512, num_layers=12`, ~100M params) — the textbook size for a
"real" Transformer. 12 of 14 tokenizer×PE combinations trained normally
for one epoch, then collapsed into repeating a single high-frequency
token for the rest of training. BLEU 0.00 across the board.

**How I found the cause:** ruled out code bugs first, systematically —
checked RoPE's rotation math, ALiBi's slope construction, sinusoidal PE
construction, encoder/decoder masking, and MHA/MQA/GQA head-repeat
generalization, all independently, all correct. Then cross-referenced an
earlier version of this same project (v1) that had hit the *identical*
failure on the *identical* data with the *identical* oversized config
months earlier — while a 6.6×-smaller config had succeeded on that same
v1 codebase. That comparison pointed straight at capacity, not code.

**Root cause:** classic overcapacity-on-small-data collapse — a
100M-parameter model on ~106K training pairs. Only the char-tokenizer
variants survived, purely because char's tiny embedding table kept
effective capacity in check by accident, not because char is more robust.

**Fix:** shrank to `emb_dim=128, num_layers=6` (~10.6M params), matching
the scale already proven to work on this data. Every variant trained
cleanly afterward.

**Why this is a good interview story:** it shows I don't treat "the model
learned to output garbage" as a training-hyperparameter problem to
randomly poke at — I diagnosed it as a capacity/data mismatch with actual
cross-run evidence before touching anything.

### 2.2 A more subtle collapse: char + sinusoidal, and *why*

Even after fixing the capacity issue, one specific combination —
character tokenization with sinusoidal positional encoding — still failed
in a much subtler way: healthy 69% held-out token accuracy under teacher
forcing, but at free-running generation time it output the **identical
French sentence for every input**, regardless of what was translated.

**Diagnosis, root-caused, not just observed:**
- Cross-attention entropy was exactly uniform at every decode step, for
  every input — the decoder was completely ignoring the source.
- Encoder output pairwise cosine similarity across positions: **1.0000**
  for sinusoidal (full rank-1 collapse) vs. **0.013** for RoPE on the
  identical architecture.
- Sinusoidal PE's row-norm was **5.22×** the token embedding's row-norm at
  the input layer — position information was dominating content from the
  very first layer.
- **Mechanism:** sinusoidal PE is injected once at the embedding layer and
  never reinforced — self-attention layers get `pe_type="none"` — so that
  one-time signal has to survive 6 rounds of self-attention + LayerNorm
  purely through the residual stream. Stacked self-attention is
  fundamentally an *averaging* operator; char's 128-token sequences (2× bpe/
  word's 64) give it twice the room to wash the signal out. RoPE instead
  re-derives *relative* position fresh at every layer's Q/K dot product,
  so it can't wash out the same way.

**Why this is a good interview story:** it's a real, measured,
mechanistic finding about *why* a specific architectural choice fails
under specific conditions — not "I tried X and it didn't work," but "I
measured exactly where and why it broke, with three independent pieces of
evidence pointing at the same mechanism."

### 2.3 Proper nouns and unseen words get mistranslated

**Problem:** OOV/unfamiliar proper nouns (a name the model never saw in
training) get hallucinated into something unrelated rather than copied
through — a well-known NMT failure mode, but I wanted to actually measure
and try to fix it rather than just note it.

**What I tried #1 — joint BPE vocabulary:** switched from two
independently-trained English/French BPE vocabularies to one shared
vocabulary over the bilingual corpus (the same technique mBART/NLLB/
M2M-100 use). This **did** improve overall translation quality (+1.15
BLEU) by letting rare shared surface forms and cross-lingual structure
(cognates, numbers, punctuation) share statistical signal — but it did
**not** fix the specific proper-noun problem it was aimed at (still only
2/6 fabricated names preserved in a stress test, same as before).
Explained honestly: shared *vocabulary* doesn't imply shared *embedding
weights* between the encoder and decoder unless you explicitly tie them,
which this didn't do.

**What I tried #2, that actually worked — NER + masking pipeline:**
production-pragmatic answer instead of an architectural one: detect a
PERSON entity with spaCy → mask it with a placeholder the model has
*proven* it reproduces reliably (a common training-data name, not an
arbitrary string — verified that arbitrary OOV strings get hallucinated
exactly like real OOV words) → translate normally → restore the original
text (or a small glossary form) afterward. No retraining required.

**Bugs found and fixed while building this** (all caught by testing
before trusting the output, not by inspection):
- NER false positives on this corpus's space-split contractions ("doesn
  t" tagged as the name "Don") — fixed with a denylist.
- Grammar corruption when a PERSON-shaped placeholder got substituted for
  a place/org entity — fixed by restricting to PERSON entities only.
- Over-extended entity spans ("mary s mistake" tagged as one span) —
  fixed by requiring single-token spans.
- **Case sensitivity:** spaCy's NER on lowercase input was inconsistent,
  not uniformly bad — "tom"/"mary"/"alice" never detected lowercase in
  any tested context, while "herry"/"paul"/"kevin" detected fine lowercase.
  Fixed with a merged three-pass detector (original casing + a
  title-cased fallback pass, guarded by an original-casing POS check to
  avoid a real new false positive I found: "surprised" title-cased to
  "Surprised" got mistagged PERSON — a curated first-name gazetteer for
  the specific gap that casing tricks alone couldn't close).
- **A restoration cascade bug** (found from a real user-reported case:
  `"John told herry about mary"` → came out as `"mary a parlé de mary à
  herry ."`): restoring multiple entities via *sequential* text
  substitution let one entity's already-restored text collide with a
  different entity's still-pending placeholder, because the fixed
  5-name placeholder pool gets reused across entities in longer
  sentences. Fixed by finding every placeholder in **one simultaneous**
  regex pass over the original text instead of one sequential call per
  entity, so an already-written value can never be mistaken for a
  still-pending placeholder.

**Honest quantified result:** 4/6 OOV names correctly handled (up from
2/6 baseline), aggregate BLEU cost negligible. Reliability is quantified,
not asserted: verified 10/10 on two-entity sentences, but only 1/4 on
three-entity sentences — because the *model itself* sometimes drops one of
three names from its own translation, which no restoration logic can fix
since the text was never generated. The UI surfaces this with a warning
rather than hiding it.

### 2.4 Bidirectional translation without training two models

**Ask:** English→French *and* French→English, without doubling the
training cost.

**Solution:** the target-tag trick from Johnson et al. 2017 (Google's
original multilingual NMT paper) — one shared vocabulary, a `tofr`/`toen`
tag prepended to the source sentence telling the *single* model which
direction to translate. Every training pair gets duplicated as both
directions with its tag. (Note: tried `>>fr<<`-style bracket tags first —
they don't survive this tokenizer's BPE vocabulary, which isn't
byte-level-complete for `<`/`>`; switched to plain alphabetic tags,
verified they round-trip stably before committing to the design — same
lesson that came up again later, see §3.)

**Honest result:** en→fr BLEU 41.34 (a real −3.61 cost vs. the specialized
single-direction model — bidirectionality isn't free), fr→en BLEU 56.03
(notably stronger — a well-documented MT asymmetry: English is an easier
generation target, simpler morphology). Recommendation: kept the
specialized model as the primary one; the bidirectional model is
documented as an optional capability, not silently presented as strictly
better.

### 2.5 The inference engine — Phase 7, the core of the project

This is the part worth spending the most interview time on. Built and
**bit-for-bit correctness-verified against a naive baseline** at every
stage before trusting any benchmark number — the same discipline
throughout.

- **KV cache:** cache previously-computed K/V instead of recomputing full
  self-attention over the whole generated prefix every step. 200/200
  bit-identical to the naive baseline. Modest 1.2-1.4× speedup at this
  project's small scale — honestly reported, not oversold (see §4).
- **PagedAttention** (the vLLM mechanism): physical KV-cache memory split
  into fixed-size blocks from one shared pool instead of reserving
  `max_seq_len` contiguously per request. Measured memory fragmentation
  drop from ~83% to ~23%, ~4.6× more concurrent requests at the same
  memory budget.
- **Continuous batching** (the Orca/vLLM scheduling innovation): evict
  finished requests and admit waiting ones *every* decode tick instead of
  waiting for a whole static batch to finish. P50 latency dropped ~1.9×
  in a staggered-arrival simulation.
- **Speculative decoding:** a small draft model (1.78M params, 16% of the
  target's size, same shared vocabulary) proposes several tokens ahead;
  the target model verifies them all in one parallel forward pass instead
  of one sequential call per token. Because this project's decode
  strategy is greedy (deterministic), the accept/reject rule is exact
  equality, not the usual probabilistic acceptance test — making this
  **bit-identical**, not just distributionally equivalent, to naive
  decoding. Up to 3.1× fewer target-model calls at k=8.
- **Assembling the engine:** combined all of the above into one serving
  loop with a request-in/translation-out API, plus a thin FastAPI server
  on top. **Two real bugs found and fixed** while wiring speculative
  decoding into the paged-memory scheduler (an indexing bug reading the
  wrong absolute position from a concatenated cache tensor from round 2
  onward, and a persist-count bug specific to the token-rejection case) —
  both caught because the correctness bar was "bit-identical to greedy
  decode," not "looks plausible."

**Why this section matters most in an interview:** tokenizer/PE/attention
sweeps are a solid ML engineering exercise, but KV cache / PagedAttention
/ continuous batching / speculative decoding are literally the
differentiating topics in *LLM systems* interviews right now — this
section is the direct answer to "do you understand how LLM inference
actually gets served in production," built and debugged by hand, not
copied from a blog post.

---

## 3. "Why didn't you apply X?" — anticipated questions, honest answers

Answer these the same way the project itself handles tradeoffs: state the
real reason, not a vague "didn't have time."

**Q: Why not use a pretrained model / HuggingFace transformers?**
A: The entire point was to build and understand every component myself —
tokenization, positional encoding, attention, decoding, and especially the
inference-serving stack. A pretrained model would have hidden exactly the
internals I wanted to demonstrate understanding of.

**Q: Why not a copy mechanism or constrained decoding for the proper-noun
problem, instead of NER+masking?**
A: Both are real, more architecturally "correct" fixes — but a copy
mechanism requires retraining with attention-based copy gates, and
constrained decoding requires changing the search algorithm itself. NER +
masking gets a large fraction of the benefit with zero retraining and
wraps *any* existing model, which was the better cost/benefit call for
this project's scope. I designed both and documented the tradeoff rather
than silently picking one.

**Q: Why not literal `[NOUN]`/`[PERSON]` bracket tokens instead of
name-based placeholders?**
A: I actually tried this and it doesn't work — `[` and `]` aren't in this
tokenizer's vocabulary at all (never appeared in the plain-text training
corpus), so a bracket tag gets split into `<unk>` fragments and the model
hallucinates something unrelated (verified: `"my name is [NOUN] ."` →
`"mon nom est en décor de reprises ."`). The real fix would be adding
dedicated tokens to the vocabulary and fine-tuning the model to treat them
as copy-through markers — a real, larger investment I scoped out and
documented rather than silently doing something broken. What I shipped
instead: the model still sees the proven-safe name placeholder
internally, but the UI *displays* it to the user with a `[PERSON_1]`-style
tag — same clarity, no OOV failure.

**Q: Why greedy decoding as the primary strategy instead of beam search
or sampling?**
A: Measured, not assumed — at this model's scale, its output distribution
isn't sharp enough for beam search's wider exploration to reliably beat
simple argmax; a 150-sentence per-sentence comparison confirmed the
smaller model's BLEU-driven "wins" were mostly rote memorization of
short template sentences, not real quality. All four strategies (greedy,
beam, top-k, top-p) are implemented and selectable in the live demo,
including tunable beam size and length penalty — greedy is just the
empirically-justified default.

**Q: Why does speculative decoding barely speed things up in your
benchmarks?**
A: Honestly reported, not hidden: target-model call reduction is real and
substantial (up to 3.1× fewer calls), but wall-clock speedup at this
project's scale (a 10.85M-parameter target model, ~10-token average
outputs) clusters around 0.8-1.0×, because the draft model's own
uncached forward passes add real overhead that roughly cancels the
savings. This is *expected*, not a bug — speculative decoding's real
payoff needs a large, expensive target model (70B-parameter class) where
a verification pass is far cheaper than the sequential calls it replaces.
I explain *why* it doesn't help here rather than pretending it does.

**Q: Why not YaRN / long-context extrapolation?**
A: Out of scope by design — YaRN solves length extrapolation via
NTK-aware RoPE frequency rescaling, and this dataset's sentences max out
at 64 tokens. There's no length-generalization problem here for YaRN to
solve; including it would've been technique-collecting, not
problem-solving.

**Q: Why doesn't it correct grammar or spelling?**
A: It's a pure translation model, not a spell-checker — I tested this
directly rather than assuming. Typos degrade output badly (become
unfamiliar subword fragments, hallucinate unrelated content); minor
grammatical slips are a coin-flip (sometimes come out grammatically
correct by statistical coincidence, sometimes lose meaning entirely, e.g.
losing an intended past tense). No explicit grammar-checking component
exists anywhere in the pipeline — flagging this as a known gap rather
than an implicit claim.

**Q: The frontend only shows word/BPE + RoPE/sinusoidal. Where's
ALiBi / char / the draft model / bidirectional?**
A: All of it is built, trained, and documented in `PROGRESS.md` and
`results/manifest.json` — the frontend was deliberately trimmed to the
core story (word/BPE tokenizers × sinusoidal/RoPE, the comparison that
actually drove the final architecture decision) so a demo doesn't drown
in 30 checkpoints. ALiBi's result is real (it lands consistently between
sinusoidal and RoPE, as attention theory predicts) but isn't the headline
finding; char+sinusoidal's *collapse* is the headline finding of its own
deep-dive section, kept in the write-up even though it's currently
unhooked from live navigation. Nothing was hidden because it failed —
it's a scope decision about what a five-minute demo should show, all of
it is one `git log`/file-read away.

---

## 4. "This is a simple project" — how to respond

Don't get defensive — reframe with specifics. The honest pitch:

> "The scope is intentionally focused — one language pair, a Transformer
> I fully control the size of. But 'simple' undersells what's actually in
> it: I found and root-caused two distinct training collapses with
> measured evidence (not guesswork), and I built a from-scratch inference
> engine — KV cache, PagedAttention, continuous batching, speculative
> decoding — the exact stack behind vLLM, correctness-verified
> bit-for-bit at every stage before I trusted a single benchmark number.
> That's not the fun-toy version of an ML project; that's the LLM-systems
> half of the stack most ML projects skip entirely, because it's the
> genuinely hard, unglamorous part. The translation quality work
> demonstrates I can run a rigorous empirical comparison and land on a
> defensible architecture decision instead of guessing hyperparameters. If
> anything, keeping the model small (10.85M params, no massive GPU
> cluster) let me spend the time budget on the inference engineering
> instead of just scaling up a bigger model and calling it a day."

If pushed further — **"then why isn't the inference engine wired into the
live demo?"** — answer honestly: it was, and was pulled from the frontend
navigation to keep the demo focused and fast to click through in an
interview setting; the code (`serving/engine.py`, `serving/
paged_attention.py`, `serving/batching.py`, `serving/speculative.py`) is
fully built, benchmarked, and documented, and can be re-enabled or walked
through directly in the repo on request. Never claim work exists that
doesn't — everything referenced above is real, checked-in code with
matching benchmark numbers in `PROGRESS.md`.

**If they ask what you'd do with more time (this is your "I know more
than I shipped" moment — say it plainly):**
- Cache the draft model's own proposal phase in speculative decoding
  (currently deliberately uncached, to avoid a second cache-rollback
  problem on top of the target's) — would reduce, not eliminate, the
  overhead that currently limits its wall-clock win.
- True token-position-tracked entity restoration instead of text
  substitution, to close the one residual collision case that's still an
  accepted limitation (the pipeline's own output independently generating
  a placeholder word as unrelated content).
- Extend the vocabulary with dedicated `[COPY]`-style marker tokens and
  fine-tune the model to treat them as copy-through instructions, instead
  of the safe-name-placeholder workaround.
- Scale the inference engine's benchmarks to a genuinely large target
  model, where speculative decoding's wall-clock win should actually
  show up strongly instead of being overhead-dominated.
- Continuous-batch the plain (non-speculative) decode path itself into one
  padded/masked tensor op across requests of different lengths, instead of
  looping per request each tick — the scheduling *policy* win is already
  demonstrated, but a real engine would combine it with batched tensor
  ops for the throughput win on top.

---

## 5. Quick-fire numbers cheat sheet

| Metric | Value |
|---|---|
| Winning config | BPE joint vocab (8K shared tokens), RoPE, MHA, 10.85M params |
| BLEU / chrF++ / TER | 43.75 / 63.12 / 37.53 |
| Corpus size | 106,531 train / 26,633 test pairs, English↔French |
| Total checkpoints trained | ~30 across tokenizer × PE × attention × tuning sweeps |
| Round-0 collapse | 12/14 variants, 100M-param config, root-caused to capacity/data mismatch |
| char+sinusoidal collapse | encoder cosine similarity 1.0000 (fully collapsed) vs. 0.013 for RoPE |
| MHA vs. GQA vs. MQA | 4× and 8× smaller KV cache for −0.2 to −0.85 BLEU — GQA sits at the quality/cache knee |
| NER+masking pipeline | 4/6 OOV names fixed (up from 2/6), BLEU cost ≈ 0, 10/10 reliable at 2 entities, 1/4 at 3 |
| Bidirectional (1 model, 2 directions) | en→fr 41.34 BLEU, fr→en 56.03 BLEU |
| Draft model (speculative decoding) | 1.78M params (16% of target), same shared vocab |
| KV cache speedup | ~1.2-1.4× (small-scale, honestly reported, not oversold) |
| PagedAttention | fragmentation 83%→23%, ~4.6× more concurrent requests at same memory |
| Continuous batching | P50 latency ~1.9× lower under staggered arrivals |
| Speculative decoding | up to 3.1× fewer target-model calls (k=8); wall-clock ~flat at this model scale — explained, not hidden |
| Correctness bar used throughout | bit-identical output vs. naive baseline, verified before trusting any benchmark |

---

## 6. If they ask you to whiteboard or explain something on the spot

Be ready to explain, from memory, in plain terms:
- **Why RoPE beats sinusoidal:** sinusoidal PE is added once at the
  embedding and has to survive N layers of self-attention (an averaging
  operation) purely through the residual stream; RoPE re-derives relative
  position at every layer's attention score directly, so it can't be
  washed out by depth or sequence length the way sinusoidal can.
- **What PagedAttention actually does:** treats GPU memory for the KV
  cache like OS virtual memory — fixed-size blocks from one shared pool,
  a per-request block table mapping logical positions to (possibly
  non-contiguous) physical blocks, allocated on demand instead of
  reserving `max_seq_len` worth of memory up front per request.
- **Why speculative decoding is exact here, not approximate:** the
  general algorithm needs a probabilistic accept/reject test because it's
  usually paired with sampling; greedy decoding is deterministic, so
  "accept iff the draft's token equals the target's own argmax" is
  already exactly correct — no approximation gap to fall back on.
- **The KV-cache-size formula actually used:**
  `bytes/token = 2 × num_layers × num_kv_heads × head_dim × 4` (the 2 is
  K and V, the 4 is fp32 bytes) — and why GQA/MQA only touch
  `num_kv_heads`, not `num_heads` overall (queries stay full multi-head;
  only keys/values are shared across groups).
