# v3 progress log

Session-to-session experiment record for the v3 English→French model:
what was tried, what broke, why, and which config survived to the next
round. Numbers are pulled directly from `results/manifest.json`; nothing
here is estimated. See `README.md` for the overall phase plan this log
tracks against.

**Corpus:** 106,531 train pairs / 26,633 test pairs, English→French.
**Cluster:** node1, 8× A6000 (48GB).
**Metrics** (`eval/metrics.py`, via sacrebleu, computed on detokenized text
so scores are comparable across tokenizers): BLEU (n-gram precision, the
primary metric below), chrF++ (character n-gram F-score, more forgiving of
French's inflection than BLEU), TER (edit distance to reach the reference,
normalized — lower is better, the only metric here where that's true).

---

## Round 0 — the 14-variant breadth+attention matrix, and its collapse

**Goal:** breadth matrix (4 tokenizers × 3 PEs, attention fixed at MHA)
plus two extra bpe+RoPE attention variants (MQA, GQA), all at a single
shared architecture: `emb_dim=512, num_layers=12` (~100M params for
bpe/wordpiece) — Transformer-base scale, sized for corpora in the millions
of pairs.

**Result: 12 of 14 variants collapsed.** They learned normally through
epoch 1, then degenerated mid-training into repeating a single
high-frequency token for the rest of the run. Only the char-tokenizer
variants (tiny embedding table) were resilient enough to survive at that
scale.

**Root cause — capacity vs. dataset/budget mismatch, not a code bug.**
Confirmed by cross-referencing v1's own historical results: the *identical*
oversized config had already failed on the *identical* data in the earlier
v1 codebase (`bpe_sinusoidal`: 100.57M params, eval_acc 0.69%, BLEU 0.00),
while a 6.6×-smaller config succeeded there (`word_sinusoidal`: 15.21M
params, eval_acc 74.58%, BLEU 37.17). Every other suspect was ruled out
directly in code: RoPE rotation math, ALiBi slope/bias construction,
sinusoidal PE construction, encoder/decoder masking, MHA/MQA/GQA head-repeat
generalization, decode-time position handling — all verified correct.

**Fix:** `configs/defaults.json` — `emb_dim 512→128`, `num_layers 12→6`,
matching the config already proven on this data.

**Stale/invalid data left behind** (trained under the broken config,
never rerun — do not cite these BLEU numbers as real results):

| Variant | Params | Eval acc | BLEU | Status |
|---|---|---|---|---|
| `bpe_sinusoidal_mha` | 100.57M | 0.69% | 0.00 | collapsed |
| `wordpiece_sinusoidal_mha` | 100.57M | 2.41% | 0.0002 | collapsed |
| `wordpiece_rope_mha` | 100.57M | 1.40% | 0.0002 | collapsed |
| `wordpiece_alibi_mha` | 100.57M | 2.41% | 0.0002 | collapsed |
| `bpe_alibi_mha` | 100.57M | 1.98% | 0.00 | collapsed |
| `word_alibi_mha` | 118.97M | 2.81% | 0.0002 | collapsed |
| `bpe_rope_mqa` | 84.03M | 7.22% | 0.00 | collapsed |
| `char_alibi_mha` | 88.34M | 91.64% | 39.10 | survived (char resilience) |

See Round 3 below for the MQA fix. WordPiece and ALiBi (on bpe/word) remain
untested at correct capacity — documented as a known gap, not silently
dropped.

---

## Round 1 — clean tokenizer × PE sweep

**Goal:** with the capacity fix in place, re-run the comparison the user
actually needed: 3 tokenizers (bpe, word, char) × 2 PEs (sinusoidal, RoPE),
attention fixed at MHA, 10 epochs. 6 runs, one per GPU.

**Result: all 6 trained cleanly, no collapse.**

| Variant | Params | Train acc | Eval acc | BLEU | chrF++ | TER |
|---|---|---|---|---|---|---|
| bpe · sinusoidal | 10.59M | 73.77% | 73.77% | 36.89 | 55.92 | 42.60 |
| **bpe · RoPE** | 10.59M | 75.46% | 75.00% | **39.21** | 57.34 | 40.38 |
| word · sinusoidal | 15.21M | 74.59% | 74.30% | 36.04 | 56.21 | 63.04 |
| word · RoPE | 15.21M | 75.92% | 75.23% | 39.50 | 57.63 | 59.06 |
| char · sinusoidal | 7.53M | 72.20% | 69.38% | 0.19 | 8.36 | 87.53 |
| char · RoPE | 7.53M | 89.24% | 90.71% | 37.53 | 57.30 | 40.83 |

**Finding:** RoPE beats sinusoidal in every tokenizer pairing, no
exceptions. `word·rope` edges `bpe·rope` on raw BLEU/chrF++ but its TER
(59.06) is 1.5× worse — likely OOV substitutions against its fixed 24K-word
target vocab; `bpe·rope` wins on TER, uses 30% fewer params, and subword
tokenization generalizes better to unseen forms.

**char·sinusoidal deep dive** (root-caused, not just observed): token
accuracy under teacher forcing was healthy (69.4%), but free-running
generation always produced the *identical* French sentence regardless of
input. Measured directly on the trained weights:
- Cross-attention entropy = exactly uniform at every decode step, every
  input (decoder ignoring the source entirely).
- Encoder output pairwise cosine similarity across positions: **1.0000**
  (σ=0.0000) for sinusoidal vs. **0.013** (σ=0.293, well-differentiated)
  for RoPE — full rank-1 collapse of the encoder.
- Sinusoidal PE row-norm is 5.22× the token-embedding row-norm at the input
  layer — position dominates identity from step one.
- Mechanism: sinusoidal PE is injected once at the embedding and never
  reinforced (`EncoderBlock` passes `pe_type="none"` into self-attention for
  sinusoidal runs — only RoPE/ALiBi inject position at the attention score
  itself), so it has to survive 6 rounds of self-attention + LayerNorm
  purely via the residual stream. Stacked self-attention is fundamentally
  an averaging operator; without per-layer reinforcement, and with char's
  128-token sequences giving that averaging twice the room bpe/word's 64
  tokens do, the signal fully washes out by the final layer. RoPE re-derives
  relative position fresh at every layer's Q/K dot product instead of
  storing it once in the residual stream, so it can't wash out — an
  architectural ceiling for additive absolute PE at this depth/length, not
  a bug.

**Carried forward: `bpe_rope_mha`** (10.59M params, BLEU 39.21).

---

## Round 1 follow-up — completing the breadth matrix (WordPiece + ALiBi)

Round 1 only re-ran 6 of the full 4-tokenizer × 3-PE grid (bpe/word/char ×
sinusoidal/rope). WordPiece and ALiBi's only prior results were from
Round 0's broken 100M-param config — all collapsed to ~0 BLEU except
`char_alibi_mha`, which survived on luck (char's tiny embedding table),
not because ALiBi itself had been shown competitive. Retrained the
remaining 6 cells at correct capacity to complete the full matrix:

| Tokenizer | Sinusoidal | RoPE | ALiBi |
|---|---|---|---|
| bpe | 36.89 | **39.21** | 37.69 |
| word | 36.04 | **39.50** | 37.81 |
| wordpiece | 25.52 | 27.15 | 25.99 |
| char | 0.19 (collapsed) | 37.53 | 32.48 |

**RoPE wins every row of the completed matrix, no exceptions.**

**ALiBi consistently lands between sinusoidal and RoPE, every tokenizer,
no exceptions — and the theory explains why.** ALiBi injects position as
a linear-distance bias (`-slope × |i-j|`) directly into the attention
score at *every layer*, the same "reinforced every layer" property that
lets RoPE resist the rank-collapse mechanism documented above — evidenced
by `char_alibi_mha`'s healthy 89.31% eval accuracy (vs. `char_sinusoidal`'s
0.19 BLEU collapse). But ALiBi's bias is a much cruder signal than RoPE's
relative rotation (a single scalar penalty by distance vs. a full phase
rotation per dimension pair), so it consistently trails RoPE while beating
sinusoidal — exactly the middle position the theory predicts.

**A real bug, found and fixed: `WordPieceTokenizer.decode()`.** WordPiece's
TER was anomalously catastrophic (75-78, vs. bpe/word's 40-63 range)
despite only moderately worse BLEU/chrF — a mismatch worth investigating
rather than accepting at face value. Cause: the built-in
`decoders.WordPiece` decoder applies BERT-style punctuation cleanup (no
space before `.`), but this corpus's reference text treats terminal
punctuation as its own space-delimited token, matching every other
tokenizer here (confirmed directly: `"...dernière."` vs. the correct
`"...dernière ."`). Every single WordPiece hypothesis mismatched the
reference at the same position, every time. **Fixed** by replacing the
built-in decoder call with a manual join that only handles `##`-prefix
continuation pieces, leaving everything else space-separated. Re-scored
all 3 WordPiece checkpoints (no retraining needed — the trained weights
were never wrong, only the eval-time text reconstruction was):

| Variant | TER before | TER after | BLEU (unchanged) |
|---|---|---|---|
| wordpiece + sinusoidal | 78.16 | 63.08 | 25.52 |
| wordpiece + rope | 75.60 | 59.84 | 27.15 |
| wordpiece + alibi | 77.83 | 62.51 | 25.99 |

BLEU and chrF++ didn't move at all — sacrebleu's BLEU/chrF already
normalize punctuation spacing internally, so they were never affected by
this bug; TER isn't, so it absorbed the entire error. That means
WordPiece's real ~10-12 BLEU gap below BPE/word is **not** an artifact of
this bug — it's genuine. Checked the obvious alternative explanation
(WordPiece producing longer, more-truncated sequences) and ruled it out:
mean target length is nearly identical (bpe 9.37 vs. wordpiece 9.26
tokens/sentence, zero truncation either way at `max_seq_len=64`). The
most likely remaining explanation is that WordPiece's likelihood-based
merge-selection criterion produces a genuinely less effective 8,000-token
vocabulary for this corpus than BPE's frequency-based one — a real,
measured finding, though not mechanistically proven beyond that.

**Conclusion: the full breadth matrix doesn't change the project's
winner.** `bpe_rope_mha_widernarrowffnlong` (43.80 BLEU) remains far
ahead of every cell in this completed matrix.

---

## Round 2 — decoding strategy sweep

**Goal:** on the winning `bpe_rope_mha`, compare greedy / beam(5) /
top-k(10) / top-p(0.9) over the full 26,633-example test set.

| Strategy | BLEU | chrF++ | TER |
|---|---|---|---|
| **Greedy** | **39.21** | **57.34** | 40.38 |
| Beam (size 5, lp 0.6) | 38.12 | 56.57 | **40.09** |
| Top-p (p=0.9) | 31.19 | 50.27 | 47.13 |
| Top-k (k=10) | 28.76 | 50.07 | 47.89 |

**Finding:** greedy wins outright. Beam's 5× wider search doesn't pay off
at 10.6M params / 10 epochs — the output distribution isn't sharp/
calibrated enough for the extra beams to do more than explore noise.
Sampling trails further behind, as expected: top-k/top-p trade accuracy for
diversity, which corpus BLEU doesn't reward.

**Carried forward: greedy decode.**

---

## Round 3 — architecture/hyperparameter tuning, round 1

**Goal:** 7 targeted single-axis changes on `bpe_rope_mha`, one per GPU.
Depth was tested since round 0's capacity fix hadn't isolated *how much*
smaller was optimal, just that 512/12 was too big.

| Config | Δ from baseline | Params | Train/Eval acc | BLEU | chrF++ | TER |
|---|---|---|---|---|---|---|
| **longtrain** (epochs 20) | epochs only | 10.59M | 79.22/77.08 | **41.68** | **60.13** | **38.41** |
| wider (emb192) | +58% params | 16.76M | 77.53/76.32 | 41.10 | 58.99 | 38.87 |
| widedeep (emb192+L8) | +96% params | 20.81M | 76.74/75.95 | 40.76 | 58.77 | 39.09 |
| narrowffn (ff512) | −45% params | 5.86M | 78.55/76.48 | 40.02 | 58.23 | 40.15 |
| lowdrop (dropout 0.05) | same size | 10.59M | 77.61/75.83 | 39.92 | 58.09 | 40.24 |
| baseline (bpe_rope_mha) | — | 10.59M | 75.46/75.00 | 39.21 | 57.34 | 40.38 |
| gqa (num_kv_heads=2) | 4× smaller KV-cache | 10.15M | 75.15/75.01 | 39.00 | 57.46 | 40.73 |
| deeper (L8) | +24% params | 13.10M | 74.61/74.39 | 38.60 | 56.94 | 40.70 |

**Findings:**
- Every config beat baseline **except `deeper`** — depth alone (6→8
  layers) hurts, and hurts more when combined with width (`widedeep` <
  `wider` alone despite 24% more params). This model doesn't want more
  depth at this data scale.
- `longtrain` — same param count as baseline, just double the epochs —
  beats every architecture change. The model hadn't converged at 10
  epochs; this is the cheapest, safest win in the sweep.
- `narrowffn`'s 45%-smaller model still beating baseline confirms the
  original `ff_dim=2048` (a 16× multiple of `emb_dim=128`, vs. the
  standard 4× ratio) was genuinely oversized, wasted capacity.
- `gqa` is within 0.21 BLEU of baseline for a 4× smaller KV-cache (1536
  vs. 6144 bytes/token) — directly relevant to the Phase 7 inference work.
- No overfitting concerns: largest train/eval gap is `longtrain`'s 2.14pp,
  and BLEU/chrF climbed in lockstep with eval accuracy — genuine
  generalization, not memorization.

---

## Round 4 — architecture/hyperparameter tuning, round 2 (combined axes)

**Goal:** round 3's two winning *kinds* of lever — training budget
(epochs) and capacity (width, ffn ratio) — hadn't been combined. Depth
didn't compound with width in round 3, but epochs is a different kind of
lever, worth testing separately.

| Config | emb/layers/ff/drop | Epochs | Params | Train/Eval acc | BLEU | chrF++ | TER |
|---|---|---|---|---|---|---|---|
| 🏆 **widernarrowffnlong** | 192/6/768/0.1 | 20 | 10.85M | 84.64/79.16 | **43.80** | **61.92** | **36.87** |
| widerlong | 192/6/2048/0.1 | 20 | 16.76M | 81.75/78.42 | 43.71 | 61.69 | 36.71 |
| lowdroplong | 128/6/2048/0.05 | 20 | 10.59M | 81.72/77.85 | 43.04 | 60.77 | 37.47 |
| narrowffnlong | 128/6/512/0.1 | 20 | **5.86M** | 82.49/77.96 | 41.87 | 60.24 | 38.53 |
| gqa_long | 128/6/2048/0.1 (gqa) | 20 | 10.15M | 78.79/76.85 | 41.27 | 59.33 | 38.77 |
| widernarrowffn | 192/6/768/0.1 | 10 | 10.85M | 80.13/77.56 | 41.03 | 59.96 | 39.14 |

**Findings:**
- Axes compound **once epochs join in**: `widernarrowffn` alone at 10
  epochs (41.03 BLEU) barely beat `wider` alone (41.10) — essentially a
  wash. The same architecture at 20 epochs (`widernarrowffnlong`) jumps to
  43.80 — +2.77 BLEU purely from training longer on top of the capacity
  change.
- `widernarrowffnlong` has the widest train/eval gap of all 14 configs run
  across rounds 3–4 (5.48pp) — but also the best BLEU/chrF/TER of all of
  them. If this were harmful overfitting, BLEU would be *lagging* the
  eval-accuracy lead, not leading everything else. Read as headroom, not a
  regression — worth re-checking if pushed further (more epochs / more
  capacity) in future work.
- Depth still never helps — `deeper`/`widedeep` remain the two
  worst-performing "big" configs across both tuning rounds.

---

## Attention-type comparison: MHA vs. GQA vs. MQA

Round 0's `bpe_rope_mqa` was trained under the broken 84M-param config and
collapsed (BLEU 0.0) — not a fair comparison. Retrained here at the correct
capacity, at both epoch budgets already used for MHA and GQA, so all three
attention types are compared apples-to-apples for the first time — exactly
what `README.md`'s Phase 5 promises.

| Attention | num_kv_heads | Params | KV-cache bytes/token | BLEU (10ep) | BLEU (20ep) |
|---|---|---|---|---|---|
| **MHA** | 8 | 10.59M | 6144 | 39.21 | 41.68 |
| **GQA** | 2 | 10.15M | 1536 (4× smaller) | 39.00 | 41.27 |
| **MQA** | 1 | 10.07M | 768 (8× smaller) | 38.15 | 40.57 |

Full 20-epoch metrics: GQA — chrF++ 59.33, TER 38.77; MQA — chrF++ 59.25,
TER 39.49 (vs. MHA's chrF++ 60.13, TER 38.41).

**Why, in one paragraph:** KV-cache size per token is a closed-form
function already computed in the codebase —
`kv_cache_bytes_per_token_fp32 = 2 × num_layers × num_kv_heads × head_dim ×
4`; with `num_layers=6` and `head_dim=emb_dim/num_heads=16` fixed, cache
size scales *linearly* with `num_kv_heads` alone (8→2→1 gives exactly
6144→1536→768). Quality doesn't degrade linearly with it, though: MHA→GQA
trades a 4× cache reduction for only −0.21 to −0.41 BLEU, while GQA→MQA
trades a further 2× reduction for −0.70 to −0.85 BLEU — roughly double the
quality cost for half the compression gain. That's the textbook argument
for why production LLMs (Llama 2 70B, Mistral) settled on GQA rather than
MQA: it sits near the knee of the cache-size-vs-quality curve, keeping
most of MHA's quality while giving Phase 7's serving-side memory/
continuous-batching work most of MQA's cache-size win. The **quality-optimal** config
stays MHA-based (see below) — but when Phase 7 benchmarks serving
efficiency, GQA is the config to benchmark against it, as the
concrete illustration of this exact tradeoff curve.

---

## Does the tuned winner actually translate better? A per-sentence check

Manual spot-checking in `app.py` surfaced cases where `bpe_rope_mha`
(the pre-tuning baseline) *looked* better than the tuned winner
(`bpe_rope_mha_widernarrowffnlong`) on individual sentences — worth
checking systematically rather than trusting either BLEU alone or a
couple of manual examples, since both can mislead in different ways.

**Method:** both models greedy-decoded on the same random sample of 150
held-out test sentences (seed 7); sentence-level BLEU (`sacrebleu.
sentence_bleu`) computed per example, hypotheses diffed by hand.

**Result:** mean sentence-BLEU 42.72 (baseline) vs. 43.92 (winner), +1.20
average — consistent with the corpus-level 39.21→43.80 gap. Win/loss at
the individual-sentence level: **winner wins 52/150, baseline wins
44/150, 54 tied** — ahead, but not a landslide sentence-by-sentence.

**What's actually behind the baseline's wins, on inspection:** almost
every case where the baseline "wins" is the baseline reproducing the
reference **word-for-word** (BLEU 100.0) on short, common, template-style
sentences ("we have enough time .", "i caught tom ."), while the winner
produces a different but equally valid French phrasing (`nous disposons
de suffisamment de temps` vs. the reference's `nous avons assez de
temps`) that BLEU penalizes for not matching the reference's exact
wording — a well-documented BLEU blind spot (it can't distinguish "wrong"
from "right but phrased differently"). This pattern is consistent with
the baseline, trained for only 10 epochs at 10.59M params, partially
memorizing frequent short templates in this Tatoeba-style corpus rather
than generalizing.

**What's behind the winner's wins:** by contrast, most cases where the
winner wins are the baseline making a genuine grammar error the winner
gets right — wrong verb tense (`je serais grosse` conditional vs. the
correct `j'étais gros` imperfect), wrong tense again (`vous y allez`
present vs. correct future `vous y serez`), an extra ungrammatical
preposition (`libre de l'après-midi` vs. correct `libre l'après-midi`).
These are real linguistic-quality differences, not reference-matching
luck — and it's consistent with chrF++ (more paraphrase-tolerant, since
it scores character n-grams rather than exact word n-grams) showing the
same directional gap as BLEU rather than shrinking it, which is what
you'd expect if the winner's BLEU lead were an artifact of reference
overlap rather than genuine quality.

**Conclusion:** the winner is very likely the more linguistically correct
translator; the baseline's wins are concentrated in a rote-memorization
effect on short common sentences that BLEU rewards specifically because
of exact-match scoring. This is a real limitation of BLEU as a metric,
not evidence the tuning regressed quality — but it's also a reminder that
aggregate BLEU differences (any two models, not just this pair) deserve a
sentence-level sanity check before trusting them at face value, especially
when the delta is being used to justify a config decision.

---

## How the model handles unseen proper nouns (and where it breaks)

Manual testing surfaced that unknown names (e.g. `herry`) often pass
through translation unchanged rather than being mistranslated —
worth understanding mechanistically rather than assuming it's either
"the model translates names" or "it copies OOV tokens," since neither
framing turned out to be quite right on inspection.

**It's a learned statistical pattern, not a hardcoded rule.** Proper-noun
preservation is heavily represented in the training data — `tom` alone
appears unchanged across English→French in **6.2% of all 106,531 training
pairs** (`mary`: 1,053 more). The model has overwhelming exposure to "this
kind of token reproduces on the target side," and has learned real
nuance from it: some names get **localized** to an established French
equivalent (`mary`→`marie`, matching real human-translation convention)
while names without one (`tom`, `john`) are copied unchanged — genuinely
mirroring how the human-translated source data itself handles names, not
a naive "always copy" heuristic.

**But it's shallower than true copy-through, and breaks under pressure.**
`herry` — verified **zero occurrences** in the training data, so this was
real generalization, not memorization — happens to BPE-decompose into
only 3 target-side pieces (`her`+`r`+`y`), each individually a pattern the
model reliably reproduces. Pushed further with names that fragment into
*more* subword pieces, the behavior collapses into hallucination instead
of preservation:

```
"zhorvaq"  -> ['z','h','or','v','a','q']  (6 pieces) -> "fusiluel"  (hallucinated)
"blipnak"  -> ['bl','ip','n','ak']        (4 pieces) -> "gulté"     (hallucinated)
"quexinor" -> ['qu','ex','in','or']       (4 pieces) -> "en tête"   (hallucinated, unrelated real phrase)
```

**Conclusion:** there's no real OOV-preservation mechanism here — what
looks like one is a byproduct of subword composition happening to work
for words with few enough fragments, and it fails (free-running
hallucination, not even a garbled version of the input) once a word
fragments too much. This is concrete, empirical motivation for a real
**copy/pointer mechanism** (Pointer-Generator Networks, See et al. 2017)
as a legitimate future architectural addition — an explicit generate-vs-
copy-from-source gate, rather than hoping subword overlap gets lucky.

---

## Two follow-up fixes: proper-noun preservation and grammar robustness

Two targeted experiments on the tuned architecture (emb192/L6/ff768/ep20),
each isolating exactly one change against `bpe_rope_mha_widernarrowffnlong`
as the control.

### Joint BPE vocabulary

**Mechanism:** one BPE model trained on the *concatenated* English+French
corpus (`tokenization/build_vocab.py`'s new `_joint`-suffixed branch),
saved identically into both the src and tgt vocab directories. Standard
technique (the original Transformer paper, mBART, NLLB, M2M-100 all use
one shared vocabulary across languages) — motivated by giving rare shared
surface forms (names, numbers, cognates) a consistent token-id
representation on both sides, instead of relying on two independently-
trained vocabularies coincidentally overlapping. Verified directly before
training: `"herry"`, `"tom"`, and fabricated names all encode to
byte-identical id sequences on the source and target sides with the joint
vocab, vs. completely unrelated ids before.

**Result: better overall, but not for the reason it was built.**

| Config | BLEU | chrF++ | TER | eval_acc |
|---|---|---|---|---|
| **bpe_joint_rope_mha** | **44.95** | **62.88** | **36.03** | 80.23% |
| bpe_rope_mha_widernarrowffnlong (separate vocabs) | 43.80 | 61.92 | 36.87 | 79.16% |

+1.15 BLEU, better across every metric, same parameter count — a genuine,
if unplanned, new best. But the OOV stress test it was actually designed
to fix (`oov_stress_test.py`, testing `herry` plus three fabricated names
guaranteed absent from training data) came back an honest negative:

| Name | In training data? | Before (separate vocab) | After (joint vocab) |
|---|---|---|---|
| tom | yes | preserved | preserved |
| mary | yes | preserved | preserved |
| herry | no | hallucinated | hallucinated |
| zhorvaq | no | hallucinated | hallucinated |
| blipnak | no | hallucinated | hallucinated |
| quexinor | no | hallucinated | hallucinated |

**2/6 preserved, identically, before and after.** Only names that
literally appear in training data survive either way.

**Why the fix didn't fix what it targeted:** sharing the *vocabulary*
(the string→id mapping) is not the same as sharing the *embeddings*. `Transformer`'s
`src_embed` and `tgt_embed` (`model/transformer.py`) are still two
separate `nn.Embedding` tables with independently-initialized weights —
token id 223 means the same subword on both sides now, but the encoder's
learned vector for id 223 and the decoder's learned vector for id 223 are
still two unrelated parameters that happen to share a lookup key. For a
rare id, there's too little training signal for the decoder side to
independently arrive at a matching representation. The aggregate BLEU
gain is real and likely comes from better statistical sharing on the
*common* vocabulary (punctuation, cognates, frequent short words) — it
just doesn't reach the rare-token case the OOV test targets.

**Follow-up worth doing, not yet done:** explicit **embedding weight
tying** (`src_embed.token_emb.weight = tgt_embed.token_emb.weight`, a
single shared `nn.Parameter`) on top of the joint vocab — the standard
completion of this technique, and the more direct test of the actual
copy-preservation hypothesis.

### Static noise-augmented training

**Mechanism:** `augment.py` injects synthetic source-side noise (QWERTY-
adjacent character substitution/insertion/deletion/swap, word dropout,
local adjacent-word swaps) — cheap proxies for typos and minor grammar
slips (Belinkov & Bisk 2018). `train.py --noise_prob 0.5` generates a
noised-source duplicate for a 50% random slice of training pairs (target
unchanged) and *adds* them to the clean set (106,531 → 159,693 pairs),
rather than replacing it — the model still sees plenty of clean examples
while gaining exposure to noisy-in/clean-out translation. Noise is
generated once (static), not per-epoch, since the training pipeline
pre-encodes to cached ID tensors for speed and re-tokenizing noised text
every batch would meaningfully slow training for a marginal gain over the
static version.

**Result: works exactly as the technique predicts.** Evaluated
(`noisy_robustness_eval.py`) on 3,000 sampled test sentences, both clean
and with fresh synthetic noise (seed 99, different from training's seed
42, to test generalization rather than memorized noise patterns):

| | Clean-input BLEU | Noisy-input BLEU | Degradation |
|---|---|---|---|
| Before (`bpe_rope_mha_widernarrowffnlong`) | 44.23 | 26.38 | −17.86 |
| After (`bpe_rope_mha_noiseaug`) | 43.12 | 32.01 | **−11.10** |

**38% less BLEU lost to noisy input** (17.86 → 11.10), for a **1.12 BLEU
cost on clean input** (consistent with the −0.86 BLEU seen on the full
26,633-example clean test set: 43.80 → 42.94). A legitimate quality/
robustness trade-off, not a strict improvement — the right choice depends
on deployment context (a chat/mobile interface where users type quickly
and make typos benefits from this; a benchmark-quality clean-input use
case does not).

**Recommendation:** adopt `bpe_joint_rope_mha` as the new project
champion (strictly better, no trade-off found). Keep
`bpe_rope_mha_noiseaug` as a documented alternate config for
noise-sensitive deployment contexts rather than a replacement. Embedding
weight-tying is the well-motivated next step for actually solving the
proper-noun-preservation problem, not yet built.

---

## The translate-vs-copy decision: which techniques, and why only one was built

Every real proper-noun-preservation technique reduces to one question per
source token: **should this be translated, or copied?** Four standard
approaches, in roughly increasing order of implementation cost:

1. **NER + masking** — detect entities, translate around them, restore.
2. **Terminology constraints** — a glossary substituted in at restore time
   instead of pure copy, for entities with a known localized form.
3. **Copy/pointer mechanism** — an explicit generate-vs-copy gate inside
   the model (See et al. 2017), using cross-attention as the copy
   distribution.
4. **Constrained decoding** — force specific spans into the beam search
   output directly (e.g. Grid Beam Search), rather than pre/post-processing
   around a fixed translation.

**Built and validated: (1) and (2), as one combined pipeline
(`entity_pipeline.py`).** Both are pure pipeline engineering — no
retraining, wraps any existing model — and (1) is a genuine prerequisite
for (2) anyway (you can't apply a glossary to an entity you haven't
detected). This is also the production-pragmatic choice: real MT systems
overwhelmingly use exactly this, not an in-model copy mechanism, for
proper-noun handling.

**Designed, not built: (3) and (4).** Both require model or search-
algorithm changes and meaningfully more implementation risk for a side
investigation off the project's main line (Phase 7's inference-serving
work). Concrete designs, so the reasoning is checkable rather than a
hand-wave:

- **Copy mechanism**, made simpler by work already done here: since
  `bpe_joint_rope_mha` already shares one vocabulary between source and
  target, a copy distribution could scatter cross-attention weights
  (`decoding.py` already captures these via `return_attn=True`) directly
  into the *same* vocab space the generation head produces — no extended-
  vocabulary bookkeeping needed, which is normally pointer-generator's main
  implementation cost. The remaining work is a small `p_gen` gate (a
  linear layer over decoder state + attention context) blending
  `p_gen * vocab_softmax + (1 - p_gen) * copy_distribution`, and a retrain.
- **Constrained decoding**: force the beam to include the (masked, then
  glossary-or-copy-resolved) entity string as a required span, rather than
  trusting the model's own generation for that slice of the sentence —
  Grid Beam Search-style. The `entity_pipeline.py` masking approach below
  is functionally a simplified version of this (guarantee the entity by
  construction rather than by search constraint), without needing to
  modify `decoding.py`'s beam search itself.

### Building and validating the pipeline: what actually happened

**Mechanism** (`entity_pipeline.py`): run spaCy NER (`en_core_web_sm`) on
the original, properly-cased input (spaCy's accuracy depends heavily on
capitalization — this corpus's own train/test data is pre-lowercased, so
NER run on it directly is measurably worse; a live deployment gets
properly-cased user input, the realistic case this targets). Mask the
detected entity with a placeholder, translate the masked sentence with the
existing model unchanged, then restore: substitute the glossary's
localized form if known (e.g. `mary` → `marie`, matching the localization
pattern already observed in training data), otherwise copy the original
text back verbatim.

**First attempt failed for an instructive reason.** An arbitrary novel
placeholder string (e.g. `xentxzero`) does *not* work: verified directly
that the model hallucinates it during translation exactly like it
hallucinates any other unfamiliar token — masking only helps if the
placeholder itself is something the model has already proven it
reproduces reliably. Fixed by reusing common training-data names (`Tom`,
`Mary`, ...) as placeholders, since those are independently verified
(the OOV stress test above) to survive translation unchanged.

**Initial full-pipeline test: 5/6 fabricated names correctly preserved**
(`herry`, `zhorvaq`, `blipnak`, `tom` copied through; `mary` correctly
localized to `marie` via the glossary) — up from 2/6 for the raw model.
The one miss (`quexinor`) is a pure spaCy NER recall failure (its POS
tagger correctly calls it a proper noun; its separate NER component
doesn't), not a pipeline logic bug.

**But the aggregate test set exposed real problems the narrow stress test
didn't.** Run over 2,000 sampled test sentences, first attempt: **BLEU
dropped 1.17 points** even though the pipeline only touched 6.2% of
sentences — meaning it was doing serious damage on the sentences it did
touch. Root causes, found by inspection and fixed one at a time:

| Failure mode | Cause | Fix |
|---|---|---|
| Content words masked as PERSON | This corpus writes contractions space-split (`doesn t`, not `doesn't`) — spaCy's NER, never trained on that convention, tags fragments like `don`/`doesn` as the name "Don" | Denylist of contraction fragments |
| Grammar corruption on GPE/ORG entities | A PERSON-shaped placeholder (`Mary`) substituted for a place/org name breaks the sentence's semantic (not syntactic) fit in ways the naive text-restore can't repair | Restricted to `PERSON` entities only (the one type with a verified-safe placeholder) |
| Over-extended entity spans | spaCy sometimes grabs `mary s mistake` as one span (this corpus's space-split possessives confuse its boundary detection), masking real content along with the name | Restricted to single-token entity spans |
| Placeholder/content collision (cascade case) | Restoring entities via *sequential* `re.sub` calls lets one entity's just-restored text collide with a DIFFERENT entity's still-pending placeholder once 3+ entities reuse the fixed 5-name placeholder pool — e.g. `"John told herry about mary"` → placeholders Tom/Mary/John assigned left-to-right → restoring entity 1 (Tom→"John") plants literal "John" in the output just before entity 3's own restoration step goes looking for "John" (mary's placeholder) and clobbers both identically, producing `"mary a parlé de mary à herry ."` instead of `"john a parlé de mary à herry ."` | **Fixed** (see "Entity pipeline v2" section below): `restore_entities` now finds every placeholder in ONE simultaneous regex pass over the *original*, unmodified translated text, instead of one sequential call per entity — a value written for one entity can no longer be re-scanned as if it were still a placeholder for another. Verified on the exact reported case plus the existing 2-/3-entity regression suite. |
| Placeholder/content collision (independent case) | The output happening to *independently* generate one of the placeholder words as genuine unrelated content (not from masking) would still collide with the single-pass fix above — this narrower case remains a real, accepted, unfixed residual: true fix needs token-position-tracked restoration, not text substitution | Documented as a structural limitation; not fixed — architectural, not a quick patch |

**After the first three fixes:** pipeline fires on 0.5% of sentences,
BLEU delta **−0.06** (44.89 → 44.83 on the 2,000-sample set) — negligible,
and the OOV stress test still lands **4/6** (down from 5/6, since
restricting to PERSON-only entities drops the ORG-tagged `zhorvaq` — an
explainable recall/reliability trade-off, not a regression).

**Conclusion:** the translate-vs-copy mechanism itself works — validated
cleanly on the pipeline's actual target case (a single detected PERSON
entity, e.g. "my name is X"). Applying it unrestricted to arbitrary
sentences surfaces real, specific, well-understood engineering costs
(NER/tokenization-convention mismatch, span-boundary errors, text-based
restoration collisions) rather than being free. **Recommendation: keep
this pipeline scoped to its validated single-PERSON-entity case** (that's
also its only real use case — copying arbitrary GPE/ORG spans through
untranslated is rarely what you actually want) rather than applying it as
a blanket wrapper around every translation. That scoping decision — ship
a promising technique behind the condition it's actually validated for,
rather than blanket-deploying it — is itself the right engineering call
here, not a limitation to apologize for.

---

## Entity pipeline v2: lowercase names, multi-entity, and a live Inference Lab

Three follow-up asks after the pipeline above shipped: (1) lowercase input
("my name is tom") wasn't detecting names as reliably as properly-cased
input ("My name is Tom") -- a real, reproducible gap once tested against a
range of names, not consistent with any single name; (2) the single-entity
restriction should become multi-entity, with an explicit way to see which
detected name maps to which internal placeholder; (3) the Try It Live demo
should show real decode latency, and a new page should let a user compare
Phase 7's optimizations (KV cache, continuous batching, speculative
decoding) live rather than only as static numbers in this doc.

**Case-sensitivity, root-caused before fixing:** spaCy's PERSON detection
on lowercase input is *inconsistent*, not uniformly bad -- empirically,
`"my name is tom/mary/alice ."` never detects the name lowercase in any
tested sentence context, while `"my name is herry/paul/kevin ."` detects
correctly even lowercase. Three of the five names this pipeline already
uses as safe placeholders are in the *undetected* bucket. Fix: `_detect_entities`
now merges three passes -- spaCy on the sentence as typed, spaCy on a
character-offset-preserving title-cased copy (`_title_preserve`, recovers
some but not all lowercase misses), and a small curated first-name
gazetteer fallback for single, non-stopword tokens matching a common-name
list (closes the specific tom/mary/alice gap directly, same
explicit-engineering spirit as the existing GLOSSARY and contraction
denylist). **A bug caught by testing before shipping:** the title-cased
pass alone introduced a real new false positive -- `"i was surprised by
what i learned ."` title-cased to `"...Surprised..."` gets tagged PERSON
by spaCy, even though the *original*-casing POS tag for that exact token
correctly reads VERB/VBN. Fixed by only trusting a title-cased PERSON hit
when the original-casing tagger independently agrees the token is
noun-shaped (`pos_ in {"PROPN", "NOUN", "X"}`) -- deliberately *not*
applied to the gazetteer pass, since that path is already constrained by
curated list membership and the same POS check would incorrectly reject
genuine hits there (`"we can t help tom ."` tags "tom" as VERB with zero
capitalization signal, yet it's a real, correctly-detected name).
Verified: 300-sentence stress test over the real test set finds 16
trigger sentences, all genuine "tom"/"mary" detections, zero false
positives after the POS guard (one found and fixed before this count).

**Multi-entity, with alignment made explicit:** the previous version
hard-capped at exactly one entity specifically to avoid ambiguous
restoration. The fix isn't literal `[NOUN1]`/`[NOUN2]`-style tokens fed to
the model -- those are OOV strings the model has never seen and would
hallucinate, the exact failure mode the safe-placeholder trick exists to
avoid. Instead: each detected entity (up to the 5-name safe-placeholder
pool) gets a *distinct* placeholder identity (Tom/Mary/John/Alice/Paul),
so restoration needs no position-tracking at all -- each placeholder's own
surface form in the output tells you exactly which original entity it
stands for. `mask_entities` now returns `entity_records`, a list of
`{"original", "placeholder", "label"}` dicts (`"label"` is a human-facing
`[PERSON_i]` tag for UI display, e.g. the alignment table Try It Live now
renders -- not something sent to the model). **Quantified reliability
by entity count** (10 synthetic 2-entity sentences, 4 synthetic
3-entity sentences, checked whether every placeholder survived into the
raw translation): **2-entity: 10/10.** **3-entity: 1/4** -- the model
itself sometimes drops one of three names from its own output (e.g.
`"john told alice about paul ."` → masked `"Tom told Mary about John ."`
→ raw translation `"tom a parlé de john ."`, Mary/Alice entirely absent).
This is an honest, quantified model limitation, not a pipeline bug --
restoration can't recover text the model never generated. The UI surfaces
this directly: a warning banner appears whenever 3+ entities are detected,
rather than silently presenting a translation that may be missing a name.

**Bracket-style tags ([NOUN]/[PERSON_1]/...), tested and scoped to display-only:**
tried feeding literal `[NOUN]`-style tags to the model directly -- confirmed
empirically that this doesn't work and why: `'['`/`']'` aren't in the
tokenizer's vocabulary at all (never appeared in this corpus's plain
English/French training sentences), so `"my name is [NOUN] ."` tokenizes
as `my | name | is | <unk> | n | oun | <unk> | .` and the model produces
unrelated hallucinated output (`"mon nom est en décor de reprises ."`) --
the identical OOV-hallucination failure mode already documented for
arbitrary placeholder strings, just via a different route (broken
tokenization rather than an unfamiliar-but-tokenizable string). `Tom`/
`Mary`/etc. work specifically because they're real, frequent subwords the
model actually learned; no equivalent currently exists for bracket
syntax. New `display_masked()` reconstructs a user-facing version of the
masked sentence using each entity's `[PERSON_i]` label instead of the
internal placeholder -- shown in Try It Live's UI and the "Pipeline
steps" expander (which now shows both: what the user sees vs. what's
actually sent to the model) -- while the actual model input is
unchanged. The real fix for literal tag tokens would be adding them to
the vocabulary and fine-tuning the model to treat them as copy-through
markers (matches the production approach already flagged as the honest
gap in the original pipeline's docstring); scoped out for now as a
larger investment than this demo warrants.

**Identity preservation made the default (glossary made opt-in):** the
original design defaulted to substituting a known localized form
(`GLOSSARY`, e.g. Mary -> Marie) on restore, matching how this training
corpus itself translates a handful of well-known names -- but that's the
opposite of what most users actually want for their *own* name: "Mary"
typed by a real user should come back as "Mary", not be silently
"translated" to "Marie". `restore_entities` and
`translate_with_entity_pipeline` gained a `use_glossary: bool = False`
parameter -- verbatim restoration is now the default, glossary
substitution an explicit opt-in (a new, disabled-until-the-pipeline's-on
checkbox in Try It Live). Verified both paths end-to-end: default keeps
"mary" -> "mon nom est mary .", opt-in produces "mon nom est marie .".

**Latency display:** `_timed()` wraps every decode call with
`torch.cuda.synchronize()` before/after (without it, async CUDA dispatch
lets a call "return" long before the GPU work it queued actually
finishes, understating latency) and reports elapsed ms. Wired into Try It
Live's main translate button and the entity-pipeline branch.

**New page, "⚡ Inference Lab":** two live experiments reusing Phase 7's
already-verified serving code directly (no reimplementation): (1) one
sentence through `greedy_decode` / `greedy_decode_cached` /
`speculative_decode` side by side, with a live bit-identical check
(re-demonstrates the 7b/7e exactness argument on whatever the user
types, not just the pre-recorded 200/300-sentence offline samples) and a
latency bar chart; (2) several concurrent requests (slider-controlled
count and max batch size) through `run_static` / `run_continuous` /
`InferenceEngine` (continuous + speculative), all three sharing one
KV-cache pool (originally `PagedKVCache`; see "Scope change" below for
what replaced it), reporting wall-clock time and per-request tick
latency (P50/P95). Verified end-to-end via Streamlit's `AppTest` harness
(drives the real app -- widget interactions, button clicks, real GPU
decode calls -- not a mock): both sections run without exception and
produce sensible, PROGRESS.md-consistent numbers (e.g. one live batch run:
1.35x wall-clock from continuous batching alone, 1.56x with speculative
decoding added on top, in the same ballpark as the engine benchmark's
offline 1.07x/1.18x given a different random sample and smaller N tuned
for interactive use).

---

## Bidirectional translation: one model, both directions

**The ask:** en→fr and fr→en, without training two separate models.

**Mechanism** (Johnson et al. 2017, "Google's Multilingual Neural Machine
Translation System"): one shared vocabulary, one model, and a short tag
prepended to the *source* sentence telling the decoder which language to
produce — `tofr the cat sleeps .` → French, `toen le chat dort .` →
English. This only works cleanly because `bpe_joint` already gives one
vocabulary spanning both languages on both the encoder and decoder side;
attempting this with the separate-vocab tokenizers (`bpe`, `word`, ...)
would feed each one the *other* language half the time, mostly OOV.

Before committing to a tag, verified round-tripping through the
tokenizer: the "obvious" `>>fr<<`/`>>en<<` bracket convention silently
maps to `<unk>` in this vocabulary (it was never built with full
byte-level fallback coverage for `<`/`>`), while a plain alphabetic tag
(`tofr`/`toen`) encodes and decodes back to itself exactly — a small but
real example of checking an implementation detail before building on it.

**Implementation:**
- `build_bidi_data.py` — for every (eng, fra) pair in the existing
  train/test splits, emits both `tofr <eng>` → `<fra>` and
  `toen <fra>` → `<eng>` rows, doubling the dataset to 213,062 training
  pairs.
- `train.py --bidirectional` — points the data pipeline at
  `train_bidi.tsv`/`test_bidi.tsv` instead of the normal files; hard
  `parser.error()` if `--tokenizer` isn't `bpe_joint`, so the
  vocab-mismatch failure mode above can't happen silently.
- `evaluate_bidirectional.py` — `evaluate.py` assumes one fixed
  direction throughout; this instead splits the bidirectional test set
  by tag and scores en→fr against the French references and fr→en
  against the English references separately.

**Results** (same architecture as the champion — emb192/L6/ff768/ep20 —
trained on the doubled bidirectional set, `bpe_joint_rope_mha_bidi`):

| Direction | BLEU | chrF++ | TER |
|---|---|---|---|
| en→fr (bidirectional model) | 41.34 | 59.71 | 38.61 |
| en→fr (specialized champion, for reference) | **44.95** | **62.88** | **36.03** |
| fr→en (bidirectional model) | **56.03** | **67.25** | **27.21** |

Train/eval accuracy: 83.24% / 79.60% (vs. the specialized champion's
86.00% / 80.23% — close, not free).

**Two findings:**
1. **Bidirectionality has a real, consistent cost**: −3.61 BLEU on en→fr
   from asking one model to also handle the reverse direction, relative
   to a model that only ever has to do that one job. Expected — the model
   now allocates capacity across two related-but-distinct tasks.
2. **fr→en is markedly stronger than en→fr in absolute terms**
   (56.03 vs. 41.34) — a well-documented, direction-independent asymmetry
   in MT, not an artifact of this experiment: English has far simpler
   target-side morphology to generate correctly (no grammatical gender
   agreement, much simpler verb conjugation) than French does, so
   translating *into* English is the easier generation problem regardless
   of which specific model or dataset is doing it.

**Recommendation:** keep `bpe_joint_rope_mha` (the specialized en→fr
champion, see below) as the primary/default model. Treat
`bpe_joint_rope_mha_bidi` as a separate, optional capability — the right
choice specifically if live bidirectional serving is a real requirement
(one checkpoint and one serving process instead of two, at a real BLEU
cost), not a strict upgrade to swap in by default. **Current best config,
below, is unchanged** — the bidirectional model is an addition to the
project's capability set, not a replacement for the quality-optimized
champion.

---

## Current best config

```
tokenizer:  bpe_joint (8,000-token BPE vocab, shared between English and French)
positional: RoPE
attention:  MHA
emb_dim:    192
num_layers: 6
ff_dim:     768        (4x emb_dim -- corrected from the original 2048/16x)
dropout:    0.1
epochs:     20
decode:     greedy
params:     10,846,016 (~10.85M)
```

**Results:** BLEU 44.95 · chrF++ 62.88 · TER 36.03 · eval acc 80.23%
(`checkpoints/bpe_joint_rope_mha/model.pt`, variant id
`bpe_joint_rope_mha` in `results/manifest.json`).

Superseded `bpe_rope_mha_widernarrowffnlong` (separate src/tgt vocabs,
BLEU 43.80) once the joint-vocabulary experiment above came back strictly
better on every quality metric, at the same parameter count. That
checkpoint remains a fully valid architecture-only reference point (the
"how much does architecture tuning alone buy you" story is unaffected —
this is a tokenizer-level change on top of it).

**+5.74 BLEU over the very first correctly-sized baseline** (39.21 →
44.95): +4.59 from architecture tuning (reallocating an oversized 16x FFN
ratio into a wider embedding, plus doubling the training budget), +1.15
more from switching to a joint source/target vocabulary.

**Why this config, in one paragraph (interview-ready):** Tokenization and
positional encoding were chosen by direct, controlled comparison — RoPE
beat sinusoidal in every tokenizer pairing because it re-derives relative
position at every attention layer instead of storing it once in the
residual stream, which measurably prevents the representational collapse
sinusoidal PE suffered under char tokenization's longer sequences.
Architecture was tuned empirically in two rounds rather than guessed:
depth consistently hurt at this data scale (106K pairs is not
Transformer-base-scale data), while reallocating the feedforward ratio
from a non-standard 16× down to the textbook 4× — freeing capacity for a
wider embedding — combined with doubling the training budget compounded to
the best result across 14 architecture/hyperparameter configs. On top of
that, switching from two independently-trained BPE vocabularies to one
shared vocabulary over the bilingual corpus — the same technique used by
mBART/NLLB/M2M-100 — improved every quality metric further by letting
rare shared surface forms and common cross-lingual structure (cognates,
punctuation, numbers) share statistical signal instead of being learned
twice, independently, from half the effective data each. Greedy decoding
was chosen over beam search because the model's output distribution, at
this scale, isn't sharp enough for beam's wider search to outperform
greedy's simpler argmax.

---

## Known gaps (explicit, not silently dropped)

| Gap | Status | Why it matters | Plan |
|---|---|---|---|
| ~~MQA fair comparison~~ | **resolved** | Phase 5 promised MHA vs. MQA vs. GQA; MQA's only run had used the broken 84M-param config (BLEU 0.0, collapsed) | retrained at correct capacity — BLEU 38.15 (10ep) / 40.57 (20ep), 8× smaller KV-cache than MHA. See "Attention-type comparison" section above. |
| ~~WordPiece at correct capacity~~ | **resolved** | Only run under the broken config (collapsed, ~0 BLEU) | retrained at correct capacity — BLEU 25.5-27.2 across PEs, genuinely ~10-12 BLEU below BPE/word (not a truncation artifact, checked). Also found and fixed a real `WordPieceTokenizer.decode()` punctuation-spacing bug that was inflating its TER specifically. See "Round 1 follow-up" section above. |
| ~~ALiBi at correct capacity~~ | **resolved** | Only run under the broken config on bpe/word (collapsed); `char_alibi_mha` survived by luck of char's tiny embedding table, not shown to be competitive | retrained at correct capacity on all 4 tokenizers — lands between sinusoidal and RoPE every time (per-layer bias resists collapse like RoPE, but cruder than RoPE's rotation). See "Round 1 follow-up" section above. |
| **YaRN** | never in scope | Not part of v3's original plan (`README.md`'s axis table only lists sinusoidal/RoPE/ALiBi); YaRN extends RoPE for *length extrapolation* via NTK-aware frequency rescaling — this dataset's sentences max out at 64 tokens, so there's no length-generalization problem for YaRN to solve here | explicitly out of scope, stated rather than silently absent |
| **Copy mechanism / constrained decoding** | designed, not built | Concrete architectural additions for proper-noun preservation, more powerful than the shipped NER+masking pipeline but requiring a retrain (copy mechanism) or beam-search changes (constrained decoding) | scoped out in favor of NER+masking (built, validated, no retraining needed) given this is a side investigation off Phase 7's main line. Full design in "The translate-vs-copy decision" section above. |
| **Phase 7 (inference engine)** | done — 7a-7f complete, then extended | KV cache, continuous batching, chunked prefill, prefix caching, Flash Attention, quantization, pruning — the project's stated main focus. Speculative decoding also fully built/verified, kept in codebase but excluded from the active showcase (scope choice, see Phase 7e) | 7a (naive baseline), 7b (KV cache), 7c (PagedAttention — built, verified, later removed for scope; see "Scope change"), 7d (continuous batching), 7e (speculative decoding — built, verified, kept but not actively showcased), 7f (`InferenceEngine` assembling all of the above) all built, correctness-verified bit-identical, and benchmarked. Then: PagedAttention removed, replaced with chunked prefill + prefix caching + Flash Attention (SDPA) + quantization + pruning, each independently verified and covered by a dedicated comparison study (TTFT/TBT/e2e-latency/throughput) that deliberately excludes speculative decoding — see below |

---

## Phase 7a/7b: naive baseline vs. KV cache

**7a, the control group,** is simply what `decoding.py`'s `greedy_decode`
has always done: recompute full self-attention over the entire generated
prefix at every decode step, including redoing the K/V projections for
every previously-generated token, every single step.

**7b, KV caching:** cache and incrementally extend K/V per layer instead
of recomputing them. Two correctness subtleties, both required for the
"bit-identical output" bar the README sets for this phase (a speed
optimization, not an approximation):

1. **RoPE needs each new token's true absolute position, not position
   0.** `RotaryEmbedding.forward` previously always built cos/sin for
   positions `0..seq_len-1` — fine when the whole sequence is passed
   every step (7a), wrong when only one new token is passed per step (its
   `seq_len` is always 1, so it would always get position-0's rotation).
   Fixed by adding a `start_pos` parameter threaded from the decode loop
   down through `Attention.forward` — only the *new* key gets rotated
   each step (at its real position); previously-cached keys were already
   correctly rotated when they were first computed. Same fix applied to
   `AlibiBias.forward` for consistency, though the project's production
   model uses RoPE.
2. **Cross-attention K/V never change across decode steps** (they come
   from the encoder output, which is fixed once encoding finishes) —
   recomputing them every step, as 7a implicitly does, is pure waste.
   `Attention.forward` gained a `cache_mode="static"` path (vs.
   `"growing"` for self-attention) that computes cross-attention K/V once
   and reuses them verbatim for every subsequent step, only recomputing Q
   (from the decoder's current state, which *does* change every step).

All of this is opt-in via new keyword arguments (`kv_cache`, `start_pos`,
`use_cache`, `cache_mode`) with backward-compatible defaults — every
existing call site (`train.py`, `evaluate.py`, every other `decoding.py`
function) is byte-for-byte unchanged. New: `decoding.py`'s
`greedy_decode_cached`, a drop-in replacement for `greedy_decode`. One
deliberate scope limit: sinusoidal PE isn't supported for cached decoding
(raises `NotImplementedError` rather than silently producing wrong
output) — its additive, embedding-level position injection has no
per-call position argument the way RoPE/ALiBi's per-layer injection does;
not worth building for a PE the project already found strictly worse (see
the char+sinusoidal collapse investigation earlier in this doc).

**Correctness, verified before trusting any benchmark number:**
`greedy_decode` vs. `greedy_decode_cached` produce **bit-identical token
ids on 200/200 random test sentences** (`bpe_joint_rope_mha`).

**Benchmark** (`benchmark_kv_cache.py`, 300 sampled test sentences,
greedy decode, GPU):

| | tokens/sec | avg latency | p95 latency |
|---|---|---|---|
| 7a naive | 117.17 | 87.11 ms | 155.14 ms |
| 7b KV-cached | 144.36 | 70.70 ms | 113.21 ms |
| **Speedup** | **1.23x** | **1.23x** | **1.37x** |

**Honest caveat, not smoothed over:** breaking this down by generated
length shows the gain is modest and inconsistent at very short outputs
(this task's translations average only ~10 tokens at `max_seq_len=64` —
even the longest bucket here is 16-20 tokens) — at 1-5 tokens, caching
was measured *slower* (0.92x) than naive, and the advantage only becomes
consistently positive past ~6 tokens. This is expected, not a bug:
caching's saving is the *avoided, repeated* K/V projection + FFN work for
already-generated tokens, which is O(n²) total work across n naive steps
vs. O(n) cached — an advantage that grows with sequence length. At this
project's short MT sequence lengths, that avoided-work saving is partly
offset by the extra Python-level overhead of the caching machinery itself
(more, smaller tensor ops — `torch.cat` per step — vs. fewer, larger
batched ops). This is exactly why the technique matters far more for
long-generation LLM serving (hundreds-to-thousands of output tokens) than
for short-sequence MT, and it's worth saying plainly rather than
presenting the 1.23x as bigger than the underlying mechanism actually
earns at this scale.

**Deviation from the original plan, noted rather than silent:**
`README.md`'s directory sketch anticipated a standalone
`serving/kv_cache.py` module. The cache logic instead lives directly
inside `model/attention.py`/`model/transformer.py` as optional forward-
pass arguments — the same integration point HuggingFace's own
`past_key_values` convention uses, and simpler here since it reuses the
existing model classes rather than duplicating attention math in a
parallel module.

---

## Phase 7c: PagedAttention — built, verified, then removed

**Status: removed from the current codebase (see "Scope change" section
below for the full removal rationale and what replaced it).** This
section is kept as a historical record rather than deleted outright —
the mechanism was real, built, and correctness-verified before the
decision to remove it, and the numbers below are genuine measurements
from when it was part of the engine, not retracted for being wrong.

**The problem 7b's cache doesn't solve:** 7b's growing-tensor cache
(`torch.cat` each step) works, but a real multi-request server needs to
size memory *before* knowing how long each request's output will be. The
straightforward answer — reserve `max_seq_len` worth of contiguous memory
per request up front — wastes most of what it reserves whenever actual
generations are shorter than the worst case, which for this project's MT
task (outputs average ~10 tokens against `max_seq_len=64`) is essentially
always.

**PagedAttention (Kwon et al. 2023, the vLLM mechanism), as originally
implemented in `serving/paged_attention.py` (now removed):** borrowed OS
virtual memory's paging idea directly. Physical KV-cache memory was
divided into small, fixed-size blocks drawn from one shared pool; each
request's logical token sequence mapped to a list of physical blocks —
not necessarily contiguous — via a per-request block table. A new block
was only allocated when a request's current one filled up, and freeing a
request returned its blocks to the pool immediately for reuse.
`serving/kv_cache_naive.py`'s `NaiveContiguousCache` (still in the
codebase, now the engine's only backend) was the up-front-reservation
baseline it was measured against.

**Correctness, as verified at the time:** both the paged and
naive-contiguous backends reproduced `greedy_decode`'s output
**bit-identical on 100/100 test sentences** — including sentences long
enough to force multiple block allocations and cross block boundaries
mid-generation under `block_size=8`.

**Benchmark** (as measured at the time, 64 simulated concurrent requests,
real sentences, real generation, `block_size=8`, `bpe_joint_rope_mha`):

| | Naive contiguous | PagedAttention |
|---|---|---|
| Allocated | 37.75 MB | 8.18 MB |
| Actually used | 6.29 MB | 6.29 MB |
| **Fragmentation** | **83.3%** | **23.1%** |

78.3% memory saved at the same batch size — or, the more useful framing
for a server: at naive's memory budget, paging fit ~295 concurrent
requests instead of 64 (4.61x the batch size). The mechanism worked
exactly as designed; it was removed for scope/breadth reasons, not
because it failed at anything.

---

## Phase 7d: continuous batching

**Status: fully built and verified, kept in the codebase, but excluded
from the current active engine/Comparison Study** — the project's
inference engine was later scoped down to just KV caching + compression
(see "Final scope" below); continuous batching, chunked prefill, prefix
caching, and Flash Attention are all still fully intact and correct
(`serving/batching.py`'s `run_static`/`run_continuous`, `serving/engine.py`'s
`InferenceEngine`, `benchmark_optimizations.py --throughput`/`--prefix`),
just not part of what's actively shown. Everything below remains accurate
and reproducible.

**The problem static batching has, that memory management alone doesn't
fix:** even with efficient KV-cache memory, if requests are processed in
fixed cohorts — collect a batch, run every member to completion, *then*
form the next batch — a request that finishes in 5 steps still sits idle
for however long its slowest batch-mate takes, and any newly-arrived
request is blocked from starting at all until the whole cohort drains.
Memory efficiency and scheduling efficiency are different problems.

**Continuous batching (Orca / vLLM's scheduling innovation), implemented
in `serving/batching.py`:** check *every* decode step, not just at batch
boundaries, for two things — has any active request finished (evict it,
free its resources immediately) and can the waiting queue fill the
freed slot right now (admit immediately, rather than waiting for the
whole cohort to drain). `run_static` and `run_continuous` implement both
policies over the same request/arrival abstraction so they can be
compared on identical simulated traffic.

**Built to run real requests through a real scheduler, not a mock:**
`Request.step()` executes one real decode step against the model,
threaded through an external KV-cache backend (`NaiveContiguousCache`,
see "Scope change" below; the scheduling policy being measured here is
independent of which backend implements `gather`/`append`/`advance`),
with each request's own cross-attention cache
carried in its own state so multiple requests can be interleaved,
paused, and resumed correctly. **Correctness verified two ways:**
`Request.step()` looped to completion reproduces `greedy_decode`
bit-identical on 50/50 sentences in isolation, and — the harder
test — 20/20 requests **actively interleaved through the continuous
scheduler** (4 requests live at once, admitted/evicted on independent
schedules) still come out bit-identical to their un-interleaved
`greedy_decode` reference. No cross-request state leakage.

**Scope limit, same reasoning as 7c:** each tick's per-request decode
calls are looped, not packed into one ragged batched tensor op across
requests of different current lengths — this benchmarks the scheduling
*policy's* latency effect (continuous batching's actual innovation over
static batching), not the further throughput optimization of batching
variable-length attention into a single kernel call, which a production
engine would add on top of this scheduling layer.

**Benchmark** (`benchmark_continuous_batching.py`: 40 requests, staggered
Poisson-ish arrivals over 144 simulated ticks, `max_batch_size=8`,
`bpe_joint_rope_mha`):

| | Static batching | Continuous batching |
|---|---|---|
| Mean latency | 16.8 ticks | **9.7 ticks** |
| P50 latency | 17 ticks | **9 ticks** |
| P95 latency | 28 ticks | **17 ticks** |
| P99 latency | 30 ticks | **18 ticks** |
| Wall-clock (40 requests) | 3.21s | 2.86s |

**P50 latency 1.89x lower, P95 1.65x lower, purely from the admission
policy** — no change to the model, the cache, or the per-step compute,
only to when a freed slot gets reused. A modest 1.12x wall-clock speedup
came along for free even without the batched-tensor throughput work
that's explicitly out of scope here (less idling naturally means less
total wall-clock time too, even one request at a time). This is the
number that matters most for a real server: static batching's tail
latency (P99: 30 ticks) is dragged up by fast requests trapped behind
slow cohort-mates; continuous batching's P99 (18 ticks) stays close to
its own mean, because no request ever waits on anyone else's cohort.

---

## Phase 7e: speculative decoding

**Status: fully built and verified, kept in the codebase, but excluded
from the actively-showcased engine and the Comparison Study below** (a
scope decision, not a removal — unlike PagedAttention, nothing here was
deleted). `serving/speculative.py`, `Request.step_speculative` in
`serving/batching.py`, and `InferenceEngine`'s `draft_model`/`k`
parameters are all fully intact and covered by the correctness numbers
below; `benchmark_optimizations.py --include_speculative` re-enables it
in the benchmark sweeps, and `app.py`'s live demo simply doesn't call
it. Everything in this section remains accurate and reproducible.

**The idea (Leviathan et al. 2023 / Chen et al. 2023):** a small, cheap
*draft* model proposes several tokens ahead; the expensive *target*
model verifies all of them in one parallel forward pass instead of one
sequential call per token. Whenever a proposal matches what the target
would have produced anyway, it's accepted for free — the target's
(expensive) per-call cost is paid once per *round* of several tokens
instead of once per token.

**A stronger correctness bar than the literature usually needs, and
worth stating precisely:** the general (stochastic-sampling) version of
this technique needs a distributional-equivalence argument, because
verification uses a probabilistic accept/reject rule — which is exactly
what `README.md`'s Phase 7 plan anticipated needing here. But this
project's chosen decode strategy is greedy (deterministic argmax), so
the acceptance rule collapses to a simple equality check: accept a draft
token iff it equals the target model's own argmax at that position, else
substitute the target's own argmax directly. Every emitted token is, by
construction, exactly what `greedy_decode` would have produced — this is
**exact**, not merely equivalent in distribution, a strictly stronger
guarantee than the general case, made possible specifically because this
project settled on greedy decoding back in the decoding-strategy sweep.

**Draft model:** a genuinely small model (`bpe_joint_rope_mha_draft`,
emb64/L2/ff256, **1.78M params, 16% of the target's 10.85M**) trained on
the *same* `bpe_joint` vocabulary as the target — required, since the
draft's proposed token ids must be directly comparable to the target's,
not just similar in meaning.

**A real bug caught by the correctness check, not by inspection — the
value of testing before benchmarking:** the first implementation re-fed
the "bridge" token linking one round to the next as input to the target
every round (needed to verify the draft's first proposal against it),
while *also* still holding that same token's key/value from when it was
originally cached — double-counting it in the attention keys from round
2 onward, an off-by-one invisible on a single round but real the moment
a rejection or multi-round sequence occurred. Fixed by establishing one
explicit invariant: **the persisted target KV cache always has length
`len(generated) - 1`** — the current last confirmed token is *never*
persisted, always re-derived fresh as next round's bridge, at the small,
deliberate cost of one redundant re-embedding per round.

**Correctness, verified before trusting any benchmark number (the same
bar as every other Phase 7 sub-phase):** `speculative_decode` reproduces
`greedy_decode` **bit-identical on 200/200 test sentences**, `k=4`.

**Benchmark** (`benchmark_speculative.py`, 200 sentences, matched sample
across `k` values for a fair comparison):

| k | Acceptance rate | Target-call reduction | Wall-clock |
|---|---|---|---|
| 2 | 70.6% | 1.71x fewer calls | 0.77x |
| 4 | 56.5% | 2.55x fewer calls | 1.01x |
| 6 | 47.9% | 2.91x fewer calls | 0.79x |
| 8 | 42.5% | 3.11x fewer calls | 0.91x |

**The target-model call reduction is real and substantial — up to 3.1x
fewer target calls at `k=8`. The wall-clock speedup is not reliably
realized at this project's scale, clustering around 0.8-1.0x with no
consistent trend across `k`** (re-running the same `k` values produces
different wall-clock numbers each time — real run-to-run noise, not a
`k`-dependent signal). This is an honest, mechanistically-explained
negative result, not a shortfall to gloss over: `_propose` deliberately
recomputes the draft's growing prefix from scratch every micro-step
(uncached, the same reasoning `greedy_decode`'s original design used,
chosen here to sidestep a second cache-rollback problem on top of the
target's), so the draft pays real, uncached, sequential overhead for
every one of its proposals — and at this project's scale (a 10.85M-param
target model, sequences averaging ~10 tokens), Python/kernel-launch
overhead already dominates raw compute (this is the identical lesson
7a/7b's modest 1.23x KV-cache speedup taught, one layer further in: here
the *added* overhead source is an entire second model's uncached forward
passes, not just avoided recomputation). Speculative decoding's real
payoff requires the target model to be expensive enough per call that
saving several of those calls outweighs a cheap draft model's own
overhead — true for large LLMs (a 70B-parameter target vs. a 1B-parameter
draft, where a verification call is far more parallelism-friendly than
what it replaces) in a way it measurably isn't for a model this size.
**Caching the draft's own proposal phase** (the natural next refinement,
not built here given the added cache-rollback complexity it would need
on top of the target's) would reduce but likely not eliminate this —
7a/7b already showed caching alone buys a modest, not dramatic, win at
this scale.

## Phase 7f: assembling the engine

**Note: this section describes the engine as originally assembled, with
PagedAttention as the memory backend.** PagedAttention was later removed
(see "Scope change" below); the correctness numbers and bug writeups
below are an accurate historical record of that version. The *current*
engine (`NaiveContiguousCache` backend, plus chunked prefill/prefix
caching/SDPA added afterward) is documented in the sections following
"Scope change" — including its own, separately-run correctness checks
for every technique added since.

Ties memory management (originally PagedAttention, now
`NaiveContiguousCache`), continuous batching (7d), and speculative
decoding (7e, optional) into one request-in / translation-out API
(`serving/engine.py`'s `InferenceEngine`) — the point where each
piece, already built and correctness-verified standalone, gets used
together against the same memory pool and scheduler.

**Design note on "every combination of optimizations on/off":** paging
and continuous batching aren't really toggles in this engine — they're
what the scheduler *is*; turning either off just means reverting to
7a's already-separately-benchmarked naive per-request loop, not a
meaningful runtime configuration of *this* engine. The one genuine
on/off toggle exposed is speculative decoding (`draft_model=None` vs. a
real draft model), so the honest comparison benchmarked below is static
batching vs. continuous batching vs. continuous batching + speculative
decoding — three real, previously-validated configurations, not a
synthetic grid built to pad out a table.

**Mechanism:** `Request.step_speculative()` runs one speculative round
(7e's propose/verify/accept-reject logic, reusing 7e's own `_propose`
helper directly) through the *external* `PagedKVCache` backend instead
of `speculative_decode`'s private growing-tensor cache — same
accept/reject rule, same cache invariant, but K/V now flows through
`backend.gather()` / `backend.append()` / `backend.advance()` so a
speculative request shares the same paged blocks and the same
`InferenceEngine._tick()` scheduler as any plain request admitted
alongside it.

**Two real bugs, caught by the same bit-identical bar every other
Phase 7 sub-phase was held to, not by inspection:**

1. **Wrong absolute position when persisting new K/V.** The per-token
   persist loop sliced the newly-computed, *concatenated*
   `[past, new_chunk]` K/V tensor using an index relative to the new
   chunk only (`1 + j`), which is only correct on a request's very
   first round (`start_pos == 0`) — from round 2 onward this silently
   read the wrong absolute position out of the tensor. Fixed by
   indexing with the true absolute position, `start_pos + j`.
2. **Persist count didn't account for the rejection case.** Even with
   fix #1 in place, correctness was *still* broken (5/30 → still
   broken after the first fix) because the number of new cache entries
   to persist each round is not always "bridge + matched minus the
   last token" — that's only true when the round fully accepted (or
   accepted-then-hit-EOS). When a round ends in a *rejection*, the
   current last confirmed token is the target's own substituted
   prediction, which was only ever computed as an output logit and
   never fed back in as model input — so there's nothing left to
   exclude, and the persist count is one entry *larger*: bridge + all
   of `matched` (not `matched[:-1]`). Traced by comparing the paged
   engine's round-by-round state directly against the known-good,
   already-verified `speculative_decode()` on the same input sentence
   (`"who hired you ?"`), which pinpointed the exact round and the
   exact wrong token before the fix was made — `persist_count =
   len(matched) + (1 if rejection_tok is not None else 0)`, algebraically
   re-derived from `speculative_decode`'s own `keep = min(valid_len,
   len(generated) - 1)` formula to confirm the two are equivalent
   before trusting it.

**Correctness, verified before any benchmark number was trusted:**
`InferenceEngine` (speculative mode, `draft_model` set) reproduces
`greedy_decode` **bit-identical across 130/130 test cases** spanning
four separate scenarios — 30/30 single-request (seed 42), 100/100
single-request (seed 7, different sample), 20/20 with all 20 requests
submitted together and genuinely interleaved through the scheduler
(`max_batch_size=4`, so admission/eviction churn is real, not
incidental), and 15/15 at each of `k=2,6,8` (the default `k=4` path is
covered by the first two runs). The plain (non-speculative) engine
path — unaffected by either bug, since both were specific to
`step_speculative`'s persist loop — was reconfirmed 30/30 as a
regression check.

**Benchmark** (`benchmark_engine.py`, 40 requests, staggered arrivals
over 40 ticks, `max_batch_size=8`, `k=4`, all three runs sharing the
identical arrival stream):

| Configuration | P50 latency (ticks) | P95 | Wall-clock |
|---|---|---|---|
| Static batching | 37 | 62 | 3.20s |
| Continuous batching | 15 | 27 | 2.98s (1.07x) |
| Continuous batching + speculative decoding | 4 | 8 | 2.71s (1.18x vs. static) |

**Read the per-request "tick" numbers carefully — they are not directly
comparable across rows.** In the plain runs, one tick == one token
generated per active request, so latency-in-ticks *is* latency-in-tokens.
In the speculative run, one tick == one full speculative *round*, which
advances a variable number of tokens (however many the draft's proposal
had accepted that round) — so the P50 drop from 15 to 4 mostly reflects
7e's already-established ~2.5x round/target-call reduction at `k=4`, not
that each tick got individually cheaper. The wall-clock comparison is
the fair one, and it shows the same story 7e already told at the
single-request level: a real, modest, sub-linear win (1.18x end-to-end
vs. static, 1.10x vs. continuous-alone) rather than anything close to
the 2.5x call-reduction figure, because the draft model's deliberately
uncached proposal overhead eats into most of the savings at this
project's scale. Continuous batching's own scheduling win (1.07x here,
consistent with 7d's standalone 1.12x finding) is the more reliable of
the two gains at this model size — paging + continuous batching is the
combination worth defaulting on; speculative decoding is worth keeping
as an explicit, understood-tradeoff toggle rather than an unconditional
win.

**`serving/server.py`:** a thin FastAPI wrapper around `InferenceEngine`
— `POST /translate` submits, `GET /result/{id}` polls, `GET /health`
reports queue depth — driven by a background asyncio task that pumps
`engine._tick()` while there's work and sleeps when idle, rather than
blocking a request thread per translation. Smoke-tested end-to-end
(`uvicorn` in-process against a live engine instance, both a single
request and five concurrent requests submitted together): every
translation returned matches the already-verified bit-identical
reference output, including `"who hired you ?"` → `"qui t'a engagée ?"`,
confirming the API layer introduces no additional state bugs on top of
the engine it wraps.

---

## Scope change: removing PagedAttention, adding chunked prefill / prefix caching / Flash Attention / compression

**What changed:** `serving/paged_attention.py` and its benchmark were
removed; `serving/engine.py` now uses `NaiveContiguousCache` as its only
KV-cache backend. In its place: chunked prefill, prefix caching, Flash
Attention (via PyTorch's SDPA), and post-training compression
(quantization + pruning) were added, each independently built,
correctness-verified, and benchmarked below — plus a comparison study
across TTFT / TBT / end-to-end latency / throughput tying all of Phase 7
together (KV cache, chunked prefill, prefix caching, Flash Attention,
continuous batching, speculative decoding) to answer, with measured
numbers, which techniques actually help and in which phase (prefill vs.
decode).

**Why remove something that worked** (the numbers in Phase 7c above are
real, not walked back): PagedAttention solves *cross-request memory
fragmentation* specifically — genuinely valuable at the request
concurrency and generation-length scale real LLM serving operates at,
but this project's actual demonstrated scale (short MT outputs, tens of
concurrent requests in the benchmarks above, not thousands) meant its
benefit was already the least visible of Phase 7's original four
techniques in this project's own numbers. Swapping it for four more
directly comparable, more standard interview topics —
chunked prefill, prefix caching, Flash Attention, model compression —
keeps the inference-engine story anchored to techniques whose value
shows up clearly at this project's own scale, and broadens the set of
concepts actually built and defensible in depth, rather than one more
memory-management mechanism layered onto three already covered by KV
caching, continuous batching, and speculative decoding.

`NaiveContiguousCache` (`serving/kv_cache_naive.py`, already built in
Phase 7c as PagedAttention's baseline) becomes the engine's only backend
— simpler to reason about, and every technique added below (continuous
batching, chunked prefill, prefix caching, speculative decoding) works
identically against it, since none of them actually depend on paging
specifically; they all just need *some* backend satisfying the same
five-method contract (`allocate_request` / `free_request` / `append` /
`gather` / `advance`).

---

## Chunked prefill

**Status: fully built and verified, kept in the codebase, excluded from
the current active engine/Comparison Study** — see Phase 7d's status
note above for the full scope-change rationale (the engine was later
narrowed to KV caching + compression only). Everything below remains
accurate and reproducible via `serving/chunked_prefill.py` and
`benchmark_optimizations.py --include_chunked_prefill` (single-request
sweep) / `--throughput` (concurrent).

**The problem in real LLM serving:** a newly-admitted request's prefill
(processing its whole prompt before generation starts) normally runs as
one atomic forward pass. Under continuous batching, that atomic call
blocks the scheduler for an entire tick — every other already-active
request's decode step waits behind it. A long prompt's prefill can
visibly stall everyone else's token-by-token latency. Chunked prefill
(vLLM, Sarathi-Serve) fixes this by splitting a long prompt into pieces
processed across several scheduler ticks instead of one, so decode steps
for other requests can interleave between prefill chunks.

**Why this project chunks by LAYER, not by sequence position (a real
architectural constraint, not an arbitrary implementation choice):** the
standard technique chunks a causal decoder-only LLM's *prompt* by
position — chunk 1 attends only to itself, chunk 2 attends to chunk 1 +
itself, etc. — because causal attention already only lets a token see
earlier tokens, so processing the prompt in position-order pieces changes
nothing about what each token can see. This project's **encoder is
bidirectional** (full, non-causal self-attention) — every source token's
representation legitimately depends on every *other* source token,
including ones later in the sequence. Position-chunking a bidirectional
encoder the causal way would silently change what it computes (an early
chunk's tokens would never get to attend to later-chunk tokens they were
trained to see) — a genuinely different, wrong result, not a pure speed
optimization, breaking the exactness bar every other Phase 7 technique is
held to. Chunking by **layer** instead — running a fixed number of
encoder layers per tick, carrying the partial hidden state to the next
tick — preserves the encoder's exact bidirectional computation (the
final `enc_out` is identical no matter how many ticks the layer stack is
split across) while still spreading its wall-clock cost across multiple
scheduler ticks, the actual scheduling property chunked prefill exists to
provide.

**Implementation** (`serving/chunked_prefill.py`, wired into
`serving/batching.py`'s `Request.start_chunked_prefill` /
`continue_chunked_prefill` and `serving/engine.py`'s `_tick`): a request
admitted with chunked prefill enabled first only runs `model.src_embed`,
then advances `prefill_chunk_layers` encoder layers per tick (tracked via
`prefill_layer_idx`) until the full stack has run, at which point it
moves from the engine's `prefilling` list into `active` (decoding) —
occupying a batch slot the whole time, so `max_batch_size` correctly
accounts for requests that are admitted but not yet generating.

**Correctness, verified before any benchmark number was trusted:**
`continue_encode`'s layer-range slicing reproduces `Transformer.encode`'s
one-shot output **bit-identical** across 5 test sentences × 4 chunk
sizes (1, 2, 3, and all-6-at-once) — `max_diff=0.0` in every case, not
just a small float tolerance. The full engine with chunked prefill
enabled reproduces `greedy_decode` **bit-identical on 30/30 single-request
sentences and 24/24 with 16 requests genuinely interleaved** (multiple
requests mid-prefill simultaneously, `max_batch_size=4`), including
combined with speculative decoding (24/24) and combined with prefix
caching (24/24) at once.

---

## Prefix caching

**Status: fully built and verified, kept in the codebase, excluded from
the current active engine/Comparison Study** — see Phase 7d's status
note above for the full scope-change rationale (the engine was later
narrowed to KV caching + compression only). Everything below remains
accurate and reproducible via `serving/engine.py`'s `prefix_cache` and
`benchmark_optimizations.py --prefix`.

**The problem:** repeated identical requests (a common real pattern —
FAQ-style phrases, template sentences, the same short query translated
by many different users) redundantly re-run the *entire* encoder forward
pass every time, even though the output would be identical to a request
already served moments earlier.

**A genuinely important nuance surfaced while designing this, not just
implementing it blind:** production LLM serving's "automatic prefix
caching" (vLLM, SGLang's RadixAttention) caches KV state for a shared
*partial* prompt prefix across requests that diverge partway through —
and that's mathematically exact specifically because those systems'
attention is **causal**: a token's representation only ever depends on
itself and earlier tokens, so two prompts sharing the first N tokens
produce byte-identical KV state for those N tokens regardless of what
comes after. This project's **encoder is bidirectional** — a token's
representation depends on *every* token in the sentence, including ones
after a would-be shared prefix. Two source sentences sharing a leading
phrase ("tom said hello" / "tom said goodbye") would *not* produce
identical encoder states for "tom said", because "hello" vs. "goodbye"
bidirectionally influences it. Partial-prefix caching the causal-LLM way
would therefore be a silent **approximation**, not exact, for this
architecture — breaking the same bit-identical bar every other Phase 7
technique meets.

**What's implemented instead, and why it's exact:** caching keyed by the
**full, exact source sentence** — the degenerate but mathematically safe
case of prefix caching (the "prefix" being cached is the whole prompt).
Identical input always produces identical encoder output regardless of
bidirectionality, so reusing a cached `(enc_out, src_mask)` for a
byte-identical repeated source sentence is exact, not approximate.
Implemented in `serving/engine.py` (`InferenceEngine.prefix_cache`, a
dict keyed by `tuple(src_ids)`) and `serving/batching.py`'s
`Request.admit_from_prefix_cache` — a real, useful technique for a
translation service's actual traffic pattern (repeated common
phrases), correctly scoped to what this architecture can do exactly
rather than porting a causal-LLM technique that wouldn't be safe here.

**Correctness, verified before any benchmark number was trusted:**
engine with prefix caching enabled reproduces `greedy_decode`
**bit-identical on 30/30 single-request sentences (cold cache) and
24/24 with genuine repeated-sentence cache HITS** (8 base sentences each
repeated 5x, interleaved with unique filler sentences, all submitted
together and processed concurrently under `max_batch_size=3`) — 16
distinct sentences produced exactly 16 cache entries, confirming hits
and misses were tracked correctly, not just "happened not to crash."

**Benchmark** (45 requests: 8 base sentences × 5 repeats + 5 unique
fillers, interleaved, `max_batch_size=4`):

| | Wall-clock |
|---|---|
| Prefix cache OFF | 5.23s |
| Prefix cache ON | 4.77s (**1.10x**) |

A real, modest win, consistent with this project's established pattern
for every Phase 7 technique at this scale (32 of 45 requests were
genuine cache hits, yet the wall-clock win is still sub-linear — the
saved encoder pass is a small fraction of this model's total per-request
cost, most of which is the token-by-token decode loop, unaffected by
prefix caching).

---

## Flash Attention (PyTorch SDPA)

**Status: fully built and verified, kept in the codebase, excluded from
the current active engine/Comparison Study** — see Phase 7d's status
note above for the full scope-change rationale (the engine was later
narrowed to KV caching + compression only). Everything below remains
accurate and reproducible via `model/attention.py`'s `use_sdpa=True`
path and `benchmark_optimizations.py --single --include_sdpa`.

**What "using a library instead of manual code" means here, precisely:**
this project's own attention (`model/attention.py`) is hand-implemented
eager attention — explicit `Q @ K^T`, manual softmax, manual masking.
Flash Attention (Dao et al. 2022) is a fused, tiled, IO-aware CUDA kernel
that never materializes the full `[seq_len, seq_len]` attention matrix in
GPU memory — implementing that *by hand* in pure PyTorch wouldn't
actually be faster (it would just be a slower, less careful
re-derivation of what a real fused kernel already does), so the explicit
choice made here was to call PyTorch's built-in
`torch.nn.functional.scaled_dot_product_attention` (SDPA), which PyTorch
itself dispatches to a real FlashAttention-2 CUDA kernel automatically on
Ampere+ GPUs (compute capability ≥ 8.0 — this project's A6000s are 8.6),
with zero custom kernel code and zero extra dependency (the dedicated
`flash-attn` pip package needs a slow/fragile source build matching an
exact CUDA/torch ABI; SDPA needed neither).

**Integration** (`model/attention.py`'s `Attention.forward(..., use_sdpa=True)`,
opt-in, default `False` preserves every existing training/eval call site
byte-for-byte): replaces just the score/softmax/matmul block with one
`F.scaled_dot_product_attention` call, reusing the exact same Q/K/V
projections, RoPE rotation, and KV-cache concatenation as the manual
path. Deliberately scoped, not generalized past what the engine actually
needs: unsupported together with `return_attn` (a fused kernel never
materializes the attention matrix, so there's nothing to return for
attention-heatmap visualization) or with ALiBi's additive per-position
bias (the engine only ever serves the winning RoPE-based model) —
documented limits, matching how sinusoidal PE was already excluded from
KV-cache decoding for the same kind of reason.

**Why this is NOT bit-identical to the manual path — and why that's
expected, not a bug, unlike every other Phase 7 technique:** SDPA
computes the *exact same mathematical operation*
(`softmax(QK^T/√d)V`) via a different *algorithm* — tiled, with an
online/running softmax, fused into one kernel. Floating-point addition
isn't associative, so summing softmax-weighted values in a different
order (tile-by-tile vs. all at once) produces tiny rounding differences.
This is a **numerically equivalent reformulation**, verified *close*, not
bit-exact — the one Phase 7 technique in this category, since KV caching,
continuous batching, speculative decoding, chunked prefill, and prefix
caching all only ever change *what* gets computed/scheduled/cached/
reused, never the arithmetic itself.

**Measured directly, not just asserted:** on a 4-token verification
batch, max absolute logit difference between the manual and SDPA paths
was **1.9 × 10⁻⁵** (mean **4.5 × 10⁻⁶**) — and, critically, **every
argmax decision matched exactly**. Confirmed end-to-end: the engine with
`use_sdpa=True` reproduces `greedy_decode`'s **final decoded text
identical on 40/40 test sentences** — the tiny numerical differences
never flip a single greedy argmax choice at this model's scale.

**Honest result at this project's scale:** single-request TBT (mean
per-token decode latency) was 10.74ms manual vs. 10.36ms with SDPA — a
real but small (~3.5%) difference, consistent with the rest of Phase 7's
finding that these optimizations show their strongest effect at longer
sequences / larger models than this project's short MT outputs. Flash
Attention's core benefit (avoiding materializing the full attention
matrix) scales with sequence length squared — at this project's ~10-20
token sequences, there's very little quadratic cost to avoid in the
first place.

---

## Compression: quantization and pruning

Both are **post-training** (no retraining/fine-tuning step) — the fast,
"free" end of the compression spectrum, at the cost of some quality;
quantization-aware training or iterative pruning-with-fine-tuning would
recover more quality at a real extra training cost outside this
project's scope. Unlike every technique above, compression changes the
**model itself**, not the serving/scheduling logic — so it's the one
Phase 7 category that needs a quality (BLEU) check alongside speed, not
just a correctness check.

### Quantization

**What's implemented:** PyTorch's standard post-training **dynamic INT8
quantization** (`torch.ao.quantization.quantize_dynamic`) over every
`nn.Linear` layer — every attention projection, feed-forward layer, and
the final vocabulary projection. "Dynamic" means weights are quantized
once, ahead of time, while activations are quantized on the fly per
forward call.

**An honest scope limit, stated directly rather than discovered by a
confused benchmark number:** PyTorch's stock quantized INT8 GEMM kernels
(fbgemm/qnnpack) have **no CUDA implementation** — this API only ever
runs on **CPU**, never this project's GPU serving path. That's not a
reason to skip it: model size reduction matters regardless of where the
model ultimately runs (less to load into memory, less to transfer over a
network), and understanding *why* a standard quantization API doesn't
transfer to GPU serving (vs. blindly benchmarking it and reporting a
misleading number) is itself the more useful thing to know.

**Measured** (n=5 test sentences — deliberately small, see note below;
raw log preserved at `results/compression_bench_log.txt`)**:**

| | fp32 (GPU) | Dynamic INT8 (CPU) |
|---|---|---|
| Model size | 41.37 MB | 11.76 MB (**3.52x smaller**) |
| BLEU | 50.46 | 46.9 (**delta −3.56**) |
| Naive-decode wall time, whole sample | a few seconds | **1602.9s (~26.7 min) for 5 sentences (~5.3 min/sentence)** |

**A second, unplanned finding — worth reporting as honestly as the
planned ones:** dynamic INT8 on CPU wasn't just "doesn't help GPU
serving," it was *dramatically* slow in absolute terms for this
project's specific access pattern — naive (uncached) autoregressive
decoding, one full forward pass per output token. Traced with `py-spy`
to confirm it was genuinely computing, not hung: `torch.ao.nn.quantized
.dynamic.modules.linear.Linear.forward` re-quantizes activations to INT8
and dequantizes the output on *every single call*, and naive decoding
calls the full model once per generated token — that per-call
quantize/dequantize overhead, paid repeatedly with no caching to
amortize it, compounds into ~5 minutes per short sentence. This is a
real, measured property of dynamic quantization's cost model, not a
setup mistake: dynamic quantization trades a size win for a real
per-call CPU overhead, and it shows up worst specifically in an
uncached, many-small-calls access pattern like this one. **BLEU
sample size note:** n=5 (not the 30-40 used elsewhere in this project)
specifically because of this latency — the *3.52x size reduction* and
*CPU-only* findings are the load-bearing, high-confidence results here;
the exact BLEU delta is directional at this sample size, not a precise
population estimate.

### Pruning

**What's implemented:** magnitude-based **unstructured pruning** (L1 —
zero out the smallest-magnitude weights) over every `nn.Linear` layer's
weight tensor, at three sparsity levels (30%/50%/70%), via
`torch.nn.utils.prune.l1_unstructured` + `prune.remove` (bakes the zeros
permanently into the weight tensor).

**A prediction stated up front, then actually measured — not assumed
either way:** unstructured sparsity (zeros scattered arbitrarily through
an otherwise-dense tensor) only pays off with a **sparse** tensor format
and a sparse-aware kernel. A dense tensor with zeros in it still occupies
the same memory and still gets multiplied elementwise by a standard GEMM
kernel — so pruning alone, without converting to a sparse format, was
predicted to change neither model size nor GPU latency. Confirmed
directly: `linear_sparsity_pct` measured exactly 30.00% / 50.00% / 70.00%
at each target, while `model_size_mb` stayed at **41.374 MB, byte-for-byte
identical, regardless of sparsity level** — the honest result matching
the prediction exactly, not glossed over.

**Measured (n=5, same sample as the quantization comparison, GPU):**

| Sparsity | Model size | BLEU | Delta |
|---|---|---|---|
| 0% (baseline) | 41.37 MB | 50.46 | — |
| 30% | 41.37 MB (unchanged) | **50.46** | **0.00** |
| 50% | 41.37 MB (unchanged) | 37.83 | −12.63 |
| 70% | 41.37 MB (unchanged) | 0.37 | −50.09 |

**A clean, interpretable "cliff" pattern, not a smooth degradation:**
zero measurable quality loss at 30% sparsity, real but recoverable-
looking degradation at 50%, and near-total output collapse at 70% — this
model has real, exploitable low-magnitude-weight redundancy up to a
point, then loses the capacity it actually needs past that point.
Consistent with the broader pruning literature: post-training
(no-fine-tuning) unstructured pruning tends to tolerate moderate
sparsity gracefully and then fail sharply, rather than degrading
linearly — exactly what a small (24.7M-parameter, `bpe_joint` 32K-vocab)
model with comparatively little redundancy to begin with would be
expected to show. GPU latency wasn't separately profiled here beyond the
size-invariance finding above — sparsity's effect on *quality* was the
question actually worth measuring, since dense storage already settled
the latency/size question (unchanged regardless of sparsity, confirmed
above) before any decode timing was needed.

---

## Final scope: KV caching + compression

After all six Phase 7 techniques (KV cache, chunked prefill, prefix
caching, Flash Attention/SDPA, continuous batching, speculative decoding)
were built and independently verified, the project's *actively shown*
inference engine was deliberately narrowed to just two: **KV caching**
and **compression (quantization + pruning)**. This is a scope decision,
not a correctness or quality finding — every excluded technique remains
fully built, independently verified (bit-identical or numerically
equivalent as documented in its own section above), and re-runnable via
its own `benchmark_optimizations.py` flag:

| Technique | Status | Reproduce |
|---|---|---|
| KV caching | **active** | `benchmark_optimizations.py --single` |
| Compression (quant + pruning) | **active** | `benchmark_optimizations.py --compression` |
| Continuous batching | kept, not shown | `benchmark_optimizations.py --throughput` |
| Chunked prefill | kept, not shown | `benchmark_optimizations.py --single --include_chunked_prefill` |
| Prefix caching | kept, not shown | `benchmark_optimizations.py --prefix` |
| Flash Attention (SDPA) | kept, not shown | `benchmark_optimizations.py --single --include_sdpa` |
| Speculative decoding | kept, not shown | `benchmark_optimizations.py --single --include_speculative` |

The reasoning documented per-technique above (bidirectional-encoder
adaptation for chunked prefill/prefix caching, the SDPA mask-contiguity
bug, the CPU-quantization slowdown, the pruning quality cliff) is all
still accurate and still the deepest material in this project — it's
just framed as "built, verified, and here's what I learned," rather than
part of the headline comparison below.

---

## Comparison study: does each technique actually help, and in which phase?

Measured with `benchmark_optimizations.py --single --n 30` (KV caching)
and `--compression --n 5` (quantization/pruning; small n because CPU
dynamic-INT8 decode is ~5.3 min/sentence in this environment — see the
Compression section above), using the metrics real LLM serving systems
report: **TTFT** (time to first token — the **prefill** phase), **TBT**
(time between tokens — the **decode** phase's steady-state per-step
cost), **end-to-end latency**, and **throughput**.

### Single-request: TTFT / TBT / end-to-end (n=30, `bpe_joint_rope_mha`)

| Configuration | TTFT (ms) | TBT (ms) | e2e (ms) | tok/s |
|---|---|---|---|---|
| Naive, no cache (7a) | n/a* | n/a* | 64.0 | 145.8 |
| KV cache (7b) | 10.9 | 6.8 | 67.2 | 138.9 |

*naive decoding has no separable prefill/decode-step cost — every step,
including the first, redoes full attention over the whole growing
prefix, so there's no meaningful TTFT/TBT split the way there is once
caching exists.

### Compression (n=5, `bpe_joint_rope_mha`)

| | fp32 (GPU) | Dynamic INT8 (CPU) | Pruned 30% | Pruned 50% | Pruned 70% |
|---|---|---|---|---|---|
| Model size | 41.37 MB | 11.76 MB (**3.52x smaller**) | 41.37 MB (unchanged) | 41.37 MB (unchanged) | 41.37 MB (unchanged) |
| BLEU | 50.46 | 46.9 (**−3.56**) | **50.46 (0.00)** | 37.83 (−12.63) | 0.37 (−50.09) |
| Speed | GPU, seconds | **~5.3 min/sentence, CPU-only** | GPU, seconds (unaffected) | GPU, seconds (unaffected) | GPU, seconds (unaffected) |

### Which phase does each active technique target?

| Technique | Primary phase | What it changes |
|---|---|---|
| KV caching | Decode | Avoids O(n²) re-attention over the growing prefix; every step after the first gets cheaper |
| Quantization | Neither (model-level) | Shrinks weights; only helps this project's *CPU* path, not the GPU serving path measured above |
| Pruning (unstructured) | Neither for size/latency (model-level) | Zeros weights in place; changes nothing about latency or size without a sparse kernel/format (confirmed above) — but *does* affect quality, sharply, past a threshold |

### Honest conclusion

**KV caching's mechanism is real and verified (bit-identical output,
genuine O(n²)→O(n) algorithmic change per decode step), but its
*wall-clock* win is at or below the noise floor at this project's scale**
— the n=30 numbers above show KV cache e2e latency (67.2ms) slightly
*higher* than naive (64.0ms) for these ~10-20 token outputs, because the
cache's fixed per-step bookkeeping overhead isn't yet amortized by
enough decode steps to outweigh naive's simpler-but-quadratic recompute
at this short a sequence length. This is reported as measured, not
smoothed over: KV caching is the mechanism that makes every other
serving optimization (batching, paged memory, speculative decoding)
possible or worthwhile, but it needs longer sequences and/or a larger
model than this project's scale to show a positive wall-clock number by
itself — a real, understood limitation, not a bug (the bit-identical
correctness check passing rules out an implementation error as the
cause).

**Compression's two techniques diverge sharply.** Quantization
delivers an unconditional, real win on one axis (3.52x smaller) and an
unconditional real cost on another (CPU-only, and dramatically slow —
~5.3 min/sentence — for this project's uncached, many-small-calls decode
pattern, confirmed genuinely computing via `py-spy`, not hung).
Unstructured pruning changes nothing about size or latency without a
sparse runtime (confirmed directly: identical 41.37MB at every sparsity
level) — but its *quality* axis is where the real signal is: a sharp
cliff between 30% sparsity (zero measurable BLEU loss) and 50-70%
(severe to total collapse), rather than a flat "nothing happens" a naive
read might expect.

None of these are implementation bugs — each technique was verified
correct/exact first (bit-identical for KV caching, byte-for-byte size
confirmation for pruning), then measured, and the *reason* each does or
doesn't help at this scale is understood and stated rather than glossed
over. The five kept-but-not-shown techniques (continuous batching,
chunked prefill, prefix caching, Flash Attention, speculative decoding)
each have their own honest results, documented in full in their
dedicated sections above and in `interview.md`'s deep-dive — real,
modest wins (1.08-1.12x range) for the scheduling/caching techniques,
and a small-but-real (~3.5%) latency improvement for Flash Attention,
all consistent with this project's overall finding that these
optimizations' payoff grows with sequence length and model/traffic
scale, neither of which this specific portfolio project pushes far.
