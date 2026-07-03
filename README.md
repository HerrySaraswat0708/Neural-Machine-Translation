# Neural Machine Translation: Tokenization x Positional Encoding x Decoding

A from-scratch Transformer (Vaswani et al., "Attention Is All You Need") for
English -> French translation, built to compare three axes of NMT design
choices side by side rather than ship a single fixed model:

- **Tokenization**: word-level, byte-level BPE, and character-level
- **Positional encoding**: sinusoidal (additive) and RoPE (rotary, applied to Q/K in attention)
- **Decoding**: greedy, beam search, and top-k / top-p sampling

That's 3 x 2 = 6 trained model variants, each evaluated with corpus-level
BLEU (`sacrebleu`) on a fixed held-out test split, plus an interactive
Streamlit app to translate, visualize cross-attention as a heatmap, and
compare configs side by side.

## Architecture

Everything in `model.py` is implemented from scratch: scaled dot-product
multi-head attention, sinusoidal positional encoding, a `RotaryEmbedding`
module (`rope.py`), position-wise feed-forward layers, residual+LayerNorm
("Add & Norm"), and a 6-layer encoder / 6-layer decoder stack with causal
masking on the decoder's self-attention.

**RoPE is applied to encoder self-attention and decoder masked
self-attention, but deliberately not to decoder cross-attention.** RoPE
encodes *relative* position between a query and a key, which is only
meaningful when both live in the same sequence's coordinate system. In
cross-attention the query is a target (French) position and the key is a
source (English) position -- two independently-indexed sequences with no
aligned notion of "distance." Rotating both would inject a spurious,
arbitrary bias from raw index arithmetic. Positional information for
cross-attention already comes from the (RoPE-aware) encoder output itself.

## Repository layout

```
data/               normalization + train/test split builder (data/splits/, gitignored)
tokenization/        word / BPE / char tokenizers behind one shared interface
model.py             Transformer, MultiHeadAttention, RoPE plumbing
rope.py              RotaryEmbedding + apply_rotary_pos_emb
decoding.py          greedy / beam / top-k / top-p decoders + attention capture
train.py             config-driven training entrypoint (one variant per run)
evaluate.py          corpus BLEU per variant/decoding strategy -> results/manifest.json
inference.py         shared model+tokenizer loading for evaluate.py and app.py
viz.py               Plotly attention heatmap (sequential blue ramp)
app.py               Streamlit frontend: Translate tab + Compare tab
configs/defaults.json shared hyperparameters (kept identical across all 6 variants)
checkpoints/         trained weights + tokenizer vocabs (gitignored, reproducible via train.py)
results/manifest.json per-variant metadata + BLEU scores (drives the frontend dropdowns)
notebooks/Trans.ipynb original prototyping notebook, kept for history
```

## How to run

```bash
pip install -r requirements.txt

# 1. Build the fixed train/test split (once)
python data/prepare_data.py

# 2. Train all 6 variants (each ~20 epochs over the full split by default)
python train.py --tokenizer word --pe sinusoidal
python train.py --tokenizer word --pe rope
python train.py --tokenizer bpe  --pe sinusoidal
python train.py --tokenizer bpe  --pe rope
python train.py --tokenizer char --pe sinusoidal
python train.py --tokenizer char --pe rope

# 3. Evaluate BLEU (greedy by default; --all-decodes runs beam/top-k/top-p too)
python evaluate.py --all --decode greedy
python evaluate.py --variant word_sinusoidal --all-decodes

# 4. Launch the frontend
streamlit run app.py
```

For a quick smoke test before committing to a full run: add `--subsample 2000
--epochs 1` to any `train.py` call.

## Design notes worth knowing before reading the code

- **All 6 variants share identical hyperparameters** (`configs/defaults.json`:
  emb_dim=128, heads=8, ff_dim=2048, layers=6, batch=32, epochs=20, Noam
  warmup, label smoothing=0.1) -- only tokenizer and positional encoding
  differ, so the comparison is controlled.
- **max_seq_len is decoupled from the 16-word data filter.** The original
  prototype conflated "how long training sentences are" with "how many
  positions the model supports," which would silently break on longer
  inputs. Word/BPE variants use `max_seq_len=64`; char-level uses `128`
  since character sequences run ~4-6x longer per sentence. `decoding.py`
  clamps any requested generation length to the model's actual capacity.
- **BLEU is computed on detokenized plain text** for every variant (every
  tokenizer's `decode()` returns normal whitespace text), so scores are
  comparable across word/BPE/char rather than being computed on
  incomparable subword or character sequences.
- **Word-level vocab has no subword fallback**, so it hits real UNK rates on
  rare words/proper nouns/numbers -- expect this to visibly cost it BLEU
  relative to BPE (near-zero UNK by construction). This is a genuine,
  presentable tradeoff, not a bug.
- **Output projection is not weight-tied** to the target embedding table.
  Kept this way for architectural simplicity; weight tying would reduce
  parameter count meaningfully for the large word-level French vocab
  (~24k types) as possible future work.

## Results

Populated by `evaluate.py` into `results/manifest.json`; the frontend reads
this file directly. Fill in after running the full training + evaluation
commands above:

| Variant | Params | BLEU (greedy) | BLEU (beam) |
|---|---|---|---|
| word_sinusoidal | | | |
| word_rope | | | |
| bpe_sinusoidal | | | |
| bpe_rope | | | |
| char_sinusoidal | | | |
| char_rope | | | |
