"""Does noise-augmented training actually improve robustness to
ungrammatical/typo'd input? Builds a synthetic noisy test set (same
mechanism as training augmentation, but a DIFFERENT seed so this measures
generalization to noise, not memorization of the exact training noise
pattern), then compares the pre-augmentation winner against the
noise-augmented model on both clean and noisy versions of the same
sentences.
"""
import os
import random

import torch
import sacrebleu

import manifest
from augment import noise_sentence
from model import load_model
from tokenization import TOKENIZER_REGISTRY
from tokenization.base import EOS_ID
from decoding import greedy_decode

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.join(THIS_DIR, "..")
BEFORE = "bpe_rope_mha_widernarrowffnlong"
AFTER = "bpe_rope_mha_noiseaug"
N = 3000
EVAL_NOISE_SEED = 99  # different from the training augmentation seed

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load(vid):
    entries = manifest.load()
    e = entries[vid]
    model, _ = load_model(os.path.join(THIS_DIR, e["checkpoint_path"]), device=device)
    tok_cls = TOKENIZER_REGISTRY[e["tokenizer_type"]]
    vdir = os.path.join(THIS_DIR, e["vocab_dir"])
    src_tok = tok_cls.load(os.path.join(vdir, "src"))
    tgt_tok = tok_cls.load(os.path.join(vdir, "tgt"))
    return model, src_tok, tgt_tok


def translate_all(model, src_tok, tgt_tok, sentences):
    hyps = []
    for s in sentences:
        src_ids = src_tok.encode(s) + [EOS_ID]
        out_ids, _ = greedy_decode(model, src_ids)
        hyps.append(tgt_tok.decode(out_ids))
    return hyps


pairs = []
with open(os.path.join(REPO_ROOT, "data", "splits", "test.tsv"), encoding="utf-8") as f:
    for line in f:
        eng, fra = line.rstrip("\n").split("\t")
        pairs.append((eng, fra))

random.seed(7)  # same sample used in the earlier A/B analysis, for continuity
sample = random.sample(pairs, N)
eng_clean = [e for e, _ in sample]
refs = [f for _, f in sample]

rng = random.Random(EVAL_NOISE_SEED)
eng_noisy = [noise_sentence(e, rng) for e in eng_clean]

model_b, src_b, tgt_b = load(BEFORE)
model_a, src_a, tgt_a = load(AFTER)

results = {}
for tag, model, src_tok, tgt_tok in [("BEFORE", model_b, src_b, tgt_b), ("AFTER", model_a, src_a, tgt_a)]:
    hyps_clean = translate_all(model, src_tok, tgt_tok, eng_clean)
    hyps_noisy = translate_all(model, src_tok, tgt_tok, eng_noisy)
    bleu_clean = sacrebleu.corpus_bleu(hyps_clean, [refs]).score
    bleu_noisy = sacrebleu.corpus_bleu(hyps_noisy, [refs]).score
    results[tag] = (bleu_clean, bleu_noisy)
    print(f"[{tag}] clean-input BLEU={bleu_clean:.2f}   noisy-input BLEU={bleu_noisy:.2f}   "
          f"degradation={bleu_clean - bleu_noisy:+.2f}")

print()
print("Sample noisy inputs (same 3 for both models):")
for i in range(3):
    print(f"  clean: {eng_clean[i]!r}")
    print(f"  noisy: {eng_noisy[i]!r}")

b_clean, b_noisy = results["BEFORE"]
a_clean, a_noisy = results["AFTER"]
print()
print(f"Noise-robustness gain from augmentation: BEFORE lost {b_clean - b_noisy:.2f} BLEU to noise, "
      f"AFTER lost {a_clean - a_noisy:.2f} BLEU to noise.")
print(f"Clean-set BLEU change from augmentation: {a_clean - b_clean:+.2f} (should be small/near-zero -- "
      f"augmentation should not regress normal translation quality)")
