"""Does wrapping the champion model with the NER+masking+glossary pipeline
change standard translation quality? Most sentences have no entities and
should be untouched; this checks that empirically rather than assuming it,
and separately reports how often the pipeline actually engages.
"""
import os
import random

import sacrebleu

import manifest
import torch
from decoding import greedy_decode
from entity_pipeline import translate_with_entity_pipeline
from model import load_model
from tokenization import TOKENIZER_REGISTRY
from tokenization.base import EOS_ID

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.join(THIS_DIR, "..")
N = 2000

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
e = manifest.load()["bpe_joint_rope_mha"]
model, _ = load_model(os.path.join(THIS_DIR, e["checkpoint_path"]), device=device)
tok_cls = TOKENIZER_REGISTRY[e["tokenizer_type"]]
vdir = os.path.join(THIS_DIR, e["vocab_dir"])
src_tok = tok_cls.load(os.path.join(vdir, "src"))
tgt_tok = tok_cls.load(os.path.join(vdir, "tgt"))


def translate_fn(sentence):
    ids = src_tok.encode(sentence) + [EOS_ID]
    out_ids, _ = greedy_decode(model, ids)
    return tgt_tok.decode(out_ids)


pairs = []
with open(os.path.join(REPO_ROOT, "data", "splits", "test.tsv"), encoding="utf-8") as f:
    for line in f:
        eng, fra = line.rstrip("\n").split("\t")
        pairs.append((eng, fra))

random.seed(7)
sample = random.sample(pairs, N)

hyps_plain, hyps_pipeline = [], []
n_entities_fired = 0
for eng, _ in sample:
    hyps_plain.append(translate_fn(eng))
    result = translate_with_entity_pipeline(eng, translate_fn)
    hyps_pipeline.append(result["final"])
    if result["entities"]:
        n_entities_fired += 1

refs = [f for _, f in sample]
bleu_plain = sacrebleu.corpus_bleu(hyps_plain, [refs]).score
bleu_pipeline = sacrebleu.corpus_bleu(hyps_pipeline, [refs]).score

print(f"N={N} sampled test sentences")
print(f"Pipeline fired (>=1 entity detected) on {n_entities_fired}/{N} sentences "
      f"({100 * n_entities_fired / N:.1f}%)")
print(f"BLEU without pipeline: {bleu_plain:.2f}")
print(f"BLEU with pipeline:    {bleu_pipeline:.2f}")
print(f"delta: {bleu_pipeline - bleu_plain:+.2f}")

diffs = [(sample[i][0], hyps_plain[i], hyps_pipeline[i]) for i in range(N) if hyps_plain[i] != hyps_pipeline[i]]
print(f"\n{len(diffs)} sentences where the pipeline changed the output. First 15:")
for src, plain, piped in diffs[:15]:
    print(f"  src:      {src}")
    print(f"  plain:    {plain}")
    print(f"  pipeline: {piped}")
