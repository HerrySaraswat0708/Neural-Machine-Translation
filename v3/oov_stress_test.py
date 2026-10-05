"""Does the joint-BPE-vocab model actually preserve unseen proper nouns
better than the separate-vocab winner? Direct before/after comparison on
names that are genuinely absent from the training data (verified in
PROGRESS.md), rather than trusting aggregate BLEU to reflect a narrow
behavior it mostly won't move the needle on.
"""
import os
import torch

import manifest
from model import load_model
from tokenization import TOKENIZER_REGISTRY
from tokenization.base import EOS_ID
from decoding import greedy_decode

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
BEFORE = "bpe_rope_mha_widernarrowffnlong"  # separate src/tgt vocabs
AFTER = "bpe_joint_rope_mha"                # shared vocab

NAMES = ["herry", "zhorvaq", "blipnak", "quexinor", "tom", "mary"]
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


def translate(model, src_tok, tgt_tok, sentence):
    src_ids = src_tok.encode(sentence) + [EOS_ID]
    out_ids, _ = greedy_decode(model, src_ids)
    return tgt_tok.decode(out_ids)


model_b, src_b, tgt_b = load(BEFORE)
model_a, src_a, tgt_a = load(AFTER)

print(f"{'name':10} | {'BEFORE (separate vocab)':45} | preserved? | {'AFTER (joint vocab)':45} | preserved?")
print("-" * 130)
n_before_ok, n_after_ok = 0, 0
for name in NAMES:
    sentence = f"my name is {name} ."
    out_b = translate(model_b, src_b, tgt_b, sentence)
    out_a = translate(model_a, src_a, tgt_a, sentence)
    ok_b = name in out_b.lower()
    ok_a = name in out_a.lower()
    n_before_ok += ok_b
    n_after_ok += ok_a
    print(f"{name:10} | {out_b:45} | {'yes' if ok_b else 'NO':10} | {out_a:45} | {'yes' if ok_a else 'NO'}")

print()
print(f"Preserved verbatim: BEFORE {n_before_ok}/{len(NAMES)}   AFTER {n_after_ok}/{len(NAMES)}")
