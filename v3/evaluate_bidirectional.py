"""Scores a bidirectional (tag-prefixed) model separately in each
direction -- evaluate.py assumes one fixed src-language/tgt-language pair,
which doesn't fit a single model that handles both en->fr and fr->en via
a prepended tag.
"""
import argparse
import os

import sacrebleu
import torch

import manifest
from decoding import greedy_decode
from model import load_model
from tokenization import TOKENIZER_REGISTRY
from tokenization.base import EOS_ID

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.join(THIS_DIR, "..")


def read_bidi_test():
    to_fr, to_en = [], []
    with open(os.path.join(REPO_ROOT, "data", "splits", "test_bidi.tsv"), encoding="utf-8") as f:
        for line in f:
            src, tgt = line.rstrip("\n").split("\t")
            if src.startswith("tofr "):
                to_fr.append((src, tgt))
            else:
                to_en.append((src, tgt))
    return to_fr, to_en


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", required=True)
    parser.add_argument("--max_examples", type=int, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    entry = manifest.load()[args.variant]
    model, _ = load_model(os.path.join(THIS_DIR, entry["checkpoint_path"]), device=device)
    tok_cls = TOKENIZER_REGISTRY[entry["tokenizer_type"]]
    vdir = os.path.join(THIS_DIR, entry["vocab_dir"])
    tok = tok_cls.load(os.path.join(vdir, "src"))  # same object serves both directions (joint vocab)

    to_fr, to_en = read_bidi_test()
    if args.max_examples:
        to_fr, to_en = to_fr[: args.max_examples], to_en[: args.max_examples]

    for direction, pairs in [("en->fr", to_fr), ("fr->en", to_en)]:
        hyps, refs = [], []
        for i, (src, ref) in enumerate(pairs):
            src_ids = tok.encode(src) + [EOS_ID]
            out_ids, _ = greedy_decode(model, src_ids)
            hyps.append(tok.decode(out_ids))
            refs.append(ref)
            if (i + 1) % 2000 == 0:
                print(f"[{args.variant}/{direction}] {i + 1}/{len(pairs)}")
        bleu = sacrebleu.corpus_bleu(hyps, [refs]).score
        chrf = sacrebleu.corpus_chrf(hyps, [refs], word_order=2).score
        ter = sacrebleu.corpus_ter(hyps, [refs]).score
        print(f"[{args.variant}/{direction}] BLEU={bleu:.2f} chrF++={chrf:.2f} TER={ter:.2f} on {len(pairs)} examples")
        manifest.upsert(args.variant, {
            f"bleu_{direction.replace('->', '2')}": bleu,
            f"chrf_{direction.replace('->', '2')}": chrf,
            f"ter_{direction.replace('->', '2')}": ter,
        })


if __name__ == "__main__":
    main()
