"""Corpus-level BLEU evaluation on the held-out test split.

Usage:
    python evaluate.py --variant bpe_rope --decode greedy
    python evaluate.py --variant bpe_rope --decode beam --beam_size 5
    python evaluate.py --variant word_sinusoidal --all-decodes
    python evaluate.py --all --decode greedy
    python evaluate.py --all --decode greedy --max_examples 2000   # faster, smaller sample

BLEU is computed on detokenized plain text (every tokenizer's decode()
returns normal whitespace text) so scores are comparable across word/BPE/
char variants -- computing BLEU directly on subword pieces or characters
would silently produce numbers that aren't comparable.
"""
import argparse
import os

import sacrebleu
import torch

import manifest
from decoding import decode as run_decode
from inference import load_variant
from tokenization.base import EOS_ID

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_TSV = os.path.join(THIS_DIR, "data", "splits", "test.tsv")

DECODE_KWARGS = {
    "greedy": {},
    "beam": {"beam_size": 5, "length_penalty": 0.6},
    "top_k": {"top_k": 10, "seed": 42},
    "top_p": {"top_p": 0.9, "seed": 42},
}


def read_test_pairs(max_examples=None):
    pairs = []
    with open(TEST_TSV, encoding="utf-8") as f:
        for line in f:
            eng, fra = line.rstrip("\n").split("\t")
            pairs.append((eng, fra))
    if max_examples:
        pairs = pairs[:max_examples]
    return pairs


def evaluate_variant(variant_id, decode_strategy, max_examples=None, device=None):
    model, src_tok, tgt_tok, entry = load_variant(variant_id, device=device)
    pairs = read_test_pairs(max_examples)

    hypotheses, references = [], []
    kwargs = DECODE_KWARGS.get(decode_strategy, {})
    for i, (eng, fra) in enumerate(pairs):
        src_ids = src_tok.encode(eng) + [EOS_ID]
        out_ids, _ = run_decode(model, src_ids, strategy=decode_strategy, return_attn=False, **kwargs)
        hypotheses.append(tgt_tok.decode(out_ids))
        references.append(fra)
        if (i + 1) % 500 == 0:
            print(f"[{variant_id}/{decode_strategy}] {i + 1}/{len(pairs)}")

    bleu = sacrebleu.corpus_bleu(hypotheses, [references]).score
    print(f"[{variant_id}/{decode_strategy}] BLEU={bleu:.2f} on {len(pairs)} examples")

    manifest.upsert(variant_id, {"bleu": {decode_strategy: bleu}})
    return bleu


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant")
    parser.add_argument("--all", action="store_true", help="evaluate every variant in the manifest")
    parser.add_argument("--decode", default="greedy", choices=list(DECODE_KWARGS))
    parser.add_argument("--all-decodes", action="store_true", help="run greedy+beam+top_k+top_p for --variant")
    parser.add_argument("--max_examples", type=int, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.all:
        variant_ids = list(manifest.load().keys())
    elif args.variant:
        variant_ids = [args.variant]
    else:
        parser.error("pass --variant NAME or --all")

    strategies = list(DECODE_KWARGS) if args.all_decodes else [args.decode]

    for vid in variant_ids:
        for strategy in strategies:
            evaluate_variant(vid, strategy, max_examples=args.max_examples, device=device)


if __name__ == "__main__":
    main()
