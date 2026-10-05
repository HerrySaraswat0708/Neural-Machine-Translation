"""Corpus-level BLEU/chrF++/TER evaluation on the held-out test split.

Usage:
    python evaluate.py --variant bpe_rope_mha --decode greedy
    python evaluate.py --all --decode greedy
    python evaluate.py --all --decode greedy --max_examples 2000   # faster, smaller sample
"""
import argparse
import os

import torch

import manifest
from decoding import decode as run_decode
from eval.metrics import all_metrics
from model import load_model
from tokenization import TOKENIZER_REGISTRY
from tokenization.base import EOS_ID

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.join(THIS_DIR, "..")
TEST_TSV = os.path.join(REPO_ROOT, "data", "splits", "test.tsv")

DECODE_KWARGS = {
    "greedy": {},
    "beam": {"beam_size": 5, "length_penalty": 0.6},
    "top_k": {"top_k": 10, "seed": 42},
    "top_p": {"top_p": 0.9, "seed": 42},
}


def load_variant(variant_id: str, device=None):
    entries = manifest.load()
    if variant_id not in entries:
        raise KeyError(f"Variant '{variant_id}' not found in manifest. Has it been trained?")
    entry = entries[variant_id]

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = os.path.join(THIS_DIR, entry["checkpoint_path"])
    model, _ = load_model(checkpoint_path, device=device)

    tok_cls = TOKENIZER_REGISTRY[entry["tokenizer_type"]]
    vdir = os.path.join(THIS_DIR, entry["vocab_dir"])
    src_tok = tok_cls.load(os.path.join(vdir, "src"))
    tgt_tok = tok_cls.load(os.path.join(vdir, "tgt"))

    return model, src_tok, tgt_tok, entry


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

    scores = all_metrics(hypotheses, references)
    print(f"[{variant_id}/{decode_strategy}] BLEU={scores['bleu']:.2f} chrF++={scores['chrf']:.2f} "
          f"TER={scores['ter']:.2f} on {len(pairs)} examples")

    manifest.upsert(variant_id, {
        "bleu": {decode_strategy: scores["bleu"]},
        "chrf": {decode_strategy: scores["chrf"]},
        "ter": {decode_strategy: scores["ter"]},
    })
    return scores


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
