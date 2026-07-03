"""Trains and saves a (src, tgt) tokenizer pair for one tokenizer type.

Vocab depends only on the tokenizer type, not on positional encoding, so it's
built once per type under checkpoints/vocab/<type>/{src,tgt}/ and shared by
both the sinusoidal and RoPE model variants that use that tokenizer.

Usage: python tokenization/build_vocab.py --tokenizer word
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from tokenization import build_tokenizer

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.join(THIS_DIR, "..")
TRAIN_TSV = os.path.join(REPO_ROOT, "data", "splits", "train.tsv")
VOCAB_ROOT = os.path.join(REPO_ROOT, "checkpoints", "vocab")


def read_columns(tsv_path):
    eng_sentences, fra_sentences = [], []
    with open(tsv_path, encoding="utf-8") as f:
        for line in f:
            eng, fra = line.rstrip("\n").split("\t")
            eng_sentences.append(eng)
            fra_sentences.append(fra)
    return eng_sentences, fra_sentences


def vocab_dir(tokenizer_name: str):
    return os.path.join(VOCAB_ROOT, tokenizer_name)


def build_and_save(tokenizer_name: str, force: bool = False):
    out_dir = vocab_dir(tokenizer_name)
    src_dir, tgt_dir = os.path.join(out_dir, "src"), os.path.join(out_dir, "tgt")
    if not force and os.path.isdir(src_dir) and os.path.isdir(tgt_dir):
        print(f"[{tokenizer_name}] vocab already exists at {out_dir}, skipping (use --force to rebuild)")
        return

    eng_sentences, fra_sentences = read_columns(TRAIN_TSV)

    src_tok = build_tokenizer(tokenizer_name)
    src_tok.train(eng_sentences)
    src_tok.save(src_dir)
    print(f"[{tokenizer_name}] src (eng) vocab_size={src_tok.vocab_size}")

    tgt_tok = build_tokenizer(tokenizer_name)
    tgt_tok.train(fra_sentences)
    tgt_tok.save(tgt_dir)
    print(f"[{tokenizer_name}] tgt (fra) vocab_size={tgt_tok.vocab_size}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", choices=["word", "bpe", "char"], required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    build_and_save(args.tokenizer, force=args.force)
