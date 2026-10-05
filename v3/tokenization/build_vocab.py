"""Trains and saves a (src, tgt) tokenizer pair for one tokenizer type.

Vocab depends only on the tokenizer type, not on positional encoding or
attention variant, so it's built once per type under
checkpoints/vocab/<type>/{src,tgt}/ and shared by every pe/attn variant
that uses that tokenizer.

Usage: python tokenization/build_vocab.py --tokenizer wordpiece
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from tokenization import build_tokenizer

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
V3_ROOT = os.path.join(THIS_DIR, "..")
REPO_ROOT = os.path.join(V3_ROOT, "..")
TRAIN_TSV = os.path.join(REPO_ROOT, "data", "splits", "train.tsv")
VOCAB_ROOT = os.path.join(V3_ROOT, "checkpoints", "vocab")


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

    if tokenizer_name.endswith("_joint"):
        # One BPE model trained on the concatenated bilingual corpus, saved
        # identically into both src/ and tgt/ -- so every downstream call
        # site (train.py, evaluate.py) that loads "src" and "tgt"
        # separately keeps working unchanged, it just loads two copies of
        # the same vocabulary. Vocab size kept equal to the per-language
        # vocabs (8000) so this is a clean "shared vs. separate" comparison,
        # not confounded by also changing vocab capacity.
        joint_tok = build_tokenizer(tokenizer_name)
        joint_tok.train(eng_sentences + fra_sentences)
        joint_tok.save(src_dir)
        joint_tok.save(tgt_dir)
        print(f"[{tokenizer_name}] joint vocab_size={joint_tok.vocab_size} (shared src+tgt)")
        return

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
    parser.add_argument("--tokenizer", choices=["word", "bpe", "bpe_joint", "wordpiece", "char"], required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    build_and_save(args.tokenizer, force=args.force)
