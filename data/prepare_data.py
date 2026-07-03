"""Builds a fixed, tokenizer-agnostic train/test split from eng-fra.txt.

Run once: `python data/prepare_data.py`. All tokenizer/PE variants train and
evaluate against the same split so BLEU scores are comparable across them.
"""
import os
import random

from normalize import normalize_english, normalize_french

MAX_LENGTH_WORDS = 16
SEED = 42
TRAIN_FRACTION = 0.8

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_PATH = os.path.join(THIS_DIR, "..", "eng-fra.txt")
SPLITS_DIR = os.path.join(THIS_DIR, "splits")


def load_pairs(path):
    lines = open(path, encoding="utf-8").read().strip().split("\n")
    pairs = []
    for line in lines:
        eng, fra = line.split("\t")[:2]
        pairs.append((normalize_english(eng), normalize_french(fra)))
    return pairs


def filter_pairs(pairs):
    return [
        (eng, fra)
        for eng, fra in pairs
        if len(eng.split()) < MAX_LENGTH_WORDS and len(fra.split()) < MAX_LENGTH_WORDS
    ]


def write_tsv(path, pairs):
    with open(path, "w", encoding="utf-8") as f:
        for eng, fra in pairs:
            f.write(f"{eng}\t{fra}\n")


def main():
    pairs = filter_pairs(load_pairs(RAW_PATH))
    random.Random(SEED).shuffle(pairs)

    split_idx = round(len(pairs) * TRAIN_FRACTION)
    train_pairs, test_pairs = pairs[:split_idx], pairs[split_idx:]

    os.makedirs(SPLITS_DIR, exist_ok=True)
    write_tsv(os.path.join(SPLITS_DIR, "train.tsv"), train_pairs)
    write_tsv(os.path.join(SPLITS_DIR, "test.tsv"), test_pairs)

    print(f"Total pairs after filtering: {len(pairs)}")
    print(f"Train: {len(train_pairs)}  Test: {len(test_pairs)}")


if __name__ == "__main__":
    main()
