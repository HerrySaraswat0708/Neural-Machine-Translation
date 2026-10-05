"""Builds a bidirectional training/test set for one model handling both
en->fr and fr->en (Johnson et al. 2017's target-tag trick: one shared
vocabulary, one model, a prepended tag telling the decoder which language
to produce) -- instead of training two separate directional models.

For every (eng, fra) pair, emits two rows:
    "tofr " + eng  \t  fra
    "toen " + fra  \t  eng

Requires the bpe_joint vocabulary (same vocab for source and target is
what makes a single shared model meaningful across both directions).
"""
import os

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.join(THIS_DIR, "..")
SPLITS_DIR = os.path.join(REPO_ROOT, "data", "splits")


def build(split_name):
    src_path = os.path.join(SPLITS_DIR, f"{split_name}.tsv")
    dst_path = os.path.join(SPLITS_DIR, f"{split_name}_bidi.tsv")
    n = 0
    with open(src_path, encoding="utf-8") as fin, open(dst_path, "w", encoding="utf-8") as fout:
        for line in fin:
            eng, fra = line.rstrip("\n").split("\t")
            fout.write(f"tofr {eng}\t{fra}\n")
            fout.write(f"toen {fra}\t{eng}\n")
            n += 2
    print(f"[{split_name}] wrote {n} bidirectional pairs to {dst_path}")


if __name__ == "__main__":
    build("train")
    build("test")
