import os, sys, torch
THIS_DIR = "/home/herrys/projects/Neural-Machine-Translation/v3"
sys.path.insert(0, THIS_DIR)
os.chdir(THIS_DIR)

from decoding import greedy_decode
from model import load_model
from tokenization import TOKENIZER_REGISTRY
from tokenization.base import EOS_ID

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tok_cls = TOKENIZER_REGISTRY["char"]
src_tok = tok_cls.load("checkpoints/vocab/char/src")
tgt_tok = tok_cls.load("checkpoints/vocab/char/tgt")

pairs = []
with open("../data/splits/test.tsv", encoding="utf-8") as f:
    for line in f:
        eng, fra = line.rstrip("\n").split("\t")
        pairs.append((eng, fra))


def run(variant, n=3):
    print("#" * 70, variant)
    model, cfg = load_model(f"checkpoints/{variant}/model.pt", device=device)
    for eng, fra in pairs[:n]:
        src_ids = src_tok.encode(eng) + [EOS_ID]
        out_ids, attn = greedy_decode(model, src_ids, return_attn=True)
        hyp = tgt_tok.decode(out_ids)
        print("-" * 60)
        print("EN:", eng, " HYP:", hyp)
        if attn is not None:
            # attn: [tgt_len-1, src_len] -- print max weight position + entropy per row
            import math
            for row in attn:
                row = row.tolist()
                mx = max(row)
                mx_idx = row.index(mx)
                ent = -sum(p * math.log(p + 1e-12) for p in row)
                print(f"  argmax_src_pos={mx_idx:3d} max_w={mx:.3f} entropy={ent:.3f} (uniform_ent={math.log(len(row)):.3f})")
    del model
    torch.cuda.empty_cache()


run("char_sinusoidal_mha")
run("char_rope_mha")
