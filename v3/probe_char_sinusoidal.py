import os, sys, torch
THIS_DIR = "/home/herrys/projects/Neural-Machine-Translation/v3"
sys.path.insert(0, THIS_DIR)
os.chdir(THIS_DIR)

from decoding import greedy_decode
from model import load_model
from tokenization import TOKENIZER_REGISTRY
from tokenization.base import EOS_ID

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

pairs = []
with open("../data/splits/test.tsv", encoding="utf-8") as f:
    for line in f:
        eng, fra = line.rstrip("\n").split("\t")
        pairs.append((eng, fra))

tok_cls = TOKENIZER_REGISTRY["char"]
src_tok = tok_cls.load("checkpoints/vocab/char/src")
tgt_tok = tok_cls.load("checkpoints/vocab/char/tgt")


def run(variant, n=10):
    print("#" * 70)
    print("VARIANT:", variant)
    model, cfg = load_model(f"checkpoints/{variant}/model.pt", device=device)
    print("config:", cfg)
    hit_max_no_eos = 0
    for eng, fra in pairs[:n]:
        src_ids = src_tok.encode(eng) + [EOS_ID]
        out_ids, _ = greedy_decode(model, src_ids, return_attn=False)
        hyp = tgt_tok.decode(out_ids)
        ended_with_eos = (len(out_ids) > 0 and out_ids[-1] == EOS_ID)
        if not ended_with_eos:
            hit_max_no_eos += 1
        print("-" * 60)
        print("EN :", eng)
        print("REF:", fra)
        print("HYP:", repr(hyp)[:200])
        print(f"len(out_ids)={len(out_ids)} max_seq_len={model.max_seq_len} eos_hit={ended_with_eos}")
        print("first 40 ids:", out_ids[:40])
    print(f"\n[{variant}] {hit_max_no_eos}/{n} examples ran to max_seq_len without emitting EOS")
    del model
    torch.cuda.empty_cache()


run("char_sinusoidal_mha", n=10)
run("char_rope_mha", n=10)
