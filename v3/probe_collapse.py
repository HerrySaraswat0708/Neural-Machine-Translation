import os, sys, torch
THIS_DIR = "/home/herrys/projects/Neural-Machine-Translation/v3"
sys.path.insert(0, THIS_DIR)
os.chdir(THIS_DIR)

from model import load_model
from tokenization import TOKENIZER_REGISTRY
from tokenization.base import EOS_ID

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tok_cls = TOKENIZER_REGISTRY["char"]
src_tok = tok_cls.load("checkpoints/vocab/char/src")

eng = "walking is a healthy form of exercise ."
src_ids = src_tok.encode(eng) + [EOS_ID]
src_tensor = torch.tensor([src_ids], dtype=torch.long, device=device)


def probe(variant):
    print("#" * 70, variant)
    model, cfg = load_model(f"checkpoints/{variant}/model.pt", device=device)
    model.eval()

    # 1. token_emb vs pos_emb magnitude at the input layer
    emb_mod = model.src_embed
    tok_w = emb_mod.token_emb.weight.detach()
    tok_norm = tok_w.norm(dim=-1).mean().item()
    print(f"  mean token_emb row norm: {tok_norm:.4f}")
    if emb_mod.pos_emb is not None:
        pe = emb_mod.pos_emb.pe[0].detach()  # [max_seq_len, emb_dim]
        pe_norm_at_len = pe[: len(src_ids)].norm(dim=-1).mean().item()
        print(f"  mean sinusoidal PE row norm (over seq positions used): {pe_norm_at_len:.4f}")
        print(f"  PE/token_emb norm ratio: {pe_norm_at_len / tok_norm:.2f}x")

    # 2. encoder output collapse: pairwise cosine similarity across positions
    with torch.no_grad():
        enc_out, src_mask = model.encode(src_tensor)
    h = enc_out[0]  # [src_len, emb_dim]
    h_norm = torch.nn.functional.normalize(h, dim=-1)
    sim = h_norm @ h_norm.T  # [src_len, src_len]
    n = sim.size(0)
    off_diag = sim[~torch.eye(n, dtype=torch.bool, device=sim.device)]
    print(f"  encoder output pairwise cosine sim: mean={off_diag.mean().item():.4f} "
          f"std={off_diag.std().item():.4f} min={off_diag.min().item():.4f} max={off_diag.max().item():.4f}")
    print(f"  encoder output per-position norm: mean={h.norm(dim=-1).mean().item():.4f} "
          f"std={h.norm(dim=-1).std().item():.4f}")

    del model
    torch.cuda.empty_cache()


probe("char_sinusoidal_mha")
probe("char_rope_mha")
