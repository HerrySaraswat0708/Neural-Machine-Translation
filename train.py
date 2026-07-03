"""Config-driven training entrypoint for one (tokenizer, positional encoding)
variant.

Usage:
    python train.py --tokenizer word --pe sinusoidal
    python train.py --tokenizer bpe   --pe rope --epochs 20
    python train.py --tokenizer char  --pe sinusoidal --subsample 2000 --epochs 1   # smoke test
"""
import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, TensorDataset

import manifest
from model import Transformer, PAD_token, SOS_token, EOS_token
from tokenization import TOKENIZER_REGISTRY
from tokenization.build_vocab import build_and_save, vocab_dir

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
SPLITS_DIR = os.path.join(THIS_DIR, "data", "splits")
CHECKPOINTS_DIR = os.path.join(THIS_DIR, "checkpoints")

with open(os.path.join(THIS_DIR, "configs", "defaults.json"), encoding="utf-8") as f:
    DEFAULTS = json.load(f)


def read_tsv(path):
    pairs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            eng, fra = line.rstrip("\n").split("\t")
            pairs.append((eng, fra))
    return pairs


def encode_pairs(pairs, src_tok, tgt_tok, max_seq_len):
    n = len(pairs)
    src_ids = np.full((n, max_seq_len), PAD_token, dtype=np.int64)
    tgt_ids = np.full((n, max_seq_len), PAD_token, dtype=np.int64)
    for i, (eng, fra) in enumerate(pairs):
        s = src_tok.encode(eng)[: max_seq_len - 1] + [EOS_token]
        t = [SOS_token] + tgt_tok.encode(fra)[: max_seq_len - 2] + [EOS_token]
        src_ids[i, : len(s)] = s
        tgt_ids[i, : len(t)] = t
    return torch.from_numpy(src_ids), torch.from_numpy(tgt_ids)


def noam_lr(step, emb_dim, warmup):
    step = max(step, 1)
    return (emb_dim ** -0.5) * min(step ** -0.5, step * warmup ** -1.5)


def variant_id(tokenizer, pe):
    return f"{tokenizer}_{pe}"


def run_eval(model, loader, criterion, device):
    model.eval()
    total_loss, total_correct, total_tokens = 0.0, 0, 0
    with torch.no_grad():
        for src, tgt in loader:
            src, tgt = src.to(device), tgt.to(device)
            tgt_input, tgt_target = tgt[:, :-1], tgt[:, 1:]
            logits = model(src, tgt_input)
            loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_target.reshape(-1))
            total_loss += loss.item()
            preds = logits.argmax(dim=-1)
            mask = tgt_target != PAD_token
            total_correct += ((preds == tgt_target) & mask).sum().item()
            total_tokens += mask.sum().item()
    avg_loss = total_loss / max(1, len(loader))
    acc = 100 * total_correct / total_tokens if total_tokens else 0.0
    return avg_loss, acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", choices=list(TOKENIZER_REGISTRY), required=True)
    parser.add_argument("--pe", choices=["sinusoidal", "rope"], required=True)
    parser.add_argument("--epochs", type=int, default=DEFAULTS["epochs"])
    parser.add_argument("--subsample", type=int, default=0, help="0 = full dataset, N>0 = quick smoke test")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--force_vocab", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(DEFAULTS["seed"])

    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto"
        else torch.device(args.device)
    )
    print("Using device:", device)

    build_and_save(args.tokenizer, force=args.force_vocab)
    tok_cls = TOKENIZER_REGISTRY[args.tokenizer]
    src_tok = tok_cls.load(os.path.join(vocab_dir(args.tokenizer), "src"))
    tgt_tok = tok_cls.load(os.path.join(vocab_dir(args.tokenizer), "tgt"))

    max_seq_len = DEFAULTS["max_seq_len_char"] if args.tokenizer == "char" else DEFAULTS["max_seq_len_default"]

    train_pairs = read_tsv(os.path.join(SPLITS_DIR, "train.tsv"))
    test_pairs = read_tsv(os.path.join(SPLITS_DIR, "test.tsv"))
    if args.subsample:
        train_pairs = train_pairs[: args.subsample]
        test_pairs = test_pairs[: max(1, args.subsample // 5)]

    train_src, train_tgt = encode_pairs(train_pairs, src_tok, tgt_tok, max_seq_len)
    test_src, test_tgt = encode_pairs(test_pairs, src_tok, tgt_tok, max_seq_len)

    train_loader = DataLoader(
        TensorDataset(train_src, train_tgt), batch_size=DEFAULTS["batch_size"], shuffle=True, drop_last=True
    )
    test_loader = DataLoader(
        TensorDataset(test_src, test_tgt), batch_size=DEFAULTS["batch_size"], shuffle=False, drop_last=True
    )

    model = Transformer(
        src_vocab_size=src_tok.vocab_size,
        tgt_vocab_size=tgt_tok.vocab_size,
        emb_dim=DEFAULTS["emb_dim"],
        num_heads=DEFAULTS["num_heads"],
        ff_dim=DEFAULTS["ff_dim"],
        num_layers=DEFAULTS["num_layers"],
        max_seq_len=max_seq_len,
        dropout=DEFAULTS["dropout"],
        pad_idx=PAD_token,
        pe_type=args.pe,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9)
    scheduler = LambdaLR(optimizer, lr_lambda=lambda step: noam_lr(step, DEFAULTS["emb_dim"], DEFAULTS["warmup_steps"]))
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_token, label_smoothing=DEFAULTS["label_smoothing"])

    vid = variant_id(args.tokenizer, args.pe)
    out_dir = os.path.join(CHECKPOINTS_DIR, vid)
    os.makedirs(out_dir, exist_ok=True)

    log = []
    start_time = time.time()
    for epoch in range(args.epochs):
        model.train()
        total_loss, total_correct, total_tokens = 0.0, 0, 0
        for src, tgt in train_loader:
            src, tgt = src.to(device), tgt.to(device)
            tgt_input, tgt_target = tgt[:, :-1], tgt[:, 1:]

            optimizer.zero_grad()
            logits = model(src, tgt_input)
            loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_target.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), DEFAULTS["grad_clip"])
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            preds = logits.argmax(dim=-1)
            mask = tgt_target != PAD_token
            total_correct += ((preds == tgt_target) & mask).sum().item()
            total_tokens += mask.sum().item()

        avg_loss = total_loss / len(train_loader)
        acc = 100 * total_correct / total_tokens if total_tokens else 0.0
        lr_now = scheduler.get_last_lr()[0]
        print(f"[{vid}] epoch {epoch + 1}/{args.epochs} loss={avg_loss:.4f} acc={acc:.2f}% lr={lr_now:.2e}")
        log.append({"epoch": epoch + 1, "loss": avg_loss, "accuracy": acc, "lr": lr_now})

    train_time_sec = time.time() - start_time
    eval_loss, eval_acc = run_eval(model, test_loader, criterion, device)
    print(f"[{vid}] eval loss={eval_loss:.4f} acc={eval_acc:.2f}%")

    config = {
        "src_vocab_size": src_tok.vocab_size,
        "tgt_vocab_size": tgt_tok.vocab_size,
        "emb_dim": DEFAULTS["emb_dim"],
        "num_heads": DEFAULTS["num_heads"],
        "ff_dim": DEFAULTS["ff_dim"],
        "num_layers": DEFAULTS["num_layers"],
        "max_seq_len": max_seq_len,
        "dropout": DEFAULTS["dropout"],
        "pad_idx": PAD_token,
        "pe_type": args.pe,
    }
    torch.save({"model_state_dict": model.state_dict(), "config": config}, os.path.join(out_dir, "model.pt"))
    with open(os.path.join(out_dir, "training_log.json"), "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2)

    num_params = sum(p.numel() for p in model.parameters())
    manifest.upsert(vid, {
        "tokenizer_type": args.tokenizer,
        "pe_type": args.pe,
        "src_vocab_size": src_tok.vocab_size,
        "tgt_vocab_size": tgt_tok.vocab_size,
        "max_seq_len": max_seq_len,
        "num_params": num_params,
        "epochs": args.epochs,
        "train_time_sec": train_time_sec,
        "final_train_loss": log[-1]["loss"] if log else None,
        "final_train_acc": log[-1]["accuracy"] if log else None,
        "final_eval_loss": eval_loss,
        "final_eval_acc": eval_acc,
        "checkpoint_path": os.path.join("checkpoints", vid, "model.pt"),
        "vocab_dir": os.path.join("checkpoints", "vocab", args.tokenizer),
        "device": str(device),
    })
    print(f"[{vid}] saved checkpoint to {out_dir}, manifest updated.")


if __name__ == "__main__":
    main()
