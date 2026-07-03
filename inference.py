import os

import torch

import manifest
from data.normalize import normalize_english
from model import load_model
from tokenization import TOKENIZER_REGISTRY

THIS_DIR = os.path.dirname(os.path.abspath(__file__))


def load_variant(variant_id: str, device=None):
    """Loads a trained model + its src/tgt tokenizers for a variant_id like
    'bpe_rope'. Returns (model, src_tokenizer, tgt_tokenizer, manifest_entry)."""
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


def translate(variant_id, sentence, strategy="greedy", return_attn=False, device=None, **decode_kwargs):
    from decoding import decode as run_decode
    from tokenization.base import EOS_ID

    model, src_tok, tgt_tok, entry = load_variant(variant_id, device=device)
    src_ids = src_tok.encode(normalize_english(sentence)) + [EOS_ID]
    out_ids, attn = run_decode(model, src_ids, strategy=strategy, return_attn=return_attn, **decode_kwargs)
    translation = tgt_tok.decode(out_ids)

    src_tokens, tgt_tokens = None, None
    if return_attn and attn is not None:
        src_tokens = [_id_to_piece(src_tok, i) for i in src_ids]
        tgt_tokens = [_id_to_piece(tgt_tok, i) for i in out_ids]

    return translation, attn, src_tokens, tgt_tokens


def _id_to_piece(tokenizer, token_id):
    """Best-effort single-token label for heatmap axes (works across
    word/BPE/char tokenizers by decoding a singleton id list)."""
    piece = tokenizer.decode([token_id])
    return piece if piece else "·"
