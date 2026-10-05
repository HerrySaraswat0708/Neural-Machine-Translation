import torch
import torch.nn as nn

from model.attention import Attention, num_kv_heads_for
from model.positional import SinusoidalPositionalEncoding

PAD_token = 0
SOS_token = 1
EOS_token = 2


class TransformerEmbedding(nn.Module):
    def __init__(self, vocab_size, emb_dim, max_seq_len, pe_type="sinusoidal"):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, emb_dim)
        nn.init.normal_(self.token_emb.weight, mean=0, std=0.1)
        # RoPE and ALiBi inject position at the attention-score level
        # (attention.py) instead of an added embedding table.
        self.pos_emb = SinusoidalPositionalEncoding(max_seq_len, emb_dim) if pe_type == "sinusoidal" else None

    def forward(self, x):
        x = self.token_emb(x)
        if self.pos_emb is not None:
            x = self.pos_emb(x)
        return x


class PositionwiseFeedForward(nn.Module):
    def __init__(self, emb_dim, ff_dim, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(emb_dim, ff_dim)
        self.linear2 = nn.Linear(ff_dim, emb_dim)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.linear2(self.dropout(self.relu(self.linear1(x))))


class AddNorm(nn.Module):
    def __init__(self, emb_dim, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(emb_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return self.norm(x + self.dropout(sublayer))


class EncoderBlock(nn.Module):
    def __init__(self, emb_dim, num_heads, num_kv_heads, ff_dim, dropout=0.1, pe_type="sinusoidal", max_seq_len=64):
        super().__init__()
        attn_pe = pe_type if pe_type in ("rope", "alibi") else "none"
        self.attn = Attention(emb_dim, num_heads, num_kv_heads, pe_type=attn_pe, max_seq_len=max_seq_len, dropout=dropout)
        self.addnorm1 = AddNorm(emb_dim, dropout)
        self.ffn = PositionwiseFeedForward(emb_dim, ff_dim, dropout)
        self.addnorm2 = AddNorm(emb_dim, dropout)

    def forward(self, x, enc_mask=None):
        x = self.addnorm1(x, self.attn(x, x, x, enc_mask))
        x = self.addnorm2(x, self.ffn(x))
        return x


class DecoderBlock(nn.Module):
    def __init__(self, emb_dim, num_heads, num_kv_heads, ff_dim, dropout=0.1, pe_type="sinusoidal", max_seq_len=64):
        super().__init__()
        attn_pe = pe_type if pe_type in ("rope", "alibi") else "none"
        self.masked_attn = Attention(emb_dim, num_heads, num_kv_heads, pe_type=attn_pe, max_seq_len=max_seq_len, dropout=dropout)
        self.addnorm1 = AddNorm(emb_dim, dropout)

        # Cross-attention deliberately never gets RoPE or ALiBi: source and
        # target are different sequences with unrelated lengths/order, so
        # there's no meaningful "relative offset" between a target position
        # and a source position for either to encode. Positional
        # information for cross-attention already comes from the
        # (position-aware, for rope/alibi models) encoder output itself.
        self.cross_attn = Attention(emb_dim, num_heads, num_kv_heads, pe_type="none", dropout=dropout)
        self.addnorm2 = AddNorm(emb_dim, dropout)

        self.ffn = PositionwiseFeedForward(emb_dim, ff_dim, dropout)
        self.addnorm3 = AddNorm(emb_dim, dropout)

    def forward(self, x, enc_out, tgt_mask=None, enc_mask=None, return_attn=False,
                self_cache=None, cross_cache=None, start_pos=0, use_cache=False):
        if use_cache:
            self_out, new_self_cache = self.masked_attn(
                x, x, x, tgt_mask, kv_cache=self_cache, start_pos=start_pos,
                use_cache=True, cache_mode="growing")
            x = self.addnorm1(x, self_out)

            if return_attn:
                cross_out, cross_attn, new_cross_cache = self.cross_attn(
                    x, enc_out, enc_out, enc_mask, return_attn=True,
                    kv_cache=cross_cache, use_cache=True, cache_mode="static")
            else:
                cross_out, new_cross_cache = self.cross_attn(
                    x, enc_out, enc_out, enc_mask,
                    kv_cache=cross_cache, use_cache=True, cache_mode="static")
            x = self.addnorm2(x, cross_out)
            x = self.addnorm3(x, self.ffn(x))

            if return_attn:
                return x, cross_attn, new_self_cache, new_cross_cache
            return x, new_self_cache, new_cross_cache

        x = self.addnorm1(x, self.masked_attn(x, x, x, tgt_mask))

        if return_attn:
            cross_out, cross_attn = self.cross_attn(x, enc_out, enc_out, enc_mask, return_attn=True)
        else:
            cross_out = self.cross_attn(x, enc_out, enc_out, enc_mask)
        x = self.addnorm2(x, cross_out)

        x = self.addnorm3(x, self.ffn(x))

        if return_attn:
            return x, cross_attn
        return x


def generate_padding_mask(src, pad_idx=0):
    return (src == pad_idx).type(torch.int16).unsqueeze(-2).unsqueeze(-2)


def generate_subsequent_mask(seq_len, device=None):
    mask = torch.triu(torch.ones((1, seq_len, seq_len), device=device), diagonal=1).type(torch.int16)
    return mask


class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, emb_dim=512, num_heads=8,
                 ff_dim=2048, num_layers=6, max_seq_len=256, dropout=0.1, pad_idx=0,
                 pe_type="sinusoidal", attn_type="mha", num_kv_heads=None):
        super().__init__()
        self.pad_idx = pad_idx
        self.pe_type = pe_type
        self.attn_type = attn_type
        self.max_seq_len = max_seq_len
        self.num_kv_heads = num_kv_heads or num_kv_heads_for(attn_type, num_heads)

        self.src_embed = TransformerEmbedding(src_vocab_size, emb_dim, max_seq_len, pe_type)
        self.tgt_embed = TransformerEmbedding(tgt_vocab_size, emb_dim, max_seq_len, pe_type)

        self.encoder = nn.ModuleList([
            EncoderBlock(emb_dim, num_heads, self.num_kv_heads, ff_dim, dropout, pe_type, max_seq_len)
            for _ in range(num_layers)
        ])
        self.decoder = nn.ModuleList([
            DecoderBlock(emb_dim, num_heads, self.num_kv_heads, ff_dim, dropout, pe_type, max_seq_len)
            for _ in range(num_layers)
        ])
        self.output_proj = nn.Linear(emb_dim, tgt_vocab_size)

    def encode(self, src):
        src_mask = (src == self.pad_idx).unsqueeze(-2).unsqueeze(-2).to(src.device)
        enc_out = self.src_embed(src)
        for layer in self.encoder:
            enc_out = layer(enc_out, src_mask)
        return enc_out, src_mask

    def decode(self, tgt, enc_out, src_mask, return_attn=False,
               kv_cache=None, start_pos=0, use_cache=False):
        if use_cache:
            # tgt is just the new token(s) for this step, not the whole
            # sequence -- every cached position is already causally "in
            # the past" by construction (nothing is ever cached out of
            # order), so no subsequent-position mask is needed here.
            if self.pe_type == "sinusoidal":
                raise NotImplementedError(
                    "KV-cache decoding needs each new token's true absolute "
                    "position; sinusoidal PE's additive embedding-level "
                    "injection is only ever computed for position 0 when "
                    "called one token at a time, unlike RoPE/ALiBi's "
                    "per-layer, position-aware injection. Not implemented "
                    "for sinusoidal PE.")
            # chunk_len > 1 happens under speculative decoding (Phase 7e):
            # the target model verifies several draft-proposed tokens in
            # one parallel call rather than one at a time, which needs a
            # causal mask WITHIN the new chunk (query i may see cached
            # history plus new-chunk positions up to itself, not beyond) --
            # chunk_len == 1 (every other cached-decode caller, 7b/7c/7d)
            # degenerates to exactly the original single-token mask.
            chunk_len = tgt.shape[1]
            cache_len = start_pos
            tgt_pad_mask = (tgt == self.pad_idx).unsqueeze(-2).to(tgt.device)  # [b, 1, chunk_len]
            if chunk_len > 1:
                causal_suffix = torch.triu(
                    torch.ones((chunk_len, chunk_len), device=tgt.device), diagonal=1
                ).type(torch.int16)  # [chunk_len, chunk_len]
                prefix_visible = torch.zeros((chunk_len, cache_len), device=tgt.device, dtype=torch.int16)
                intra_chunk_mask = torch.cat([prefix_visible, causal_suffix], dim=1)  # [chunk_len, cache_len+chunk_len]
                pad_key_mask = torch.cat([
                    torch.zeros(tgt.size(0), cache_len, device=tgt.device, dtype=torch.int16),
                    tgt_pad_mask.squeeze(1),
                ], dim=1)  # [b, cache_len+chunk_len]
                combined_tgt_mask = (pad_key_mask.unsqueeze(1) | intra_chunk_mask.unsqueeze(0)).unsqueeze(1)
            else:
                combined_tgt_mask = tgt_pad_mask.unsqueeze(1)

            dec_out = self.tgt_embed(tgt)
            if kv_cache is None:
                kv_cache = [None] * len(self.decoder)
            new_kv_cache = []
            last_attn = None
            for layer, layer_cache in zip(self.decoder, kv_cache):
                self_cache, cross_cache = layer_cache if layer_cache is not None else (None, None)
                if return_attn:
                    dec_out, last_attn, new_self_cache, new_cross_cache = layer(
                        dec_out, enc_out, combined_tgt_mask, src_mask, return_attn=True,
                        self_cache=self_cache, cross_cache=cross_cache,
                        start_pos=start_pos, use_cache=True)
                else:
                    dec_out, new_self_cache, new_cross_cache = layer(
                        dec_out, enc_out, combined_tgt_mask, src_mask,
                        self_cache=self_cache, cross_cache=cross_cache,
                        start_pos=start_pos, use_cache=True)
                new_kv_cache.append((new_self_cache, new_cross_cache))

            logits = self.output_proj(dec_out)
            if return_attn:
                return logits, last_attn, new_kv_cache
            return logits, new_kv_cache

        tgt_mask = generate_subsequent_mask(tgt.shape[1], device=tgt.device)
        tgt_pad_mask = (tgt == self.pad_idx).unsqueeze(-2).to(tgt.device)
        combined_tgt_mask = (tgt_pad_mask | tgt_mask).unsqueeze(1)

        dec_out = self.tgt_embed(tgt)
        last_attn = None
        for layer in self.decoder:
            if return_attn:
                dec_out, last_attn = layer(dec_out, enc_out, combined_tgt_mask, src_mask, return_attn=True)
            else:
                dec_out = layer(dec_out, enc_out, combined_tgt_mask, src_mask)

        logits = self.output_proj(dec_out)
        if return_attn:
            return logits, last_attn
        return logits

    def forward(self, src, tgt):
        enc_out, src_mask = self.encode(src)
        return self.decode(tgt, enc_out, src_mask)


def load_model(path, device=None):
    """Loads a Transformer from a checkpoint saved by train.py, which stores
    {"model_state_dict": ..., "config": {...kwargs for Transformer...}}.
    The config travels with the checkpoint so callers never need to
    separately track architecture hyperparameters out-of-band."""
    device = device or torch.device("cpu")
    checkpoint = torch.load(path, map_location=device)
    model = Transformer(**checkpoint["config"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    return model, checkpoint["config"]
