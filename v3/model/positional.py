import math

import numpy as np
import torch
import torch.nn as nn


class SinusoidalPositionalEncoding(nn.Module):
    """Classic additive sinusoidal PE (Vaswani et al.) -- added once to the
    token embedding before the encoder/decoder stacks, not touched per-layer."""

    def __init__(self, max_seq_len, emb_dim):
        super().__init__()
        pe = torch.zeros(max_seq_len, emb_dim)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, emb_dim, 2, dtype=torch.float) * -(np.log(10000.0) / emb_dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class RotaryEmbedding(nn.Module):
    """Rotary positional embedding (RoPE), applied to Q/K within one
    attention head. Only meaningful for self-attention, where query and key
    positions live in the same sequence's coordinate system -- attention.py
    never applies this to cross-attention for that reason."""

    def __init__(self, head_dim: int, max_seq_len: int, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int):
        t = torch.arange(seq_len, device=self.inv_freq.device).float()
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, seq_len: int, start_pos: int = 0):
        # start_pos matters for KV-cached incremental decoding (Phase 7b):
        # a newly generated token at absolute position p needs cos/sin for
        # position p, not position 0 -- RoPE has to know where in the full
        # sequence this token actually sits, even though only one token is
        # being processed this call.
        end = start_pos + seq_len
        if end > self.cos_cached.size(0):
            self._build_cache(end)
        return (
            self.cos_cached[start_pos:end].to(self.inv_freq.device),
            self.sin_cached[start_pos:end].to(self.inv_freq.device),
        )


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    # q, k: [batch, heads, seq_len, head_dim]; cos, sin: [seq_len, head_dim]
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    q_rot = q * cos + rotate_half(q) * sin
    k_rot = k * cos + rotate_half(k) * sin
    return q_rot, k_rot


def get_alibi_slopes(num_heads: int) -> list:
    """Press et al. (2021) geometric slope schedule. Closed-form for
    power-of-2 head counts; falls back to interpolation otherwise so this
    doesn't silently misbehave if a future config picks an odd head count."""

    def _power_of_2_slopes(n):
        start = 2 ** (-2 ** -(math.log2(n) - 3))
        return [start * (start ** i) for i in range(n)]

    if math.log2(num_heads).is_integer():
        return _power_of_2_slopes(num_heads)

    closest_pow2 = 2 ** math.floor(math.log2(num_heads))
    slopes = _power_of_2_slopes(closest_pow2)
    extra = get_alibi_slopes(2 * closest_pow2)[0::2][: num_heads - closest_pow2]
    return slopes + extra


class AlibiBias(nn.Module):
    """Linear-distance attention bias (ALiBi), added directly to attention
    scores in place of a positional embedding -- no token/position
    embedding table at all. bias[h, i, j] = -slope_h * |i - j|.

    Using |i - j| rather than a causal-only (i - j) form works uniformly
    for both encoder (bidirectional) and decoder (causal) self-attention:
    in the causal case every j > i entry the formula computes for is
    masked out by the subsequent-position mask anyway, so only the i >= j
    region -- where |i - j| == i - j -- ever reaches softmax. Like RoPE,
    never applied to cross-attention (see attention.py)."""

    def __init__(self, num_heads: int, max_seq_len: int):
        super().__init__()
        slopes = torch.tensor(get_alibi_slopes(num_heads), dtype=torch.float32)
        self.register_buffer("slopes", slopes, persistent=False)
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int):
        pos = torch.arange(seq_len, device=self.slopes.device)
        dist = (pos[None, :] - pos[:, None]).abs().float()  # [seq_len, seq_len]
        bias = -self.slopes[:, None, None] * dist[None, :, :]  # [heads, seq_len, seq_len]
        self.register_buffer("bias_cached", bias, persistent=False)

    def forward(self, seq_len_q: int, seq_len_k: int, start_pos: int = 0):
        # Same start_pos reasoning as RotaryEmbedding.forward: a cached
        # decode step's query is at absolute position start_pos, not 0.
        needed = max(start_pos + seq_len_q, seq_len_k)
        if needed > self.bias_cached.size(-1):
            self._build_cache(needed)
        return self.bias_cached[:, start_pos:start_pos + seq_len_q, :seq_len_k]
