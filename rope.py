import torch
import torch.nn as nn


class RotaryEmbedding(nn.Module):
    """Rotary positional embedding (RoPE), applied to Q/K within one attention
    head. Only meaningful for self-attention, where query and key positions
    live in the same sequence's coordinate system -- see model.py for where
    this is (and is not) wired in."""

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

    def forward(self, seq_len: int):
        if seq_len > self.cos_cached.size(0):
            self._build_cache(seq_len)
        return (
            self.cos_cached[:seq_len].to(self.inv_freq.device),
            self.sin_cached[:seq_len].to(self.inv_freq.device),
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
