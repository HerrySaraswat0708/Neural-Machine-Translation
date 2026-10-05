import torch
import torch.nn as nn
import torch.nn.functional as F

from model.positional import RotaryEmbedding, AlibiBias, apply_rotary_pos_emb

ATTN_TYPES = ("mha", "mqa", "gqa")


def num_kv_heads_for(attn_type: str, num_heads: int) -> int:
    """The one place attn_type -> num_kv_heads mapping lives, so train.py,
    checkpoint configs, and KV-cache-size math (v3/README.md Phase 5) all
    agree. MHA: one KV head per query head (today's v1 behaviour). MQA: a
    single KV head shared by every query head (maximum cache compression).
    GQA: query heads share KV heads in groups of 4 -- the middle ground
    real GQA models (e.g. Llama 2 70B) use."""
    if attn_type == "mha":
        return num_heads
    if attn_type == "mqa":
        return 1
    if attn_type == "gqa":
        return max(1, num_heads // 4)
    raise ValueError(f"Unknown attn_type '{attn_type}', expected one of {ATTN_TYPES}")


class Attention(nn.Module):
    """Scaled dot-product attention generalized over MHA/MQA/GQA via
    num_kv_heads (num_kv_heads == num_heads is plain MHA), plus optional
    RoPE or ALiBi positional signal injected at the score level. Q always
    has num_heads heads; K/V have num_kv_heads heads and are
    repeat_interleave'd up to num_heads right before the QK^T matmul --
    that's the entire mechanistic difference from MHA, everything else
    (masking, softmax, output projection) is identical."""

    def __init__(self, emb_dim, num_heads, num_kv_heads=None, pe_type="none", max_seq_len=64, dropout=0.1):
        super().__init__()
        assert emb_dim % num_heads == 0
        num_kv_heads = num_kv_heads or num_heads
        assert num_heads % num_kv_heads == 0, "num_heads must be a multiple of num_kv_heads (GQA groups)"

        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.num_groups = num_heads // num_kv_heads
        self.head_dim = emb_dim // num_heads
        self.pe_type = pe_type

        self.W_q = nn.Linear(emb_dim, num_heads * self.head_dim)
        self.W_k = nn.Linear(emb_dim, num_kv_heads * self.head_dim)
        self.W_v = nn.Linear(emb_dim, num_kv_heads * self.head_dim)
        self.W_o = nn.Linear(num_heads * self.head_dim, emb_dim)
        self.dropout = nn.Dropout(dropout)

        self.rotary = RotaryEmbedding(self.head_dim, max_seq_len) if pe_type == "rope" else None
        self.alibi = AlibiBias(num_heads, max_seq_len) if pe_type == "alibi" else None

    def _split_heads(self, x, heads):
        b, t, _ = x.shape
        return x.view(b, t, heads, self.head_dim).transpose(1, 2)  # [b, heads, t, head_dim]

    def forward(self, q_in, k_in, v_in, mask=None, return_attn=False,
                kv_cache=None, start_pos=0, use_cache=False, cache_mode="growing",
                use_sdpa=False):
        """kv_cache/start_pos/use_cache/cache_mode are all opt-in (Phase
        7b) -- every existing call site (unchanged default args) gets
        exactly the original behavior and return signature.

        cache_mode="growing" (self-attention): kv_cache holds everything
        generated so far; this call's new K/V get computed and appended.
        cache_mode="static" (cross-attention): encoder output never
        changes across decode steps, so once kv_cache is populated its K/V
        are reused as-is and k_in/v_in (still =enc_out, unchanged) are
        never re-projected -- only Q (from the decoder's current state)
        is recomputed every step.

        use_sdpa=True: replaces the manual score/softmax/matmul block
        below with torch.nn.functional.scaled_dot_product_attention (see
        serving/flash_attention.py for why -- PyTorch dispatches this to a
        real fused FlashAttention-2 CUDA kernel on this project's A6000s).
        Opt-in and deliberately scoped to what the inference engine
        actually needs: unsupported together with return_attn (a fused
        kernel never materializes the attention matrix, so there's nothing
        to return -- attention-heatmap visualization stays on the manual
        path) or with ALiBi (its additive per-position bias isn't wired
        through here, since the engine only ever serves the winning
        RoPE-based model -- a deliberate scope limit, not a bug, matching
        how sinusoidal PE was already excluded from KV-cache decoding for
        the same kind of reason)."""
        batch_size = q_in.size(0)
        Q = self._split_heads(self.W_q(q_in), self.num_heads)      # [b, H, Tq, D]

        if cache_mode == "static" and kv_cache is not None:
            K, V = kv_cache
        else:
            K_new = self._split_heads(self.W_k(k_in), self.num_kv_heads)   # [b, Hkv, Tk, D]
            V_new = self._split_heads(self.W_v(v_in), self.num_kv_heads)   # [b, Hkv, Tk, D]

            if self.rotary is not None:
                # Self-attention only (see class docstring / positional.py).
                # Only the NEW keys get rotated here -- cached keys were
                # already rotated at their own true positions when they
                # were first computed, in an earlier call.
                cos, sin = self.rotary(Q.size(2), start_pos=start_pos)
                Q, K_new = apply_rotary_pos_emb(Q, K_new, cos, sin)

            if cache_mode == "growing" and kv_cache is not None:
                past_K, past_V = kv_cache
                K = torch.cat([past_K, K_new], dim=2)
                V = torch.cat([past_V, V_new], dim=2)
            else:
                K, V = K_new, V_new

        new_cache = (K, V)

        if self.num_groups > 1:
            K_rep = K.repeat_interleave(self.num_groups, dim=1)  # [b, H, Tk, D]
            V_rep = V.repeat_interleave(self.num_groups, dim=1)
        else:
            K_rep, V_rep = K, V

        if use_sdpa:
            assert not return_attn, "use_sdpa=True can't also return_attn -- a fused kernel never materializes the attention matrix"
            assert self.alibi is None, "use_sdpa=True doesn't support ALiBi's additive bias -- deliberate scope limit, see docstring"
            # This project's own mask convention is "1 = blocked"
            # (masked_fill(mask == 1, ...)); SDPA's boolean convention is
            # the opposite ("True = attend"), so translate it directly
            # rather than assume it's always empty.
            # mask's last dim is often a broadcast placeholder (size 1,
            # stride 0) under the manual path's masked_fill, which happily
            # broadcasts -- but SDPA's fused kernel needs a real,
            # materialized last dimension matching Tk, not a broadcast
            # view, or it raises "last dimension must be contiguous".
            if mask is not None:
                sdpa_mask = (mask != 1).expand(Q.size(0), Q.size(1), Q.size(2), K_rep.size(2)).contiguous()
            else:
                sdpa_mask = None
            out = F.scaled_dot_product_attention(
                Q, K_rep, V_rep, attn_mask=sdpa_mask,
                dropout_p=self.dropout.p if self.training else 0.0,
            )
        else:
            scores = Q @ K_rep.transpose(-2, -1) / (self.head_dim ** 0.5)  # [b, H, Tq, Tk]

            if self.alibi is not None:
                scores = scores + self.alibi(Q.size(2), K_rep.size(2), start_pos=start_pos)[None, :, :, :]

            if mask is not None:
                scores = scores.masked_fill(mask == 1, value=-1e9)

            attn = torch.softmax(scores, dim=-1)
            attn = self.dropout(attn)
            out = attn @ V_rep  # [b, H, Tq, D]

        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.head_dim)
        output_final = self.W_o(out)

        if use_cache:
            if return_attn:
                return output_final, attn, new_cache
            return output_final, new_cache
        if return_attn:
            return output_final, attn
        return output_final

    