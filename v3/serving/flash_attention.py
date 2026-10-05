"""Flash-Attention-accelerated attention for the inference engine's decode
path, via PyTorch's built-in torch.nn.functional.scaled_dot_product_attention
(SDPA) -- not a hand-rolled kernel. This project's from-scratch ethos
deliberately stops here: a real fused, tiled, IO-aware attention kernel
needs custom CUDA; reimplementing one by hand in pure PyTorch wouldn't
actually be faster -- it would just be a slower, less numerically careful
re-derivation of what SDPA already does correctly. PyTorch dispatches SDPA
to a real FlashAttention-2 CUDA kernel automatically on Ampere+ GPUs
(compute capability >= 8.0 -- this project's A6000s are 8.6) whenever the
call shape/dtype is supported, with no separate installation or custom
kernel code required -- this IS the library, exactly as requested.

The actual integration point is model/attention.py's Attention.forward
(`use_sdpa=True`), threaded through from InferenceEngine so every existing
training/eval call site (unchanged default `use_sdpa=False`) keeps its
original behavior byte-for-byte. This module exists to hold the design
rationale in one place, not new attention math.

Why SDPA is NOT bit-identical to this project's manual eager attention --
and why that's expected, not a bug: Flash Attention computes the exact
same mathematical operation (softmax(QK^T/sqrt(d))V) via a different
*algorithm* -- tiled, with an online/running softmax, fused into one
kernel so the full [seq_len, seq_len] attention matrix is never
materialized in memory. Floating-point addition is not associative, so
summing softmax-weighted values in a different order (tile-by-tile vs.
all at once) produces tiny rounding differences. This is a NUMERICALLY
EQUIVALENT reformulation, verified *close* (allclose within float
tolerance, and argmax-identical for actual generated tokens) rather than
bit-exact -- unlike KV caching, continuous batching, speculative decoding,
chunked prefill, and prefix caching, which only ever change WHAT gets
computed/scheduled/cached/reused, never the arithmetic itself, and so are
all verified bit-for-bit exact instead. Knowing which category a given
optimization falls into -- and why -- is itself part of understanding it.
"""
