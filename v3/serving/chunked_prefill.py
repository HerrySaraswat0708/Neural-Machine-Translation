"""Chunked prefill: spreads a request's ENCODER forward pass ("prefill")
across multiple scheduler ticks -- a fixed number of encoder layers per
tick -- instead of running all layers in one atomic call. Without this, a
newly-admitted request's prefill blocks the scheduler for an entire tick,
during which every other already-active request's decode step is delayed
(head-of-line blocking) -- exactly the problem chunked prefill exists to
fix in real LLM serving (vLLM, Sarathi-Serve).

Why this is chunked by LAYER, not by sequence POSITION (the way chunked
prefill works for a causal decoder-only LLM's prompt): this project's
encoder is bidirectional -- full, non-causal self-attention, so every
source token's representation legitimately depends on every OTHER source
token, including ones later in the sequence. Splitting the encoder's INPUT
SEQUENCE into position chunks the way a causal decoder's prefill can
(chunk 1 attends only to itself, chunk 2 attends to chunk 1 + itself, etc.)
would silently change what the encoder computes: an early chunk's tokens
would never get to attend to later-chunk tokens they were trained to see,
producing a genuinely DIFFERENT result, not a pure speed optimization --
the same exactness bar this project holds every other Phase 7 technique to
would be broken. Chunking by LAYER instead preserves the encoder's exact
bidirectional computation (identical final output for identical input,
regardless of how many ticks the layer stack is split across -- verified
bit-identical, see benchmark_new_optimizations.py) while still spreading
its wall-clock cost across multiple scheduler ticks, which is the actual
scheduling property chunked prefill is for.
"""
import torch


def start_encode(model, src_ids, device):
    """Embeds the source and returns the initial state for chunked
    encoding -- exactly what Transformer.encode() computes before its own
    internal encoder-layer loop starts."""
    src_tensor = torch.tensor([src_ids], dtype=torch.long, device=device)
    with torch.no_grad():
        src_mask = (src_tensor == model.pad_idx).unsqueeze(-2).unsqueeze(-2).to(device)
        enc_out = model.src_embed(src_tensor)
    return enc_out, src_mask


def continue_encode(model, enc_out, src_mask, start_layer, end_layer):
    """Runs encoder layers [start_layer, end_layer) on the current partial
    state -- bypasses Transformer.encode()'s own internal loop (which owns
    the whole pass end-to-end) so a caller can interrupt/resume it across
    scheduler ticks. Encoder layers are purely sequential with no
    cross-call state beyond enc_out itself, so running them in several
    separate calls covering disjoint contiguous layer ranges is
    mathematically identical to one call over all of them."""
    with torch.no_grad():
        for layer in model.encoder[start_layer:end_layer]:
            enc_out = layer(enc_out, src_mask)
    return enc_out
