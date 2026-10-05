import torch
import torch.nn.functional as F

from tokenization.base import SOS_ID, EOS_ID


def _capture_attention(model, generated_ids, enc_out, src_mask, device):
    """One extra forced-decode pass over the final generated sequence to get
    a clean [tgt_len, src_len] cross-attention matrix (head-averaged) for the
    heatmap, instead of stitching together per-step attention during
    generation."""
    if len(generated_ids) < 2:
        return None
    tgt_input = torch.tensor([generated_ids[:-1]], dtype=torch.long, device=device)
    with torch.no_grad():
        _, attn = model.decode(tgt_input, enc_out, src_mask, return_attn=True)
    return attn.mean(dim=1)[0].cpu()  # rows align with generated_ids[1:]


def greedy_decode(model, src_ids, max_len=None, return_attn=False):
    device = next(model.parameters()).device
    max_len = min(max_len or model.max_seq_len, model.max_seq_len)
    src_tensor = torch.tensor([src_ids], dtype=torch.long, device=device)

    model.eval()
    with torch.no_grad():
        enc_out, src_mask = model.encode(src_tensor)
        generated = [SOS_ID]
        for _ in range(max_len - 1):
            tgt_tensor = torch.tensor([generated], dtype=torch.long, device=device)
            logits = model.decode(tgt_tensor, enc_out, src_mask)
            next_id = logits[0, -1].argmax().item()
            generated.append(next_id)
            if next_id == EOS_ID:
                break

    attn = _capture_attention(model, generated, enc_out, src_mask, device) if return_attn else None
    return generated[1:], attn


def greedy_decode_cached(model, src_ids, max_len=None, return_attn=False):
    """Phase 7b: identical decoding logic to greedy_decode, but K/V for
    every previously-generated token are cached and incrementally
    extended instead of recomputed from scratch every step (see
    model/attention.py, model/transformer.py). Must produce BIT-IDENTICAL
    token ids to greedy_decode for the same model/input -- this is a pure
    speed optimization, not an approximation (v3/README.md Phase 7b)."""
    device = next(model.parameters()).device
    max_len = min(max_len or model.max_seq_len, model.max_seq_len)
    src_tensor = torch.tensor([src_ids], dtype=torch.long, device=device)

    model.eval()
    with torch.no_grad():
        enc_out, src_mask = model.encode(src_tensor)
        generated = [SOS_ID]
        kv_cache = None
        for step in range(max_len - 1):
            tgt_tensor = torch.tensor([[generated[-1]]], dtype=torch.long, device=device)
            logits, kv_cache = model.decode(
                tgt_tensor, enc_out, src_mask, kv_cache=kv_cache,
                start_pos=step, use_cache=True)
            next_id = logits[0, -1].argmax().item()
            generated.append(next_id)
            if next_id == EOS_ID:
                break

    attn = _capture_attention(model, generated, enc_out, src_mask, device) if return_attn else None
    return generated[1:], attn


def greedy_decode_with_backend(model, src_ids, backend, request_id, max_len=None):
    """Phase 7c: decodes through an external KV-cache backend (either
    serving/kv_cache_naive.py's NaiveContiguousCache or
    serving/paged_attention.py's PagedKVCache) instead of the simple
    growing-tensor cache greedy_decode_cached uses -- lets PagedAttention
    be exercised with REAL generation and checked for bit-identical
    correctness, not just modeled as a memory-accounting exercise.
    Bypasses Transformer.decode's own per-layer loop (which owns its
    cache end-to-end) to intercept each layer's newly-computed K/V and
    hand it to the external backend instead. Caller must have already
    called backend.allocate_request(request_id); freeing it afterwards is
    also the caller's job, since a real server keeps a request's blocks
    alive across many decode calls, not just one function call.

    No return_attn support (a benchmark/serving path has no need for
    attention-heatmap visualization) and no sinusoidal-PE support, same
    reasoning as greedy_decode_cached."""
    if model.pe_type == "sinusoidal":
        raise NotImplementedError("KV-cache decoding needs each new token's true "
                                   "absolute position; not implemented for sinusoidal PE.")
    device = next(model.parameters()).device
    max_len = min(max_len or model.max_seq_len, model.max_seq_len)
    src_tensor = torch.tensor([src_ids], dtype=torch.long, device=device)

    model.eval()
    with torch.no_grad():
        enc_out, src_mask = model.encode(src_tensor)
        generated = [SOS_ID]
        cross_caches = [None] * len(model.decoder)  # static cache, not paged -- see module docstring

        for step in range(max_len - 1):
            tgt_tensor = torch.tensor([[generated[-1]]], dtype=torch.long, device=device)
            tgt_pad_mask = (tgt_tensor == model.pad_idx).unsqueeze(-2).to(device)
            combined_tgt_mask = tgt_pad_mask.unsqueeze(1)
            dec_out = model.tgt_embed(tgt_tensor)

            for layer_idx, layer in enumerate(model.decoder):
                past = backend.gather(request_id, layer_idx)
                self_out, new_self_cache = layer.masked_attn(
                    dec_out, dec_out, dec_out, combined_tgt_mask,
                    kv_cache=past, start_pos=step, use_cache=True, cache_mode="growing")
                dec_out = layer.addnorm1(dec_out, self_out)

                new_k = new_self_cache[0][:, :, -1:, :]
                new_v = new_self_cache[1][:, :, -1:, :]
                backend.append(request_id, layer_idx, new_k, new_v)

                cross_out, new_cross_cache = layer.cross_attn(
                    dec_out, enc_out, enc_out, src_mask,
                    kv_cache=cross_caches[layer_idx], use_cache=True, cache_mode="static")
                cross_caches[layer_idx] = new_cross_cache
                dec_out = layer.addnorm2(dec_out, cross_out)
                dec_out = layer.addnorm3(dec_out, layer.ffn(dec_out))

            backend.advance(request_id)
            logits = model.output_proj(dec_out)
            next_id = logits[0, -1].argmax().item()
            generated.append(next_id)
            if next_id == EOS_ID:
                break

    return generated[1:], None


def beam_search_decode(model, src_ids, max_len=None, beam_size=5, length_penalty=0.6, return_attn=False):
    device = next(model.parameters()).device
    max_len = min(max_len or model.max_seq_len, model.max_seq_len)
    src_tensor = torch.tensor([src_ids], dtype=torch.long, device=device)

    def lp_score(seq, score):
        lp = ((5 + len(seq)) ** length_penalty) / (6 ** length_penalty)
        return score / lp

    model.eval()
    with torch.no_grad():
        enc_out, src_mask = model.encode(src_tensor)
        enc_out_beam = enc_out.repeat(beam_size, 1, 1)
        src_mask_beam = src_mask.repeat(beam_size, 1, 1, 1)

        beams = [([SOS_ID], 0.0)]
        completed = []

        for _ in range(max_len - 1):
            cur_size = len(beams)
            tgt_tensor = torch.tensor([b[0] for b in beams], dtype=torch.long, device=device)
            logits = model.decode(tgt_tensor, enc_out_beam[:cur_size], src_mask_beam[:cur_size])
            log_probs = F.log_softmax(logits[:, -1, :], dim=-1)

            k = min(beam_size, log_probs.size(-1))
            topk_logprob, topk_idx = log_probs.topk(k, dim=-1)

            candidates = []
            for i, (seq, score) in enumerate(beams):
                for j in range(k):
                    token = topk_idx[i, j].item()
                    new_seq = seq + [token]
                    new_score = score + topk_logprob[i, j].item()
                    (completed if token == EOS_ID else candidates).append((new_seq, new_score))

            if not candidates:
                break
            candidates.sort(key=lambda sc: lp_score(*sc), reverse=True)
            beams = candidates[:beam_size]

            if len(completed) >= beam_size:
                break

        finished = completed if completed else beams
        best_seq, _ = max(finished, key=lambda sc: lp_score(*sc))

    attn = _capture_attention(model, best_seq, enc_out, src_mask, device) if return_attn else None
    return best_seq[1:], attn


def sample_decode(model, src_ids, max_len=None, strategy="top_k", top_k=10, top_p=0.9,
                   temperature=1.0, return_attn=False, seed=None):
    device = next(model.parameters()).device
    max_len = min(max_len or model.max_seq_len, model.max_seq_len)
    src_tensor = torch.tensor([src_ids], dtype=torch.long, device=device)
    if seed is not None:
        torch.manual_seed(seed)

    model.eval()
    with torch.no_grad():
        enc_out, src_mask = model.encode(src_tensor)
        generated = [SOS_ID]
        for _ in range(max_len - 1):
            tgt_tensor = torch.tensor([generated], dtype=torch.long, device=device)
            logits = model.decode(tgt_tensor, enc_out, src_mask)[0, -1] / max(temperature, 1e-6)
            probs = F.softmax(logits, dim=-1)

            if strategy == "top_k":
                topk_probs, topk_idx = probs.topk(min(top_k, probs.size(-1)))
                topk_probs = topk_probs / topk_probs.sum()
                next_id = topk_idx[torch.multinomial(topk_probs, 1)].item()
            elif strategy == "top_p":
                sorted_probs, sorted_idx = torch.sort(probs, descending=True)
                cumulative = torch.cumsum(sorted_probs, dim=-1)
                cutoff = (cumulative > top_p).nonzero()
                cutoff_idx = cutoff[0].item() + 1 if len(cutoff) > 0 else len(sorted_probs)
                sorted_probs = sorted_probs[:cutoff_idx]
                sorted_idx = sorted_idx[:cutoff_idx]
                sorted_probs = sorted_probs / sorted_probs.sum()
                next_id = sorted_idx[torch.multinomial(sorted_probs, 1)].item()
            else:
                raise ValueError(f"Unknown sampling strategy '{strategy}'")

            generated.append(next_id)
            if next_id == EOS_ID:
                break

    attn = _capture_attention(model, generated, enc_out, src_mask, device) if return_attn else None
    return generated[1:], attn


DECODE_STRATEGIES = {
    "greedy": greedy_decode,
    "beam": beam_search_decode,
    "top_k": lambda model, src_ids, **kw: sample_decode(model, src_ids, strategy="top_k", **kw),
    "top_p": lambda model, src_ids, **kw: sample_decode(model, src_ids, strategy="top_p", **kw),
}


def decode(model, src_ids, strategy="greedy", **kwargs):
    if strategy not in DECODE_STRATEGIES:
        raise ValueError(f"Unknown decode strategy '{strategy}', expected one of {list(DECODE_STRATEGIES)}")
    return DECODE_STRATEGIES[strategy](model, src_ids, **kwargs)
