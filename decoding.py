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
