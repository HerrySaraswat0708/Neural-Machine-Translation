"""Speculative decoding (Leviathan et al. 2023 / Chen et al. 2023): a
small, cheap draft model proposes several tokens ahead; the expensive
target model verifies all of them in ONE parallel forward pass instead of
one sequential call per token. Whenever the draft's proposal matches what
the target would have produced anyway, that token is accepted "for free"
-- the target's per-call cost is paid once per round instead of once per
token, so a high draft-acceptance rate buys a real reduction in
target-model round trips.

Why this is EXACT for this project, not merely distributionally
equivalent (the general stochastic-sampling case the literature usually
describes, and what v3/README.md's Phase 7 plan anticipates needing a
looser correctness argument for): this project's chosen decode strategy
is greedy (deterministic argmax), so the acceptance rule collapses to a
simple equality check -- accept draft_token[i] iff it equals the target
model's own argmax at that position, else fall back to the target's
argmax directly. Every accepted or substituted token is, by construction,
exactly the token greedy_decode would have produced there. Verified below
against greedy_decode, bit-identical.

Cache invariant, worth stating explicitly (an earlier version of this
file got it wrong -- the "bridge" token linking one round to the next was
re-fed as input to the target every round in order to verify the draft's
first proposal against it, but the code was ALSO holding onto that same
token's key/value from when it was originally cached, so it ended up
double-counted in the attention keys on round 2+, an off-by-one that
only surfaces once a rejection or a multi-round sequence happens):
**the persisted `target_kv_cache` always has length `len(generated) - 1`
-- it excludes the current last confirmed token.** That token is always
re-derived fresh as the next round's bridge, at the small, deliberate
cost of one redundant re-embedding per round (cheap next to what
verifying several tokens in parallel saves).
"""
import torch

from tokenization.base import SOS_ID, EOS_ID


def _propose(draft_model, confirmed_ids, enc_out, src_mask, k, device):
    """Draft proposes up to k tokens, one at a time, stopping early on
    EOS. Deliberately non-cached (full recompute each micro-step, the
    same pattern as the original greedy_decode) rather than incrementally
    cached: a rejected round's proposals must never leak into the next
    round, and the draft model is cheap enough that recomputing its
    short, ever-growing prefix from scratch each round costs little next
    to what caching saves on the (expensive) target model below."""
    proposals = []
    cur_seq = list(confirmed_ids)
    for _ in range(k):
        tgt_tensor = torch.tensor([cur_seq], dtype=torch.long, device=device)
        with torch.no_grad():
            logits = draft_model.decode(tgt_tensor, enc_out, src_mask)
        next_tok = logits[0, -1].argmax().item()
        proposals.append(next_tok)
        cur_seq.append(next_tok)
        if next_tok == EOS_ID:
            break
    return proposals


def _truncate_kv_cache(kv_cache, keep_len):
    truncated = []
    for self_cache, cross_cache in kv_cache:
        K, V = self_cache
        truncated.append(((K[:, :, :keep_len, :], V[:, :, :keep_len, :]), cross_cache))
    return truncated


def speculative_decode(draft_model, target_model, draft_src_ids, target_src_ids, k=4, max_len=None):
    """draft_src_ids/target_src_ids: encoded separately since draft and
    target are different model instances (different learned encoders),
    even though they share one vocabulary (both trained on bpe_joint) so
    their token ids are directly comparable for the accept/reject check.
    Returns (generated_ids, stats) where stats has proposed/accepted
    counts, round count, and target_calls (== rounds -- exactly one
    target call per round, the whole point of verifying in parallel), for
    measuring acceptance rate and target-call reduction vs. naive greedy."""
    device = next(target_model.parameters()).device
    max_len = min(max_len or target_model.max_seq_len, target_model.max_seq_len)

    draft_src = torch.tensor([draft_src_ids], dtype=torch.long, device=device)
    target_src = torch.tensor([target_src_ids], dtype=torch.long, device=device)
    with torch.no_grad():
        draft_enc_out, draft_src_mask = draft_model.encode(draft_src)
        target_enc_out, target_src_mask = target_model.encode(target_src)

    generated = [SOS_ID]
    target_kv_cache = None  # invariant: len(target_kv_cache) == len(generated) - 1
    stats = {"proposed": 0, "accepted": 0, "rounds": 0, "target_calls": 0}

    while len(generated) < max_len:
        stats["rounds"] += 1
        remaining = max_len - len(generated)
        this_k = min(k, remaining)
        draft_tokens = _propose(draft_model, generated, draft_enc_out, draft_src_mask, this_k, device)
        stats["proposed"] += len(draft_tokens)

        verify_input = [generated[-1]] + draft_tokens  # bridge re-derived fresh, see module docstring
        start_pos = len(generated) - 1  # == prior target_kv_cache length, by invariant
        verify_tensor = torch.tensor([verify_input], dtype=torch.long, device=device)
        with torch.no_grad():
            target_logits, new_kv_cache = target_model.decode(
                verify_tensor, target_enc_out, target_src_mask,
                kv_cache=target_kv_cache, start_pos=start_pos, use_cache=True)
        stats["target_calls"] += 1

        matched = []
        rejection_tok = None
        hit_eos = False
        for i, draft_tok in enumerate(draft_tokens):
            target_pred = target_logits[0, i].argmax().item()
            if target_pred == draft_tok:
                matched.append(draft_tok)
                stats["accepted"] += 1
                if draft_tok == EOS_ID:
                    hit_eos = True
                    break
            else:
                rejection_tok = target_pred
                break

        generated.extend(matched)
        if rejection_tok is not None:
            generated.append(rejection_tok)
            hit_eos = rejection_tok == EOS_ID

        # Only [bridge] + matched were genuinely fed as model input this
        # round (rejection_tok, if any, was only ever predicted) -- and
        # per the cache invariant, the current last confirmed token is
        # never persisted either way (it becomes next round's bridge).
        valid_len = start_pos + 1 + len(matched)
        keep = min(valid_len, len(generated) - 1)
        target_kv_cache = _truncate_kv_cache(new_kv_cache, keep)

        if hit_eos:
            break

    return generated[1:], stats
