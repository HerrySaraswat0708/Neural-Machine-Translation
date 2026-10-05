"""Continuous batching (the Orca / vLLM scheduling innovation): instead
of running a fixed batch of requests to completion before admitting new
ones (static batching), the scheduler checks *every* decode step whether
any active request has finished (evict it, free its resources
immediately) and whether the waiting queue can be admitted into the
freed slot right away -- rather than leaving that slot idle, and leaving
new arrivals queued, until the whole batch's slowest member finishes.

Scope: this benchmarks the SCHEDULING POLICY's effect on per-request
latency under a simulated stream of staggered arrivals -- the actual
mechanism that distinguishes continuous from static batching. Each tick
decodes its active requests with real model calls, looped per request
(reusing the KV-cache backends from Phase 7c) rather than packed into one
ragged batched tensor op across requests of different current lengths --
that's a further, separate throughput optimization (padding + masking a
variable-length batch into a single attention call) a production engine
would layer on top of this scheduling policy, not something the
admit/evict policy itself requires to demonstrate its latency benefit.

Time model: ticks are logical steps, not wall-clock seconds -- one tick
== one decode step for whatever is active. Request arrival times are
given in ticks, a standard simplification for reproducible scheduling
simulations (real wall-clock throughput is measured separately, around
the whole run).
"""
import torch

from serving.speculative import _propose
from tokenization.base import SOS_ID, EOS_ID


class Request:
    def __init__(self, request_id, src_ids, arrival_tick, max_len):
        self.id = request_id
        self.src_ids = src_ids
        self.arrival_tick = arrival_tick
        self.max_len = max_len
        self.admitted_tick = None
        self.finish_tick = None
        self.generated = [SOS_ID]
        self.finished = False
        self.enc_out = None
        self.src_mask = None
        self.step_count = 0
        self._cross_caches = {}
        # Only used in speculative mode (step_speculative) -- the draft
        # model's own encoder output for this request.
        self.draft_enc_out = None
        self.draft_src_mask = None
        # Only used with chunked prefill (see serving/chunked_prefill.py) --
        # tracks how many of the target model's encoder layers have run so
        # far on this request's (partially-encoded) enc_out.
        self.prefilling = False
        self.prefill_layer_idx = 0

    def start_chunked_prefill(self, model, tick, draft_model=None):
        """Begins chunked prefill (serving/chunked_prefill.py): embeds the
        source and stops before the encoder-layer loop, leaving
        continue_chunked_prefill to advance it a few layers per scheduler
        tick instead of running the whole encoder in one atomic call. The
        draft model's own (tiny, 2-layer) encoding is computed fully here,
        unchunked -- chunking it wasn't worth the added bookkeeping for
        this project's scope."""
        from serving.chunked_prefill import start_encode
        device = next(model.parameters()).device
        self.enc_out, self.src_mask = start_encode(model, self.src_ids, device)
        self.prefill_layer_idx = 0
        self.prefilling = True
        self.admitted_tick = tick
        if draft_model is not None:
            src_tensor = torch.tensor([self.src_ids], dtype=torch.long, device=device)
            with torch.no_grad():
                self.draft_enc_out, self.draft_src_mask = draft_model.encode(src_tensor)

    def continue_chunked_prefill(self, model, chunk_layers):
        """Advances this request's encoder pass by up to chunk_layers
        layers. Sets self.prefilling = False once the full encoder stack
        has run -- the request is then ready to decode, with an enc_out
        bit-identical to what a single non-chunked encode() call would
        have produced (verified in benchmark_new_optimizations.py)."""
        from serving.chunked_prefill import continue_encode
        num_layers = len(model.encoder)
        end = min(self.prefill_layer_idx + chunk_layers, num_layers)
        self.enc_out = continue_encode(model, self.enc_out, self.src_mask, self.prefill_layer_idx, end)
        self.prefill_layer_idx = end
        if self.prefill_layer_idx >= num_layers:
            self.prefilling = False

    def admit(self, model, tick, draft_model=None):
        device = next(model.parameters()).device
        src_tensor = torch.tensor([self.src_ids], dtype=torch.long, device=device)
        with torch.no_grad():
            self.enc_out, self.src_mask = model.encode(src_tensor)
            if draft_model is not None:
                # Same bpe_joint vocabulary serves both models (Phase 7e),
                # so the same encoded src_tensor is valid input to either.
                self.draft_enc_out, self.draft_src_mask = draft_model.encode(src_tensor)
        self.admitted_tick = tick

    def admit_from_prefix_cache(self, model, tick, enc_out, src_mask, draft_model=None):
        """Prefix-cache hit (see InferenceEngine's prefix_cache): reuses a
        previously-computed encoder output verbatim instead of calling
        model.encode() again -- exact for repeated identical source
        sentences (a common real pattern: FAQ-style or template phrases
        translated many times), not an approximation. The draft model's
        own encoding is still computed fresh here (its encoder is tiny --
        2 layers, 1.78M params -- caching it separately wasn't worth the
        added bookkeeping for this project's scope)."""
        self.enc_out, self.src_mask = enc_out, src_mask
        if draft_model is not None:
            device = next(model.parameters()).device
            src_tensor = torch.tensor([self.src_ids], dtype=torch.long, device=device)
            with torch.no_grad():
                self.draft_enc_out, self.draft_src_mask = draft_model.encode(src_tensor)
        self.admitted_tick = tick

    def step_speculative(self, target_model, draft_model, backend, k):
        """One speculative round (Phase 7e) through an external paged
        backend (Phase 7c) instead of speculative_decode()'s private
        growing-tensor cache -- same accept/reject rule and the same
        cache invariant (backend cache always excludes the current last
        confirmed token; it's re-derived fresh as next round's bridge),
        so a request can run speculative decoding while sharing the same
        paged memory pool and scheduler as plain (non-speculative)
        requests admitted alongside it."""
        device = next(target_model.parameters()).device
        remaining = self.max_len - len(self.generated)
        this_k = min(k, remaining)
        if this_k <= 0:
            self.finished = True
            return

        draft_tokens = _propose(draft_model, self.generated, self.draft_enc_out, self.draft_src_mask, this_k, device)

        verify_input = [self.generated[-1]] + draft_tokens
        start_pos = self.step_count  # == backend's seq_len for this request, by invariant
        gathered_kv = [
            (backend.gather(self.id, layer_idx), self._cross_caches.get(layer_idx))
            for layer_idx in range(len(target_model.decoder))
        ]
        verify_tensor = torch.tensor([verify_input], dtype=torch.long, device=device)
        with torch.no_grad():
            target_logits, new_kv_cache = target_model.decode(
                verify_tensor, self.enc_out, self.src_mask,
                kv_cache=gathered_kv, start_pos=start_pos, use_cache=True)

        matched = []
        rejection_tok = None
        hit_eos = False
        for i, draft_tok in enumerate(draft_tokens):
            target_pred = target_logits[0, i].argmax().item()
            if target_pred == draft_tok:
                matched.append(draft_tok)
                if draft_tok == EOS_ID:
                    hit_eos = True
                    break
            else:
                rejection_tok = target_pred
                break

        self.generated.extend(matched)
        if rejection_tok is not None:
            self.generated.append(rejection_tok)
            hit_eos = rejection_tok == EOS_ID
        persist_count = len(matched) + (1 if rejection_tok is not None else 0)
        for j in range(persist_count):
            abs_pos = start_pos + j
            for layer_idx, (self_cache_layer, _) in enumerate(new_kv_cache):
                K_full, V_full = self_cache_layer
                backend.append(self.id, layer_idx,
                                K_full[:, :, abs_pos:abs_pos + 1, :],
                                V_full[:, :, abs_pos:abs_pos + 1, :])
            backend.advance(self.id)
        for layer_idx, (_, cross_cache_layer) in enumerate(new_kv_cache):
            self._cross_caches[layer_idx] = cross_cache_layer

        self.step_count = start_pos + persist_count

        if hit_eos or len(self.generated) >= self.max_len:
            self.finished = True

    def step(self, model, backend, use_sdpa=False):
        """One decode step -- deliberately duplicates decoding.py's
        greedy_decode_with_backend's per-layer loop rather than reusing it
        directly: that function owns its own full generation loop
        start-to-finish, but a scheduler needs to pause after exactly one
        step per active request per tick, so each Request keeps its own
        state (cross-attention cache, step count) across calls instead.

        use_sdpa=True: both attention calls below use PyTorch's fused
        scaled_dot_product_attention (see serving/flash_attention.py and
        model/attention.py's Attention.forward) instead of the manual
        score/softmax/matmul path -- opt-in, defaults to the original
        behavior."""
        device = next(model.parameters()).device
        tgt_tensor = torch.tensor([[self.generated[-1]]], dtype=torch.long, device=device)
        tgt_pad_mask = (tgt_tensor == model.pad_idx).unsqueeze(-2).to(device)
        combined_tgt_mask = tgt_pad_mask.unsqueeze(1)
        with torch.no_grad():
            dec_out = model.tgt_embed(tgt_tensor)
            for layer_idx, layer in enumerate(model.decoder):
                past = backend.gather(self.id, layer_idx)
                self_out, new_self_cache = layer.masked_attn(
                    dec_out, dec_out, dec_out, combined_tgt_mask,
                    kv_cache=past, start_pos=self.step_count, use_cache=True, cache_mode="growing",
                    use_sdpa=use_sdpa)
                dec_out = layer.addnorm1(dec_out, self_out)

                new_k = new_self_cache[0][:, :, -1:, :]
                new_v = new_self_cache[1][:, :, -1:, :]
                backend.append(self.id, layer_idx, new_k, new_v)

                cross_out, new_cross_cache = layer.cross_attn(
                    dec_out, self.enc_out, self.enc_out, self.src_mask,
                    kv_cache=self._cross_caches.get(layer_idx), use_cache=True, cache_mode="static",
                    use_sdpa=use_sdpa)
                self._cross_caches[layer_idx] = new_cross_cache
                dec_out = layer.addnorm2(dec_out, cross_out)
                dec_out = layer.addnorm3(dec_out, layer.ffn(dec_out))

            backend.advance(self.id)
            logits = model.output_proj(dec_out)

        next_id = logits[0, -1].argmax().item()
        self.generated.append(next_id)
        self.step_count += 1
        if next_id == EOS_ID or self.step_count >= self.max_len - 1:
            self.finished = True


def run_static(model, backend, arrivals, max_batch_size, max_len, safety_ticks=100000):
    """Static batching: a new batch is only formed once ALL of the
    previous batch's requests have finished -- a fast request that
    finishes early still occupies (and wastes) its slot until the
    slowest member of its batch is done, and waiting requests are
    blocked from starting no matter how long they've been queued."""
    queue = list(arrivals)
    active, completed = [], []
    tick = 0
    while queue or active:
        if not active:
            while queue and len(active) < max_batch_size and queue[0][2] <= tick:
                rid, src_ids, arrival = queue.pop(0)
                req = Request(rid, src_ids, arrival, max_len)
                backend.allocate_request(rid)
                req.admit(model, tick)
                active.append(req)
        for req in active:
            if not req.finished:
                req.step(model, backend)
        if active and all(r.finished for r in active):
            for r in active:
                r.finish_tick = tick
                backend.free_request(r.id)
                completed.append(r)
            active = []
        tick += 1
        if tick > safety_ticks:
            raise RuntimeError("static batching simulation exceeded safety tick limit")
    return completed


def run_continuous(model, backend, arrivals, max_batch_size, max_len, safety_ticks=100000):
    """Continuous batching: evict finished requests and admit waiting
    ones every single tick, so a freed slot is reused immediately rather
    than sitting idle until its whole original cohort finishes."""
    queue = list(arrivals)
    active, completed = [], []
    tick = 0
    while queue or active:
        still_active = []
        for req in active:
            if req.finished:
                req.finish_tick = tick
                backend.free_request(req.id)
                completed.append(req)
            else:
                still_active.append(req)
        active = still_active

        while queue and len(active) < max_batch_size and queue[0][2] <= tick:
            rid, src_ids, arrival = queue.pop(0)
            req = Request(rid, src_ids, arrival, max_len)
            backend.allocate_request(rid)
            req.admit(model, tick)
            active.append(req)

        for req in active:
            req.step(model, backend)
        tick += 1
        if tick > safety_ticks:
            raise RuntimeError("continuous batching simulation exceeded safety tick limit")
    return completed
