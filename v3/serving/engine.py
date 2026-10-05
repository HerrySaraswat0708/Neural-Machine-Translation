"""Ties continuous batching, speculative decoding (optional), and the
serving-time optimizations in this directory into one serving loop with a
request-in / translation-out API -- the point where each piece, already
built and correctness-verified standalone, gets used together.

KV-cache backend: NaiveContiguousCache (serving/kv_cache_naive.py) --
each request gets its own contiguous per-layer buffer sized for
max_seq_len. An earlier version of this engine used a PagedAttention-style
block allocator here (the vLLM memory-paging mechanism); it was removed to
keep this project's serving-optimization set focused and each technique
clearly, individually benchmarked (see PROGRESS.md) rather than spread
thin across an ever-growing list. The naive backend is simpler to reason
about and sufficient at this project's request-concurrency scale; nothing
else in this engine depends on paging specifically -- continuous batching,
speculative decoding, chunked prefill, and prefix caching all work
identically against any backend satisfying the same 5-method interface
(allocate_request/free_request/append/gather/advance).

Design note on "every combination of optimizations on/off": continuous
batching isn't really a toggle here -- it's what the scheduler *is*;
turning it off means going back to 7a's naive per-request loop, already
benchmarked on its own terms. The real toggles this engine exposes are
speculative decoding (`draft_model=None` vs. a real draft model), chunked
prefill (`prefill_chunk_layers=None` vs. an int), and prefix caching
(`enable_prefix_cache`) -- each independently on/off, each separately
benchmarked in the comparison study.
"""
import torch

from serving.batching import Request
from serving.kv_cache_naive import NaiveContiguousCache


class InferenceEngine:
    def __init__(self, target_model, src_tok, tgt_tok, max_batch_size=8,
                 draft_model=None, k=4, prefill_chunk_layers=None, enable_prefix_cache=False,
                 use_sdpa=False):
        if use_sdpa and draft_model is not None:
            raise ValueError("use_sdpa and speculative decoding (draft_model) aren't wired together "
                              "-- step_speculative() calls Transformer.decode()'s own internal loop, "
                              "which doesn't thread use_sdpa through; deliberate scope limit, not a bug.")
        self.target_model = target_model
        self.draft_model = draft_model
        self.k = k
        self.src_tok = src_tok
        self.tgt_tok = tgt_tok
        self.max_batch_size = max_batch_size
        self.max_len = target_model.max_seq_len
        self.use_sdpa = use_sdpa
        # None (default) = admit synchronously, one full encode() call per
        # request, identical to the original engine's behavior. An int N =
        # spread each request's encoder pass over ceil(num_layers/N) ticks
        # (serving/chunked_prefill.py) instead.
        self.prefill_chunk_layers = prefill_chunk_layers
        self.enable_prefix_cache = enable_prefix_cache
        self.prefix_cache = {}  # tuple(src_ids) -> (enc_out, src_mask); only used if enable_prefix_cache

        device = next(target_model.parameters()).device
        num_layers = len(target_model.decoder)
        num_kv_heads = target_model.num_kv_heads
        head_dim = target_model.decoder[0].masked_attn.head_dim
        self.backend = NaiveContiguousCache(num_layers, num_kv_heads, head_dim, self.max_len, device)

        self.waiting = []      # [(request_id, src_ids, arrival_tick)]
        self.prefilling = []   # requests mid-chunked-prefill (admitted, not yet decoding)
        self.active = []
        self.results = {}      # request_id -> decoded text
        self._next_id = 0
        self.tick = 0

    def submit(self, text: str) -> int:
        from tokenization.base import EOS_ID
        rid = self._next_id
        self._next_id += 1
        src_ids = self.src_tok.encode(text) + [EOS_ID]
        self.waiting.append((rid, src_ids, self.tick))
        return rid

    def _tick(self):
        still_active = []
        for req in self.active:
            if req.finished:
                req.finish_tick = self.tick
                self.backend.free_request(req.id)
                self.results[req.id] = self.tgt_tok.decode(req.generated[1:])
            else:
                still_active.append(req)
        self.active = still_active

        # A request counts against max_batch_size whether it's actively
        # decoding or still mid-prefill -- both hold a backend slot.
        self.waiting.sort(key=lambda w: w[2])
        while (self.waiting and len(self.active) + len(self.prefilling) < self.max_batch_size
               and self.waiting[0][2] <= self.tick):
            rid, src_ids, arrival = self.waiting.pop(0)
            req = Request(rid, src_ids, arrival, self.max_len)
            self.backend.allocate_request(rid)

            cache_key = tuple(src_ids) if self.enable_prefix_cache else None
            if self.enable_prefix_cache and cache_key in self.prefix_cache:
                enc_out, src_mask = self.prefix_cache[cache_key]
                req.admit_from_prefix_cache(self.target_model, self.tick, enc_out, src_mask,
                                             draft_model=self.draft_model)
                self.active.append(req)
            elif self.prefill_chunk_layers is not None:
                req.start_chunked_prefill(self.target_model, self.tick, draft_model=self.draft_model)
                self.prefilling.append(req)
            else:
                req.admit(self.target_model, self.tick, draft_model=self.draft_model)
                if self.enable_prefix_cache:
                    self.prefix_cache[cache_key] = (req.enc_out, req.src_mask)
                self.active.append(req)

        still_prefilling = []
        for req in self.prefilling:
            req.continue_chunked_prefill(self.target_model, self.prefill_chunk_layers)
            if req.prefilling:
                still_prefilling.append(req)
            else:
                if self.enable_prefix_cache:
                    self.prefix_cache[tuple(req.src_ids)] = (req.enc_out, req.src_mask)
                self.active.append(req)
        self.prefilling = still_prefilling

        for req in self.active:
            if self.draft_model is not None:
                req.step_speculative(self.target_model, self.draft_model, self.backend, self.k)
            else:
                req.step(self.target_model, self.backend, use_sdpa=self.use_sdpa)
        self.tick += 1

    def run_until_idle(self, safety_ticks=100000):
        while self.waiting or self.prefilling or self.active:
            self._tick()
            if self.tick > safety_ticks:
                raise RuntimeError("InferenceEngine exceeded safety tick limit")

    def get_result(self, request_id: int) -> str:
        return self.results[request_id]

    def translate(self, text: str) -> str:
        """Convenience: submit one request and run the engine alone until
        it completes. For a single request this is equivalent to (and
        should reproduce bit-identical output to) whichever primitive the
        engine is configured with -- greedy_decode_with_backend or
        speculative_decode -- since with only one active request, the
        continuous-batching scheduler has nothing to interleave."""
        rid = self.submit(text)
        self.run_until_idle()
        return self.get_result(rid)
