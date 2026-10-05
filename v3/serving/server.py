"""Phase 7f: a thin async API in front of InferenceEngine (README's own
phrasing) -- just enough surface to submit a translation request and poll
for its result, so the engine's paging + continuous batching (+ optional
speculative decoding) are exercised through a real request/response cycle
rather than only a Python-level benchmark harness. No queueing/auth/rate
limiting beyond what InferenceEngine itself already does -- that's out of
scope for a portfolio project's serving demo.

The engine's scheduler (`_tick`) is driven by a background asyncio task
that loops as long as there's work, sleeping briefly when idle instead of
busy-spinning -- ticks are logical decode steps, not real time, so this
just pumps them as fast as the event loop allows.

Run:  uvicorn serving.server:app --reload   (from the v3/ directory)
"""
import asyncio

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

import manifest
from model import load_model
from serving.engine import InferenceEngine
from tokenization import TOKENIZER_REGISTRY

app = FastAPI(title="NMT v3 inference engine")

_engine: InferenceEngine | None = None
_pump_task: asyncio.Task | None = None
_pump_lock = asyncio.Lock()


def get_engine(target_variant="bpe_joint_rope_mha", draft_variant="bpe_joint_rope_mha_draft",
               use_speculative=True, max_batch_size=8, k=4):
    global _engine
    if _engine is not None:
        return _engine
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    entries = manifest.load()
    target_model, _ = load_model(entries[target_variant]["checkpoint_path"], device=device)
    draft_model = None
    if use_speculative:
        draft_model, _ = load_model(entries[draft_variant]["checkpoint_path"], device=device)
    src_tok = TOKENIZER_REGISTRY["bpe_joint"].load(entries[target_variant]["vocab_dir"] + "/src")
    tgt_tok = TOKENIZER_REGISTRY["bpe_joint"].load(entries[target_variant]["vocab_dir"] + "/tgt")
    _engine = InferenceEngine(target_model, src_tok, tgt_tok, max_batch_size=max_batch_size,
                               draft_model=draft_model, k=k)
    return _engine


async def _pump():
    """Advances the engine's scheduler one tick at a time as long as
    there's admitted or waiting work, yielding to the event loop between
    ticks so concurrent requests can be submitted mid-run instead of only
    between fully-idle periods."""
    engine = get_engine()
    while True:
        async with _pump_lock:
            has_work = bool(engine.waiting or engine.active)
            if has_work:
                engine._tick()
        if not has_work:
            await asyncio.sleep(0.01)
        else:
            await asyncio.sleep(0)


@app.on_event("startup")
async def startup():
    global _pump_task
    get_engine()
    _pump_task = asyncio.create_task(_pump())


class TranslateRequest(BaseModel):
    text: str


class TranslateResponse(BaseModel):
    request_id: int


@app.post("/translate", response_model=TranslateResponse)
async def translate(req: TranslateRequest):
    engine = get_engine()
    async with _pump_lock:
        rid = engine.submit(req.text)
    return TranslateResponse(request_id=rid)


@app.get("/result/{request_id}")
async def result(request_id: int):
    engine = get_engine()
    if request_id in engine.results:
        return {"request_id": request_id, "status": "done", "translation": engine.results[request_id]}
    pending_ids = {r.id for r in engine.active} | {rid for rid, _, _ in engine.waiting}
    if request_id in pending_ids:
        return {"request_id": request_id, "status": "pending"}
    raise HTTPException(status_code=404, detail="unknown request_id")


@app.get("/health")
async def health():
    engine = get_engine()
    return {
        "active": len(engine.active),
        "waiting": len(engine.waiting),
        "completed": len(engine.results),
        "speculative": engine.draft_model is not None,
    }
