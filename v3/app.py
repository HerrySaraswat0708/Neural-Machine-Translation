import os
import sys

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS_DIR)

import math
import random
import time
import torch
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import manifest
from model import load_model
from tokenization import TOKENIZER_REGISTRY
from tokenization.base import EOS_ID
from decoding import decode as run_decode, greedy_decode, greedy_decode_cached

WINNER_ID = "bpe_joint_rope_mha"
BASELINE_ID = "bpe_rope_mha"
DRAFT_ID = "bpe_joint_rope_mha_draft"

DECODE_KWARGS = {
    "greedy": {},
    "beam": {"beam_size": 5, "length_penalty": 0.6},
    "top_k": {"top_k": 10, "seed": 42},
    "top_p": {"top_p": 0.9, "seed": 42},
}

SAMPLE_SENTENCES = [
    "i advised him to come back at once .",
    "you ve been very helpful .",
    "he slept well last night .",
    "why don t you mind your own business ?",
    "we can t help tom .",
]

# variant_id -> (round label, status). Status: "valid" (correct-capacity,
# trustworthy), "collapsed" (correct capacity but a genuine finding -- the
# char+sinusoidal rank-collapse bug), "broken" (pre-fix oversized config,
# not a fair result), "winner".
ROUND_TAGS = {
    "bpe_sinusoidal_mha": ("Tokenizer x PE sweep", "valid"),
    "bpe_rope_mha": ("Tokenizer x PE sweep", "valid"),
    "word_sinusoidal_mha": ("Tokenizer x PE sweep", "valid"),
    "word_rope_mha": ("Tokenizer x PE sweep", "valid"),
    "char_sinusoidal_mha": ("Tokenizer x PE sweep", "collapsed"),
    "char_rope_mha": ("Tokenizer x PE sweep", "valid"),
    "bpe_rope_mha_deeper": ("Tuning round 1", "valid"),
    "bpe_rope_mha_wider": ("Tuning round 1", "valid"),
    "bpe_rope_mha_widedeep": ("Tuning round 1", "valid"),
    "bpe_rope_mha_narrowffn": ("Tuning round 1", "valid"),
    "bpe_rope_mha_longtrain": ("Tuning round 1", "valid"),
    "bpe_rope_mha_lowdrop": ("Tuning round 1", "valid"),
    "bpe_rope_gqa": ("Tuning round 1 (attention)", "valid"),
    "bpe_rope_mha_widerlong": ("Tuning round 2 (combined)", "valid"),
    "bpe_rope_mha_narrowffnlong": ("Tuning round 2 (combined)", "valid"),
    "bpe_rope_mha_widernarrowffn": ("Tuning round 2 (combined)", "valid"),
    "bpe_rope_mha_widernarrowffnlong": ("Tuning round 2 (combined)", "valid"),
    "bpe_rope_mha_lowdroplong": ("Tuning round 2 (combined)", "valid"),
    "bpe_rope_gqa_long": ("Tuning round 2 (attention)", "valid"),
    "bpe_rope_mqa": ("Attention-type check", "valid"),
    "bpe_rope_mqa_long": ("Attention-type check", "valid"),
    "wordpiece_sinusoidal_mha": ("Breadth matrix completion", "valid"),
    "wordpiece_rope_mha": ("Breadth matrix completion", "valid"),
    "wordpiece_alibi_mha": ("Breadth matrix completion", "valid"),
    "bpe_alibi_mha": ("Breadth matrix completion", "valid"),
    "word_alibi_mha": ("Breadth matrix completion", "valid"),
    "char_alibi_mha": ("Breadth matrix completion", "valid"),
    "bpe_joint_rope_mha": ("Noun-preservation experiment (joint vocab)", "winner"),
    "bpe_rope_mha_noiseaug": ("Grammar-robustness experiment (noise aug)", "valid"),
}
STATUS_COLOR = {
    "winner": "#1baf7a",
    "valid": "#2a78d6",
    "collapsed": "#e34948",
    # "broken": "#898781",
}
STATUS_LABEL = {
    "winner": "🏆 Winner",
    "valid": "✅ Valid",
    "collapsed": "🐛 Collapsed (real finding)",
    # "broken": "⚠️ Broken config (not comparable)",
}

st.set_page_config(page_title="NMT", page_icon="🧭", layout="wide")


# ---------------------------------------------------------------- data ----

@st.cache_resource(show_spinner=False)
def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@st.cache_data(show_spinner=False, ttl=30)
def load_manifest():
    return manifest.load()


# Frontend is scoped to the core word/bpe x sinusoidal/RoPE breadth+tuning
# story -- char (see prior commit), ALiBi (a third PE included in the full
# sweep but not part of the main narrative), and the draft/bidirectional
# side-experiments are excluded from every results/architecture listing.
# Still fully present in results/manifest.json and PROGRESS.md -- just not
# surfaced in this UI.
_FRONTEND_EXCLUDED_IDS = {"bpe_joint_rope_mha_draft", "bpe_joint_rope_mha_bidi"}


def load_frontend_manifest():
    return {
        vid: e for vid, e in load_manifest().items()
        if e.get("tokenizer_type") != "char"
        and e.get("pe_type") != "alibi"
        and vid not in _FRONTEND_EXCLUDED_IDS
    }


@st.cache_data(show_spinner=False)
def read_arch_config(checkpoint_path):
    """Read just the saved architecture config out of a checkpoint, without
    keeping the (possibly large) state dict alive past this call."""
    full_path = os.path.join(THIS_DIR, checkpoint_path)
    if not os.path.exists(full_path):
        return {}
    ckpt = torch.load(full_path, map_location="cpu")
    cfg = dict(ckpt.get("config", {}))
    del ckpt
    return cfg


@st.cache_resource(show_spinner="Loading model...")
def load_variant(variant_id):
    entries = load_manifest()
    entry = entries[variant_id]
    device = get_device()
    checkpoint_path = os.path.join(THIS_DIR, entry["checkpoint_path"])
    model, cfg = load_model(checkpoint_path, device=device)
    tok_cls = TOKENIZER_REGISTRY[entry["tokenizer_type"]]
    vdir = os.path.join(THIS_DIR, entry["vocab_dir"])
    src_tok = tok_cls.load(os.path.join(vdir, "src"))
    tgt_tok = tok_cls.load(os.path.join(vdir, "tgt"))
    return model, src_tok, tgt_tok, cfg


def token_pieces(tok, ids):
    """Best-effort per-id display string, for attention-heatmap axis labels."""
    pieces = []
    for i in ids:
        if hasattr(tok, "_tok"):
            p = tok._tok.id_to_token(i)
            p = p.replace("Ġ", "▁") if p else f"<{i}>"  # byte-level space marker -> visible "_"
            pieces.append(p)
        elif hasattr(tok, "index2char"):
            pieces.append(tok.index2char.get(i, f"<{i}>"))
        elif hasattr(tok, "index2word"):
            pieces.append(tok.index2word.get(i, f"<{i}>"))
        else:
            pieces.append(str(i))
    return pieces


def _timed(fn, *args, **kwargs):
    """Wall-clock timing for one decode call, in ms. torch.cuda.synchronize()
    brackets the call so GPU kernels are actually finished before the clock
    stops -- without it, async CUDA dispatch would let the call "return"
    long before the work it queued is done, understating latency."""
    device = get_device()
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    result = fn(*args, **kwargs)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - t0) * 1000
    return result, elapsed_ms


def translate(variant_id, sentence, strategy="greedy", extra_kwargs=None):
    model, src_tok, tgt_tok, cfg = load_variant(variant_id)
    src_ids = src_tok.encode(sentence) + [EOS_ID]
    kwargs = dict(DECODE_KWARGS.get(strategy, {}))
    if extra_kwargs:
        kwargs.update(extra_kwargs)
    (out_ids, attn), elapsed_ms = _timed(
        run_decode, model, src_ids, strategy=strategy, return_attn=True, **kwargs
    )
    hyp = tgt_tok.decode(out_ids)
    return {
        "hyp": hyp, "out_ids": out_ids, "src_ids": src_ids, "attn": attn,
        "src_pieces": token_pieces(src_tok, src_ids),
        "tgt_pieces": token_pieces(tgt_tok, out_ids),
        "elapsed_ms": elapsed_ms,
    }


@st.cache_data(show_spinner=False)
def load_test_sentences():
    path = os.path.join(THIS_DIR, "..", "data", "splits", "test.tsv")
    pairs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            eng, _ = line.rstrip("\n").split("\t")
            pairs.append(eng)
    return pairs


def attention_entropy(attn):
    if attn is None or attn.numel() == 0:
        return None
    row_entropy = -(attn.clamp_min(1e-12) * attn.clamp_min(1e-12).log()).sum(dim=-1)
    uniform = math.log(attn.size(-1))
    return row_entropy.mean().item(), uniform


def build_results_df():
    entries = load_frontend_manifest()
    rows = []
    for vid, e in entries.items():
        round_label, status = ROUND_TAGS.get(vid, ("Other / untagged", "valid"))
        arch = read_arch_config(e.get("checkpoint_path", ""))
        rows.append({
            "variant": vid,
            "round": round_label,
            "status": status,
            "tokenizer": e.get("tokenizer_type"),
            "pe": e.get("pe_type"),
            "attn": e.get("attn_type"),
            "emb_dim": arch.get("emb_dim"),
            "layers": arch.get("num_layers"),
            "ff_dim": arch.get("ff_dim"),
            "dropout": arch.get("dropout"),
            "epochs": e.get("epochs"),
            "params": e.get("num_params"),
            "kv_cache_bytes/tok": e.get("kv_cache_bytes_per_token_fp32"),
            "train_acc": e.get("final_train_acc"),
            "eval_acc": e.get("final_eval_acc"),
            "bleu": (e.get("bleu") or {}).get("greedy"),
            "chrf": (e.get("chrf") or {}).get("greedy"),
            "ter": (e.get("ter") or {}).get("greedy"),
        })
    return pd.DataFrame(rows)


# ------------------------------------------------------------- sidebar ----

st.sidebar.title("🧭 NMT")
# st.sidebar.caption("English -> French, from tokenizer collapse to a tuned production config.")
entries_preview = load_manifest()
winner_entry = entries_preview.get(WINNER_ID, {})
winner_bleu = (winner_entry.get("bleu") or {}).get("greedy")
if winner_bleu:
    st.sidebar.metric("Current best BLEU", f"{winner_bleu:.2f}", help=f"{WINNER_ID}, greedy decode")

page = st.sidebar.radio(
    "Navigate",
    # ["🏠 Overview", "📊 Experiment Results", "🌐 Try It Live", "⚡ Inference Lab",
    #  "🐛 The Collapse Bug", "⚙️ Architecture Explorer"],
    ["📊 Experiment Results", "🌐 Live Demo", "⚡ Inference Optimization",
     "⚙️ Architecture Explorer"],
    label_visibility="collapsed",
)
st.sidebar.divider()
# st.sidebar.caption("Full written log: `v3/PROGRESS.md`")


# --------------------------------------------------------------- pages ----

def page_overview():
    st.title("English -> French, from collapse to a tuned config")
    st.markdown(
        "A tokenizer x positional-encoding x attention study that broke twice before it worked -- "
        "once from a transformer 6.6x too big for its data, once from a positional-encoding scheme "
        "whose signal couldn't survive six layers of self-attention. This walks through both, and "
        "the two rounds of architecture tuning that followed."
    )

    e = load_manifest().get(WINNER_ID, {})
    bleu = (e.get("bleu") or {}).get("greedy")
    chrf = (e.get("chrf") or {}).get("greedy")
    ter = (e.get("ter") or {}).get("greedy")
    arch = read_arch_config(e.get("checkpoint_path", ""))

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("BLEU", f"{bleu:.2f}" if bleu else "—")
    c2.metric("chrF++", f"{chrf:.2f}" if chrf else "—")
    c3.metric("TER", f"{ter:.2f}" if ter else "—", help="lower is better")
    c4.metric("Parameters", f"{e.get('num_params', 0) / 1e6:.2f}M")

    st.code(
        f"tokenizer:  bpe (8,000-token BPE vocab)\n"
        f"positional: RoPE\n"
        f"attention:  MHA\n"
        f"emb_dim:    {arch.get('emb_dim', '—')}\n"
        f"num_layers: {arch.get('num_layers', '—')}\n"
        f"ff_dim:     {arch.get('ff_dim', '—')}   (4x emb_dim -- corrected from the original 2048 / 16x)\n"
        f"dropout:    {arch.get('dropout', '—')}\n"
        f"epochs:     {e.get('epochs', '—')}\n"
        f"decode:     greedy",
        language="text",
    )

    st.subheader("The journey")
    steps = [
        ("0", "Collapse", "14 variants at Transformer-base scale (100M params) on a 106K-pair dataset. "
                            "12 of 14 collapsed to repeating one token. Root-caused via v1's own history: "
                            "identical config, identical collapse, months earlier."),
        ("1", "Fix + re-sweep", "emb_dim 512->128, num_layers 12->6. Clean 6-way tokenizer x PE sweep. "
                                  "RoPE beat sinusoidal in every pairing -- and char+sinusoidal collapsed "
                                  "again, differently: a real rank-1 encoder collapse, measured directly."),
        ("2", "Decoding sweep", "Greedy beat beam/top-k/top-p on the winning bpe+RoPE config -- the model "
                                  "isn't sharp enough at this scale for wider search to pay off."),
        ("3", "Tuning x2", "14 architecture/hyperparameter configs across two rounds. Depth never helped. "
                             "Reallocating an oversized 16x FFN ratio into a wider embedding, plus doubling "
                             "the training budget, compounded to +4.6 BLEU over the first correct baseline."),
        ("4", "Joint vocab", "Switched to one shared BPE vocabulary over English+French (mBART/NLLB's "
                               "approach). +1.15 more BLEU -- though the OOV-name problem it targeted turned "
                               "out to need embedding weight-tying too, not just a shared vocabulary."),
    ]
    cols = st.columns(5)
    for col, (n, title, body) in zip(cols, steps):
        with col:
            st.markdown(f"**{n} · {title}**")
            st.caption(body)

    with st.expander("Known gaps -- what's not fully validated yet"):
        st.markdown(
            "- **Proper-noun preservation** -- joint BPE vocab (this model) improved overall BLEU "
            "but did NOT fix preservation of genuinely unseen names (herry/zhorvaq/blipnak/quexinor "
            "still hallucinate) -- see **The Collapse Bug** page's sibling investigation in "
            "`PROGRESS.md`. Sharing the vocabulary doesn't tie the encoder/decoder embedding "
            "*weights*; that's the well-motivated next step, not yet built.\n"
            "- **YaRN** -- never in scope. It extends RoPE for length extrapolation; this corpus's "
            "sentences max out at 64 tokens, so there's no length-generalization problem for it to solve.\n"
            "- **Phase 7 (inference engine)** -- KV cache, PagedAttention, continuous batching, "
            "speculative decoding. The project's stated main focus. Not started -- this app benchmarks "
            "quality, not serving."
        )


def page_results():
    st.title("Experiment Results")
    df = build_results_df()

    # show_broken = st.checkbox("Show broken-config runs (pre-fix, not comparable)", value=False)
    # if not show_broken:
    #     df = df[df["status"] != "broken"]
    df = df[df["status"] != "broken"]
    metric = st.radio("Metric", ["bleu", "chrf", "ter", "eval_acc"], horizontal=True)
    ascending = metric == "ter"
    plot_df = df.dropna(subset=[metric]).sort_values(metric, ascending=ascending)

    fig = px.bar(
        plot_df, x="variant", y=metric, color="status",
        color_discrete_map=STATUS_COLOR,
        hover_data=["round", "tokenizer", "pe", "attn", "params", "epochs"],
        title=f"{metric.upper()} by variant",
    )
    fig.update_layout(xaxis_tickangle=-45, showlegend=True, height=480)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Parameter efficiency")
    eff_df = df.dropna(subset=["bleu", "params"])
    fig2 = px.scatter(
        eff_df, x="params", y="bleu", color="status", color_discrete_map=STATUS_COLOR,
        hover_name="variant", log_x=True,
        labels={"params": "Parameters (log scale)", "bleu": "BLEU"},
        title="BLEU vs. parameter count",
    )
    fig2.update_traces(marker=dict(size=11, line=dict(width=1, color="white")))
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Full table")
    st.dataframe(
        df.sort_values("bleu", ascending=False, na_position="last")
          .drop(columns=["round", "tokenizer", "status", "pe", "attn"]),
        use_container_width=True, hide_index=True,
        column_config={
            "bleu": st.column_config.NumberColumn(format="%.2f"),
            "chrf": st.column_config.NumberColumn(format="%.2f"),
            "ter": st.column_config.NumberColumn(format="%.2f"),
            "train_acc": st.column_config.NumberColumn(format="%.2f%%"),
            "eval_acc": st.column_config.NumberColumn(format="%.2f%%"),
            "params": st.column_config.NumberColumn(format="%d"),
        },
    )


def page_try_it():
    st.title("Live Demo")

    sentence = st.text_input("English sentence", value="My name is Herry .")

    frontend_ids = sorted(load_frontend_manifest().keys())
    default_selection = [WINNER_ID] if WINNER_ID in frontend_ids else frontend_ids[:1]
    variants = st.multiselect(
        "Architecture(s)", frontend_ids, default=default_selection,
        help=f"Defaults to the current best model ({WINNER_ID}). Pick 2 or more to compare "
             "side by side on the same sentence.",
    )

    strategy = st.selectbox("Decoding strategy", list(DECODE_KWARGS), index=0)
    extra_kwargs = {}
    if strategy == "beam":
        c1, c2 = st.columns(2)
        with c1:
            extra_kwargs["beam_size"] = st.slider("Beam size", 1, 10, DECODE_KWARGS["beam"]["beam_size"])
        with c2:
            extra_kwargs["length_penalty"] = st.slider(
                "Length penalty", 0.0, 2.0, DECODE_KWARGS["beam"]["length_penalty"], step=0.1,
                help="<1.0 favors shorter output, >1.0 favors longer output, 1.0 is neutral.",
            )

    if st.button("Translate", type="primary") and sentence.strip() and variants:
        from entity_pipeline import mask_entities, restore_entities, display_masked

        # Entity detection/masking is model-independent (spaCy runs once on
        # the ORIGINAL sentence, before any model sees it), so it's done
        # once here and reused for every selected architecture below --
        # each model gets the identical masked input, an apples-to-apples
        # comparison.
        masked, entity_records = mask_entities(sentence)
        if entity_records:
            st.caption(f"{len(entity_records)} entity(ies) detected -- masked, translated, restored:")
            st.code(display_masked(sentence, entity_records), language=None)
            st.dataframe(
                pd.DataFrame([
                    {"Slot": r["label"], "Detected as typed": r["original"]}
                    for r in entity_records
                ]),
                use_container_width=True, hide_index=True,
            )
            if len(entity_records) > 2:
                st.warning(
                    f"{len(entity_records)} entities in one sentence -- verified reliability "
                    "drops sharply past 2 (the model itself may drop a name from its own "
                    "translation; restoration can't recover text the model never generated)."
                )

        cols = st.columns(len(variants))
        for col, vid in zip(cols, variants):
            with col:
                st.markdown(f"**{vid}**")
                with st.spinner("Translating..."):
                    masked_out = translate(vid, masked, strategy, extra_kwargs)
                raw_translation = masked_out["hyp"]
                final = restore_entities(raw_translation, entity_records) if entity_records else raw_translation

                st.success(final)
                st.caption(f"⏱️ {masked_out['elapsed_ms']:.1f} ms ({strategy} decode, naive/uncached)")

                if entity_records:
                    with st.expander("Pipeline steps"):
                        st.write({
                            "input": sentence,
                            "shown_to_user (tags)": display_masked(sentence, entity_records),
                            "actually_sent_to_model (safe-name placeholders)": masked,
                            "raw_translation": raw_translation,
                            "final": final,
                        })

                attn = masked_out["attn"]
                if attn is not None and attn.numel() > 0:
                    # attn row i is the attention distribution used to
                    # predict tgt_pieces[i] -- one row per output token,
                    # same length. Axis labels reflect the MASKED sentence
                    # actually fed to the model (placeholder names, not the
                    # original entity text).
                    fig = go.Figure(data=go.Heatmap(
                        z=attn.numpy(),
                        x=masked_out["src_pieces"],
                        y=masked_out["tgt_pieces"],
                        colorscale="Blues",
                    ))
                    fig.update_layout(
                        title="Cross-attention", xaxis_title="source", yaxis_title="generated",
                        height=380, margin=dict(t=40),
                    )
                    st.plotly_chart(fig, use_container_width=True)

        if entity_records:
            st.caption(
                "[NOUN]/[PERSON_i]-style tags can't be sent to the model itself -- "
                "'[' and ']' aren't in this tokenizer's vocabulary, so the model would "
                "just see <unk> fragments and hallucinate (verified: 'my name is [NOUN] .' "
                "-> 'mon nom est en décor de reprises .'). The tags shown above are "
                "display-only; the model still sees a safe placeholder name it has proven "
                "it translates reliably."
            )


def page_collapse_bug():
    st.title("The Collapse Bug")
    st.markdown(
        "`char_sinusoidal_mha` trains to a healthy 69% held-out token accuracy, then at "
        "free-running generation time outputs the **same French sentence for every input**. "
        "`char_rope_mha` -- identical architecture, identical data -- doesn't. This page "
        "reproduces the finding live."
    )

    if st.button("Run 5 sample sentences through both models", type="primary"):
        rows = []
        entropies = {"char_sinusoidal_mha": [], "char_rope_mha": []}
        with st.spinner("Translating..."):
            for s in SAMPLE_SENTENCES:
                r_sin = translate("char_sinusoidal_mha", s, "greedy")
                r_rope = translate("char_rope_mha", s, "greedy")
                rows.append({"English": s, "char + sinusoidal": r_sin["hyp"], "char + RoPE": r_rope["hyp"]})
                for name, r in (("char_sinusoidal_mha", r_sin), ("char_rope_mha", r_rope)):
                    ent = attention_entropy(r["attn"])
                    if ent:
                        entropies[name].append(ent[0] / ent[1])  # normalized: 1.0 == exactly uniform

        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        identical = len(set(r["char + sinusoidal"] for r in rows)) == 1
        st.markdown(
            f"**{'✅ Reproduced' if identical else '⚠️ Not reproduced this run'}**: "
            f"all {len(rows)} sinusoidal outputs are {'identical' if identical else 'different'}, "
            f"regardless of input."
        )

        c1, c2 = st.columns(2)
        for col, name, label in ((c1, "char_sinusoidal_mha", "char + sinusoidal"), (c2, "char_rope_mha", "char + RoPE")):
            avg = sum(entropies[name]) / len(entropies[name]) if entropies[name] else None
            with col:
                st.metric(
                    f"{label}: mean cross-attention entropy / uniform entropy",
                    f"{avg:.3f}" if avg is not None else "—",
                    help="1.00 = exactly uniform (source-blind). Well below 1.00 = real, peaked attention.",
                )

    with st.expander("Why this happens (mechanism)"):
        st.markdown(
            "- Sinusoidal PE is injected **once**, at the embedding layer, with a row-norm 5.22x "
            "the token embedding's -- position dominates identity from step one.\n"
            "- It's **never reinforced**: `EncoderBlock` passes `pe_type=\"none\"` into self-attention "
            "for sinusoidal runs, so that one-time signal has to survive 6 layers of self-attention + "
            "LayerNorm purely via the residual stream.\n"
            "- Stacked self-attention is fundamentally an averaging operator. Char's 128-token sequences "
            "(2x bpe/word's 64) give it twice the room to wash the signal out.\n"
            "- RoPE re-derives relative position fresh at every layer's Q/K dot product instead of "
            "storing it once -- it can't wash out. Measured: encoder-output cosine similarity across "
            "positions is 1.0000 (sinusoidal, fully collapsed) vs. 0.013 (RoPE, well-differentiated)."
        )


def page_architecture():
    st.title("Architecture Explorer")
    entries = load_frontend_manifest()
    ids = sorted(entries.keys())
    default_idx_a = ids.index(WINNER_ID) if WINNER_ID in ids else 0
    BASELINE_ID = "bpe_rope_mha"
    b_options = ["(none)"] + ids
    default_idx_b = b_options.index(BASELINE_ID) if BASELINE_ID in ids else 0

    c1, c2 = st.columns(2)
    with c1:
        vid_a = st.selectbox("Checkpoint A", ids, index=default_idx_a)
    with c2:
        vid_b = st.selectbox("Checkpoint B (optional, for comparison)", b_options, index=default_idx_b)

    def render(vid):
        e = entries[vid]
        arch = read_arch_config(e.get("checkpoint_path", ""))
        round_label, status = ROUND_TAGS.get(vid, ("Other / untagged", "valid"))
        st.caption(f"{STATUS_LABEL.get(status, status)} · {round_label}")
        st.json({
            "tokenizer": e.get("tokenizer_type"), "pe": e.get("pe_type"), "attn": e.get("attn_type"),
            "num_kv_heads": e.get("num_kv_heads"),
            "emb_dim": arch.get("emb_dim"), "num_layers": arch.get("num_layers"),
            "ff_dim": arch.get("ff_dim"), "dropout": arch.get("dropout"),
            "epochs": e.get("epochs"), "params": e.get("num_params"),
            "kv_cache_bytes_per_token": e.get("kv_cache_bytes_per_token_fp32"),
        }, expanded=True)
        b = (e.get("bleu") or {}).get("greedy")
        c = (e.get("chrf") or {}).get("greedy")
        t = (e.get("ter") or {}).get("greedy")
        m1, m2, m3 = st.columns(3)
        m1.metric("BLEU", f"{b:.2f}" if b is not None else "—")
        m2.metric("chrF++", f"{c:.2f}" if c is not None else "—")
        m3.metric("TER", f"{t:.2f}" if t is not None else "—")

    if vid_b == "(none)":
        render(vid_a)
    else:
        col_a, col_b = st.columns(2)
        with col_a:
            st.subheader(vid_a)
            render(vid_a)
        with col_b:
            st.subheader(vid_b)
            render(vid_b)


def _tick_latencies(pairs):
    """pairs: [(arrival_tick, finish_tick), ...]. Ticks are logical decode
    steps (see serving/batching.py's own docstring) -- comparable within
    one config, not across configs that advance a different number of
    tokens per tick (plain vs. speculative); the wall-clock numbers
    alongside these are the fair cross-config comparison."""
    lat = sorted(f - a for a, f in pairs)
    n = len(lat)
    if n == 0:
        return {"p50": None, "p95": None, "max": None, "n": 0}
    return {"p50": lat[n // 2], "p95": lat[min(n - 1, int(n * 0.95))], "max": lat[-1], "n": n}


def page_inference_lab():
    st.title("⚡ Inference Optimization")
    st.caption(
        "The from-scratch inference engine, scoped to two techniques: KV caching and "
        "post-training compression (quantization + pruning). Continuous batching, chunked "
        "prefill, prefix caching, Flash Attention, and speculative decoding are all also "
        "fully built and separately verified (see `PROGRESS.md`'s Phase 7 sections and "
        "`benchmark_optimizations.py`'s `--include_*` flags) -- just not part of this live "
        "demo or the current comparison table, a deliberate scope choice, not a limitation."
    )

    # ---- Section 1: KV caching ----
    st.header("1. KV caching: naive vs. cached decode")
    sentence = st.text_input(
        "English sentence", value="tom is the fastest runner on our team .", key="lab_sentence"
    )
    if st.button("Run KV cache comparison", type="primary") and sentence.strip():
        from serving.batching import Request
        from serving.kv_cache_naive import NaiveContiguousCache

        model, src_tok, tgt_tok, _ = load_variant(WINNER_ID)
        src_ids = src_tok.encode(sentence) + [EOS_ID]
        device = get_device()
        num_layers = len(model.decoder)
        num_kv_heads = model.num_kv_heads
        head_dim = model.decoder[0].masked_attn.head_dim
        max_len = model.max_seq_len

        def run_cached():
            backend = NaiveContiguousCache(num_layers, num_kv_heads, head_dim, max_len, device)
            backend.allocate_request(0)
            req = Request(0, src_ids, 0, max_len)
            req.admit(model, 0)
            while not req.finished:
                req.step(model, backend)
            backend.free_request(0)
            return req.generated[1:]

        with st.spinner("Warming up + timing..."):
            # Warmup: a shape/model's first CUDA call pays a one-time
            # kernel-compilation cost that would otherwise dominate the
            # very first timed measurement -- run each path once,
            # unmeasured, before the timed runs below.
            greedy_decode(model, src_ids)
            run_cached()

            (naive_ids, _), t_naive = _timed(greedy_decode, model, src_ids)
            cached_ids, t_cached = _timed(run_cached)

        exact = naive_ids == cached_ids
        st.markdown(
            ("✅ **Bit-identical output**" if exact else "⚠️ **Outputs differ this run**") +
            " between naive and KV-cached decode -- greedy decode is deterministic, so "
            "caching is a pure speed optimization here, not an approximation."
        )
        st.success(tgt_tok.decode(naive_ids))

        lat_df = pd.DataFrame([
            {"Strategy": "Naive (recompute every step)", "Latency (ms)": t_naive},
            {"Strategy": "KV cache", "Latency (ms)": t_cached},
        ])
        fig = px.bar(lat_df, x="Strategy", y="Latency (ms)", color="Strategy",
                     title="Single-sentence decode latency")
        st.plotly_chart(fig, use_container_width=True)

        st.caption(
            "A single short sentence is exactly the regime where KV caching shows the "
            "*least* benefit -- Python/kernel-launch overhead dominates the O(n²)->O(n) "
            "compute savings at this scale (the honest finding documented in PROGRESS.md's "
            "Phase 7b section). The saving grows with output length and model size, neither "
            "of which this project pushes far."
        )

    st.divider()

    # ---- Section 2: compression ----
    st.header("2. Compression: quantization and pruning")

    st.subheader("Quantization (INT8 dynamic, CPU-only)")
    st.caption(
        "Computed live below: model size before/after (fast, no decoding needed). BLEU and "
        "CPU decode latency are shown as pre-measured reference numbers, not re-run live -- "
        "PyTorch's quantized CPU kernels are dramatically slow for this project's uncached, "
        "one-full-forward-pass-per-token decode loop (measured: ~5.3 min/sentence), so "
        "re-running that on every button click isn't practical in a live demo."
    )
    if st.button("Quantize the model (live, size only)", type="primary"):
        from serving.compression import model_size_mb, quantize_dynamic_int8

        model, src_tok, tgt_tok, _ = load_variant(WINNER_ID)
        with st.spinner("Quantizing (CPU build, ~1-2 min)..."):
            orig_mb = model_size_mb(model)
            qmodel, t_quant = _timed(quantize_dynamic_int8, model)
            quant_mb = model_size_mb(qmodel)

        c1, c2, c3 = st.columns(3)
        c1.metric("fp32 size", f"{orig_mb:.2f} MB")
        c2.metric("INT8 size", f"{quant_mb:.2f} MB", delta=f"{orig_mb / quant_mb:.2f}x smaller")
        c3.metric("Quantize time", f"{t_quant / 1000:.1f} s")

    st.caption(
        "**Pre-measured reference (n=5, `results/compression_bench_log.txt`):** BLEU "
        "50.46 (fp32, GPU) -> 46.9 (INT8, CPU) -- a real quality cost, not free. CPU decode: "
        "~5.3 min/sentence for the quantized model, traced with `py-spy` to confirm it's "
        "genuinely computing (repeated quantize/dequantize per call, uncached), not hung -- "
        "dynamic quantization is CPU-only in stock PyTorch (no CUDA kernel), so none of this "
        "touches the GPU serving path measured in Section 1."
    )

    st.subheader("Pruning (magnitude-based unstructured)")
    st.caption(
        "Live below, on GPU -- unlike quantization, pruning + evaluation is fast enough here "
        "to run on a button click (a few seconds for 5 sentences)."
    )
    sparsity_pct = st.select_slider("Sparsity", options=[30, 50, 70], value=30, key="lab_sparsity")
    if st.button("Prune the model (live, GPU)", type="primary"):
        from eval.metrics import all_metrics
        from serving.compression import linear_sparsity_pct, model_size_mb, prune_unstructured

        model, src_tok, tgt_tok, _ = load_variant(WINNER_ID)
        pairs = [
            tuple(line.rstrip("\n").split("\t"))
            for line in open(os.path.join(THIS_DIR, "..", "data", "splits", "test.tsv"), encoding="utf-8")
        ]
        sample = random.Random(42).sample(pairs, 5)

        def bleu_of(m):
            hyps, refs = [], []
            for eng, fra in sample:
                out_ids, _ = greedy_decode(m, src_tok.encode(eng) + [EOS_ID])
                hyps.append(tgt_tok.decode(out_ids))
                refs.append(fra)
            return all_metrics(hyps, refs)["bleu"]

        with st.spinner(f"Pruning to {sparsity_pct}% sparsity + evaluating..."):
            base_bleu = bleu_of(model)
            pmodel, t_prune = _timed(prune_unstructured, model, sparsity_pct / 100)
            pruned_bleu = bleu_of(pmodel)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Sparsity", f"{linear_sparsity_pct(pmodel):.1f}%")
        c2.metric("Model size", f"{model_size_mb(pmodel):.2f} MB", delta="unchanged (dense storage)")
        c3.metric("Baseline BLEU", f"{base_bleu:.2f}")
        c4.metric("Pruned BLEU", f"{pruned_bleu:.2f}", delta=f"{pruned_bleu - base_bleu:+.2f}")
        st.caption(
            "Model size stays identical regardless of sparsity -- a dense tensor with zeros in "
            "it is still the same size on disk and still runs through the same GEMM kernel; "
            "unstructured pruning only pays off with a sparse tensor format this project "
            "doesn't use. Quality is the axis that actually moves: the pre-measured reference "
            "sweep (n=5) found a sharp cliff -- 0.00 BLEU loss at 30%, real degradation by 50%, "
            "near-total collapse by 70% (`results/compression_bench_log.txt`)."
        )


PAGES = {
    # "🏠 Overview": page_overview,
    "📊 Experiment Results": page_results,
    "🌐 Live Demo": page_try_it,
    "⚡ Inference Optimization": page_inference_lab,
    # "🐛 The Collapse Bug": page_collapse_bug,
    "⚙️ Architecture Explorer": page_architecture,
}
PAGES[page]()
