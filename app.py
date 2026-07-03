import streamlit as st

import manifest
from decoding import DECODE_STRATEGIES
from inference import load_variant
from tokenization.base import EOS_ID
from viz import build_attention_heatmap

st.set_page_config(page_title="NMT Translator", layout="wide")


def is_dark_mode() -> bool:
    try:
        return st.context.theme.type == "dark"
    except Exception:
        return False


@st.cache_data(ttl=5)
def get_manifest():
    return manifest.load()


@st.cache_resource
def get_variant(variant_id: str):
    return load_variant(variant_id)


def decode_param_widgets(strategy: str, key_prefix: str) -> dict:
    if strategy == "beam":
        c1, c2 = st.columns(2)
        beam_size = c1.slider("Beam size", 2, 10, 5, key=f"{key_prefix}_beam_size")
        length_penalty = c2.slider("Length penalty", 0.0, 1.5, 0.6, key=f"{key_prefix}_len_pen")
        return {"beam_size": beam_size, "length_penalty": length_penalty}
    if strategy == "top_k":
        top_k = st.slider("Top-k", 1, 50, 10, key=f"{key_prefix}_top_k")
        return {"top_k": top_k}
    if strategy == "top_p":
        top_p = st.slider("Top-p", 0.1, 1.0, 0.9, key=f"{key_prefix}_top_p")
        return {"top_p": top_p}
    return {}


def run_translation(variant_id, sentence, strategy, decode_kwargs):
    from data.normalize import normalize_english

    model, src_tok, tgt_tok, entry = get_variant(variant_id)
    src_ids = src_tok.encode(normalize_english(sentence)) + [EOS_ID]
    out_ids, attn = DECODE_STRATEGIES[strategy](model, src_ids, return_attn=True, **decode_kwargs)
    translation = tgt_tok.decode(out_ids)

    src_tokens = [src_tok.decode([i]) or "·" for i in src_ids]
    tgt_tokens = [tgt_tok.decode([i]) or "·" for i in out_ids]
    return translation, attn, src_tokens, tgt_tokens, entry


st.title("Neural Machine Translation (English -> French)")
st.caption("Compare tokenization, positional encoding, and decoding strategies on a from-scratch Transformer.")

entries = get_manifest()
if not entries:
    st.error("No trained variants found in results/manifest.json yet. Run train.py first.")
    st.stop()

tokenizer_options = sorted({e["tokenizer_type"] for e in entries.values()})
pe_options = sorted({e["pe_type"] for e in entries.values()})
decode_options = list(DECODE_STRATEGIES.keys())

tab_translate, tab_compare = st.tabs(["Translate", "Compare"])

with tab_translate:
    c1, c2, c3 = st.columns(3)
    tokenizer = c1.selectbox("Tokenization", tokenizer_options, key="t_tokenizer")
    pe = c2.selectbox("Positional encoding", pe_options, key="t_pe")
    strategy = c3.selectbox("Decoding", decode_options, key="t_strategy")

    variant_id = f"{tokenizer}_{pe}"
    if variant_id not in entries:
        st.warning(f"Variant '{variant_id}' hasn't been trained yet.")
    else:
        decode_kwargs = decode_param_widgets(strategy, "t")
        sentence = st.text_area("Enter an English sentence:", key="t_sentence")

        if st.button("Translate", key="t_button"):
            if not sentence.strip():
                st.warning("Please enter a sentence.")
            else:
                translation, attn, src_tokens, tgt_tokens, entry = run_translation(
                    variant_id, sentence, strategy, decode_kwargs
                )
                st.success(f"**Translated:** {translation}")

                bleu = entry.get("bleu", {}).get(strategy)
                if bleu is not None:
                    st.caption(f"Held-out test-set BLEU for {variant_id}/{strategy}: {bleu:.2f}")

                if attn is not None:
                    st.plotly_chart(
                        build_attention_heatmap(src_tokens, tgt_tokens, attn, dark_mode=is_dark_mode()),
                        use_container_width=True,
                    )

with tab_compare:
    combo_labels = [
        f"{vid} · {strat}"
        for vid in entries
        for strat in decode_options
    ]
    selected = st.multiselect(
        "Pick 2+ configs to compare (tokenization+PE combo and/or decoding strategy)",
        combo_labels,
        key="c_selected",
    )
    sentence = st.text_area("Enter an English sentence:", key="c_sentence")

    if st.button("Compare", key="c_button"):
        if len(selected) < 2:
            st.warning("Select at least 2 configs to compare.")
        elif not sentence.strip():
            st.warning("Please enter a sentence.")
        else:
            st.caption(
                "BLEU shown below is each variant's held-out test-set corpus score, "
                "not a score for this specific sentence (a single sentence has no reference to score against)."
            )
            columns = st.columns(len(selected))
            for col, label in zip(columns, selected):
                variant_id, strategy = label.split(" · ")
                with col:
                    st.markdown(f"**{label}**")
                    translation, attn, src_tokens, tgt_tokens, entry = run_translation(
                        variant_id, sentence, strategy, {}
                    )
                    st.write(translation)

                    bleu = entry.get("bleu", {}).get(strategy)
                    if bleu is not None:
                        st.metric("Corpus BLEU", f"{bleu:.2f}")

                    if attn is not None:
                        st.plotly_chart(
                            build_attention_heatmap(src_tokens, tgt_tokens, attn, dark_mode=is_dark_mode()),
                            use_container_width=True,
                        )
