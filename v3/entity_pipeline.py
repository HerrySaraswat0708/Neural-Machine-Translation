"""NER + masking + terminology-constraint pipeline: the production-
pragmatic answer to "should this source token be translated or copied?" --
answered by explicit engineering (detect -> mask -> translate -> restore)
rather than hoping the neural model learns it from data alone.

Pipeline:
  1. Detect PERSON entities via `_detect_entities` -- three merged passes
     (see its docstring) to cover both properly-cased AND lowercase input,
     since a live user is just as likely to type "my name is tom" as
     "My name is Tom", and spaCy's small English model's PERSON detection
     is measurably, inconsistently worse on lowercase text (empirically
     verified: "tom"/"mary"/"alice" go undetected in every lowercase
     context tested, while "herry"/"paul"/"kevin" are detected reliably
     even lowercase -- casing help alone doesn't fully close the gap,
     hence the gazetteer fallback below).
  2. Mask each detected entity with a DISTINCT placeholder from a small
     pool of names that survive this project's BPE round-trip unchanged
     AND that the model has proven it reproduces reliably (verified
     empirically -- arbitrary novel strings get hallucinated like any
     other OOV token; these don't). Using a distinct identity per entity
     (rather than one generic marker for all of them) is what makes
     multi-entity alignment unambiguous without any position-tracking:
     after translation, each placeholder's own surface form in the output
     tells you exactly which original entity it stands for.
  3. Translate the masked (now placeholder-bearing) sentence with the
     existing NMT model, unchanged.
  4. Restore each placeholder: if the entity has a known localized form
     (GLOSSARY, e.g. Mary -> Marie -- the same pattern already observed
     in this project's training data), substitute that; otherwise copy
     the original entity text verbatim.

This does NOT touch the model or require retraining -- it wraps whatever
model you point it at.
"""
import re

import spacy

_NLP = None

ENTITY_LABELS = {"PERSON"}

# This corpus pre-splits contractions with spaces ("doesn t" instead of
# "doesn't") -- a convention spaCy's NER wasn't trained to expect. Combined
# with lowercase-input accuracy loss, it produces a specific, repeatable
# false-positive class: contraction fragments that happen to look like the
# name "Don" (don/doesn/didn/isn/wasn/etc.) get tagged PERSON. Denylisting
# them (plus single-character spans, another spurious-detection pattern
# observed) removes the dominant failure mode without touching the general
# PERSON-detection logic.
_CONTRACTION_FRAGMENTS = {
    "don", "doesn", "didn", "isn", "wasn", "weren", "aren", "hasn",
    "hadn", "couldn", "wouldn", "shouldn", "won", "can",
}


def _is_spurious(text: str) -> bool:
    t = text.strip().lower()
    # Multi-word spans are a second failure mode on this corpus's
    # space-split possessives: spaCy sometimes over-extends a PERSON span
    # across the following word(s) (e.g. "mary s mistake" as one entity,
    # instead of just "mary"). Restricting to single-token spans catches
    # this directly.
    return len(t) <= 1 or t in _CONTRACTION_FRAGMENTS or " " in t


# Deliberately small and hand-curated, not data-mined -- the point is to
# demonstrate the terminology-constraint layer working alongside the
# copy-through fallback, not to build a production-scale glossary. Real
# systems populate this from translation memories / prior human
# translations of the same domain's terminology.
GLOSSARY = {
    "mary": "marie",
    "john": "jean",
    "peter": "pierre",
    "paris": "paris",
    "london": "londres",
}

# Fallback gazetteer for the case-sensitivity gap: spaCy's small English
# model misses short/common first names when they're lowercase, and it
# misses them *inconsistently* -- "tom"/"mary"/"alice" (three of the five
# names this pipeline itself uses as safe placeholders!) go undetected in
# every lowercase context tested, while others ("herry", "paul", "kevin")
# are fine even lowercase. Casing tricks alone (see `_title_preserve`)
# don't fully close this; a small curated allowlist does, in the same
# explicit-engineering spirit as GLOSSARY and the contraction denylist
# above. Applied only to single alphabetic tokens spaCy's own passes
# didn't already flag, and only when spaCy's tagger doesn't call the token
# a stopword -- see `_detect_entities`.
_NAME_GAZETTEER = {
    "tom", "mary", "john", "alice", "paul", "peter", "sarah", "david",
    "anna", "mike", "lisa", "kevin", "emma", "james", "linda", "robert",
    "susan", "michael", "jennifer",
}

# Safe placeholders: common training-data names the model has proven it
# reproduces reliably (see PROGRESS.md's OOV stress test -- preserved 100%
# of the time), reused as stand-ins precisely because the model treats
# them as safe pass-through tokens. One distinct placeholder per detected
# entity (up to this pool's size) is what makes restoring several
# entities from one translated sentence unambiguous -- no separate
# position/index tracking needed, just "does placeholder X's surface form
# appear in the output". Real risk this simplification accepts: a source
# sentence that already naturally contains one of these names could
# collide with the substitution step -- acceptable for a scoped
# prototype, would need a collision check (or a model fine-tuned to
# recognize a dedicated copy-marker token) in a production version.
_SAFE_NAMES = ["Tom", "Mary", "John", "Alice", "Paul"]


def _nlp():
    global _NLP
    if _NLP is None:
        _NLP = spacy.load("en_core_web_sm")
    return _NLP


def _placeholder(i: int) -> str:
    return _SAFE_NAMES[i % len(_SAFE_NAMES)]


def _title_preserve(sentence: str) -> str:
    """Capitalizes the first alphabetic character of each whitespace-split
    token, leaving every other character (including punctuation) exactly
    where it was -- preserves character offsets exactly, so entity spans
    found in this transformed copy map directly back onto the original
    string. Deliberately not str.title() (mishandles apostrophes) or
    str.capitalize() (only the sentence-initial word)."""
    words = []
    for w in sentence.split(" "):
        for i, c in enumerate(w):
            if c.isalpha():
                w = w[:i] + c.upper() + w[i + 1:]
                break
        words.append(w)
    return " ".join(words)


def _overlaps(span, taken):
    s, e = span
    return any(s < te and ts < e for ts, te in taken)


def _detect_entities(sentence: str):
    """Three merged detection passes, each only adding spans the previous
    ones missed (first-found wins on overlap):
      1. spaCy NER on the sentence as typed -- highest trust, handles
         already-properly-cased input ("My name is Herry").
      2. spaCy NER on a title-cased copy (see `_title_preserve`) --
         recovers many, not all, lowercase-input misses ("my name is
         herry" -- verified empirically to help but not fully close the
         gap on its own).
      3. Curated gazetteer fallback (`_NAME_GAZETTEER`) on remaining
         single alphabetic tokens spaCy's tagger doesn't call a stopword
         -- closes the specific, verified gap left by 1+2 (common short
         names like "tom"/"mary"/"alice" going undetected lowercase in
         every context tested).
    Returns entities sorted left-to-right, each as {"start", "end", "text"}
    with `text` always sliced from the ORIGINAL sentence (never the
    title-cased copy), so restoring later always echoes back what the
    user actually typed."""
    nlp = _nlp()
    doc_orig = nlp(sentence)

    taken = []
    results = []

    def _add(start, end, text):
        # Reject spurious spans (e.g. spaCy over-extending "Mary" into
        # "Mary Went") BEFORE reserving them in `taken` -- otherwise a
        # rejected span still blocks a later, better-targeted pass (the
        # gazetteer fallback's precise single-token "mary") from claiming
        # the same region just because it arrived first.
        if _is_spurious(text):
            return
        span = (start, end)
        if _overlaps(span, taken):
            return
        taken.append(span)
        results.append({"start": start, "end": end, "text": text})

    for e in doc_orig.ents:
        if e.label_ in ENTITY_LABELS:
            _add(e.start_char, e.end_char, e.text)

    # Original-casing POS tags, keyed by start char -- used below to guard
    # the title-cased pass. Capitalizing a token can flip spaCy's NER
    # verdict on words that are plainly *not* names (verified: "i was
    # surprised by..." title-cased to "...Surprised..." gets tagged
    # PERSON, while the SAME token's original-casing POS tag correctly
    # reads VERB/VBN). Only trust a title-cased PERSON hit when the
    # original-casing tagger -- which still sees the token in its real
    # sentence position/context, just not capitalized -- independently
    # agrees it's noun-shaped. Not applied to the gazetteer pass below:
    # that one is already constrained by curated list membership, and
    # this same POS check would incorrectly reject genuine hits there
    # (e.g. "we can t help tom ." tags "tom" as VERB with no capitalization
    # signal at all, yet it's a real, correctly-detected name).
    pos_by_start = {tok.idx: tok.pos_ for tok in doc_orig}

    titled = _title_preserve(sentence)
    if titled != sentence:
        doc_titled = nlp(titled)
        for e in doc_titled.ents:
            if e.label_ not in ENTITY_LABELS:
                continue
            pos = pos_by_start.get(e.start_char)
            if pos is not None and pos not in ("PROPN", "NOUN", "X"):
                continue
            _add(e.start_char, e.end_char, sentence[e.start_char:e.end_char])

    for tok in doc_orig:
        if tok.is_alpha and not tok.is_stop and tok.text.lower() in _NAME_GAZETTEER:
            _add(tok.idx, tok.idx + len(tok.text), tok.text)

    results.sort(key=lambda r: r["start"])
    return results


def mask_entities(sentence: str):
    """Returns (masked_sentence, entity_records). entity_records[i] is a
    dict {"original", "placeholder", "label"} for the i-th entity in
    left-to-right order -- "label" is a human-facing [PERSON_i] tag for
    displaying the detected-entity -> placeholder correspondence (e.g. in
    a UI), not something fed to the model. Detected entities beyond the
    safe-placeholder pool's size are dropped (documented truncation, not
    a silent one) -- sentences needing more than that are rare for this
    corpus's short, simple examples."""
    ents = _detect_entities(sentence)
    if not ents:
        return sentence, []
    if len(ents) > len(_SAFE_NAMES):
        ents = ents[: len(_SAFE_NAMES)]

    entity_records = []
    masked = sentence
    # Replace back-to-front so earlier spans' character offsets don't shift.
    for i, ent in enumerate(reversed(ents)):
        idx = len(ents) - 1 - i
        placeholder = _placeholder(idx)
        entity_records.insert(0, {
            "original": ent["text"],
            "placeholder": placeholder,
            "label": f"[PERSON_{idx + 1}]",
            "start": ent["start"],
            "end": ent["end"],
        })
        masked = masked[: ent["start"]] + placeholder + masked[ent["end"]:]
    return masked, entity_records


def display_masked(sentence: str, entity_records: list) -> str:
    """Human-facing version of the masked sentence, substituting each
    entity's [PERSON_i] label instead of the internal safe-name
    placeholder (Tom/Mary/...) that's actually fed to the model. UI-only
    -- e.g. Try It Live's "masked sentence" preview -- so a user sees the
    entity boundaries directly rather than an implementation detail
    (bracket-tag tokens like [NOUN] can't be fed to the model itself: '['
    and ']' aren't in this tokenizer's vocabulary at all, so the model
    would just see <unk> fragments and hallucinate -- verified empirically,
    see PROGRESS.md)."""
    out = sentence
    for rec in sorted(entity_records, key=lambda r: r["start"], reverse=True):
        out = out[: rec["start"]] + rec["label"] + out[rec["end"]:]
    return out


def restore_entities(translated_masked: str, entity_records: list, use_glossary: bool = False) -> str:
    """use_glossary=False (default): always restore the name exactly as
    the user typed it -- identity preservation, the behavior most users
    actually want for their own name ("Mary" should stay "Mary", not
    become "Marie"). use_glossary=True: substitute a known localized form
    from GLOSSARY when one exists (e.g. Mary->Marie, matching how this
    training corpus itself translates well-known names) -- an opt-in
    terminology-constraint demo, not the default, since it's the opposite
    of identity preservation.

    Replaces every placeholder in ONE simultaneous regex pass over the
    original translated text, rather than one sequential re.sub call per
    entity. Sequential substitution has a real cascade bug once 3+
    entities are in play: the fixed 5-name placeholder pool gets reused
    across entities in a sentence, so one entity's RESTORED text can
    equal a DIFFERENT entity's still-pending PLACEHOLDER -- e.g. "John
    told herry about mary" assigns placeholders Tom/Mary/John (in that
    left-to-right order), so restoring entity 1 (Tom -> "John") plants
    the literal text "John" in the output just before entity 3's own
    restoration step goes looking for "John" (mary's placeholder) and
    clobbers both occurrences identically -- observed producing "mary a
    parlé de mary à herry ." instead of "john a parlé de mary à herry .".
    A single combined pass can't cascade like this: every placeholder is
    located in the UNMODIFIED text before any replacement is written, so
    a value written for one entity is never re-scanned as if it were
    still a placeholder for another."""
    if not entity_records:
        return translated_masked

    placeholder_to_replacement = {}
    for rec in entity_records:
        replacement = GLOSSARY.get(rec["original"].lower(), rec["original"]) if use_glossary else rec["original"]
        placeholder_to_replacement[rec["placeholder"].lower()] = replacement

    # Placeholders survive BPE + greedy decode as literal substrings
    # (verified), but decode() lowercases via the tokenizer's own
    # convention, so match case-insensitively.
    pattern = re.compile(
        "|".join(re.escape(p) for p in sorted(placeholder_to_replacement, key=len, reverse=True)),
        re.IGNORECASE,
    )
    return pattern.sub(lambda m: placeholder_to_replacement[m.group(0).lower()], translated_masked)


def translate_with_entity_pipeline(sentence: str, translate_fn, use_glossary: bool = False) -> dict:
    """translate_fn: str -> str, e.g. wrapping decoding.decode() + tokenizer
    encode/decode for one model. Returns a dict with every intermediate
    step, for inspection/debugging."""
    masked, entity_records = mask_entities(sentence)
    raw_translation = translate_fn(masked)
    final = restore_entities(raw_translation, entity_records, use_glossary) if entity_records else raw_translation
    return {
        "input": sentence,
        "entities": entity_records,
        "masked": masked,
        "raw_translation": raw_translation,
        "final": final,
    }
