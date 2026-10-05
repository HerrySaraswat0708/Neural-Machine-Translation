"""Synthetic source-side noise for training-time robustness (Belinkov &
Bisk 2018): cheap proxies for typos, dropped/extra words, and minor
word-order slips. Applied only to the English source text -- the French
target stays clean, so the model learns noisy-in / clean-out rather than
learning to reproduce noise.

Static augmentation, not per-epoch: a fixed fraction of training pairs get
one noised-source duplicate generated once (seeded, reproducible) and
concatenated onto the clean set, rather than re-noising on the fly every
epoch. train.py's data pipeline pre-encodes the whole dataset into cached
ID tensors once before training for speed; re-noising raw text per batch
would mean re-tokenizing every batch on CPU, which would make training
much slower for a marginal robustness gain over the static version.
"""
import random

QWERTY_NEIGHBORS = {
    "a": "qwsz", "b": "vghn", "c": "xdfv", "d": "erfcxs", "e": "wsdr",
    "f": "rtgdcv", "g": "tyhfvb", "h": "yujgbn", "i": "ujko", "j": "uikhnm",
    "k": "iolmj", "l": "opk", "m": "njk", "n": "bhjm", "o": "iklp",
    "p": "ol", "q": "wa", "r": "edft", "s": "awedxz", "t": "rfgy",
    "u": "yhji", "v": "cfgb", "w": "qase", "x": "zsdc", "y": "tghu",
    "z": "asx",
}


def _char_noise(word: str, rng: random.Random, prob: float) -> str:
    if len(word) < 3 or rng.random() >= prob:
        return word
    chars = list(word)
    op = rng.choice(["swap", "delete", "substitute", "insert"])
    i = rng.randrange(len(chars))
    if op == "swap" and len(chars) > 1:
        j = min(i + 1, len(chars) - 1)
        chars[i], chars[j] = chars[j], chars[i]
    elif op == "delete":
        del chars[i]
    elif op == "substitute" and chars[i] in QWERTY_NEIGHBORS:
        chars[i] = rng.choice(QWERTY_NEIGHBORS[chars[i]])
    elif op == "insert" and chars[i] in QWERTY_NEIGHBORS:
        chars.insert(i, rng.choice(QWERTY_NEIGHBORS[chars[i]]))
    return "".join(chars)


def noise_sentence(sentence: str, rng: random.Random,
                    char_prob: float = 0.15, drop_prob: float = 0.06,
                    swap_prob: float = 0.06) -> str:
    words = sentence.split()
    if not words:
        return sentence

    kept = [w for w in words if rng.random() > drop_prob]
    words = kept if kept else words  # never drop every word

    i = 0
    while i < len(words) - 1:
        if rng.random() < swap_prob:
            words[i], words[i + 1] = words[i + 1], words[i]
            i += 2
        else:
            i += 1

    words = [_char_noise(w, rng, char_prob) for w in words]
    return " ".join(words)


def augment_pairs(pairs: list, noise_prob: float, seed: int) -> list:
    """Returns noised-source duplicates for a `noise_prob` fraction of
    `pairs` (target unchanged) -- to be concatenated onto the original
    clean pairs by the caller, not used as a replacement for them."""
    rng = random.Random(seed)
    extra = []
    for eng, fra in pairs:
        if rng.random() < noise_prob:
            extra.append((noise_sentence(eng, rng), fra))
    return extra
