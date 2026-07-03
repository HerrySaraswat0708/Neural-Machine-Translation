import re


def normalize_english(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return re.sub(r"\s+", " ", s).strip()


def normalize_french(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zàâçéèêëîïôûùüÿñæœ.!?'-]+", r" ", s)
    return re.sub(r"\s+", " ", s).strip()
