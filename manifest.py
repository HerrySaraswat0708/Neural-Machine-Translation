import json
import os
import threading

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
MANIFEST_PATH = os.path.join(THIS_DIR, "results", "manifest.json")

_lock = threading.Lock()


def load() -> dict:
    if not os.path.exists(MANIFEST_PATH):
        return {}
    with open(MANIFEST_PATH, encoding="utf-8") as f:
        return json.load(f)


def upsert(variant_id: str, fields: dict) -> None:
    """Merge `fields` into the manifest entry for `variant_id`. Dict-valued
    fields (e.g. "bleu") are merged key-by-key rather than overwritten, so
    e.g. evaluate.py can add a "beam" BLEU score without erasing a
    previously-recorded "greedy" one."""
    with _lock:
        data = load()
        entry = data.setdefault(variant_id, {})
        for key, value in fields.items():
            if isinstance(value, dict) and isinstance(entry.get(key), dict):
                entry[key].update(value)
            else:
                entry[key] = value
        os.makedirs(os.path.dirname(MANIFEST_PATH), exist_ok=True)
        with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
