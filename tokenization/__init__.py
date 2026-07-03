from tokenization.word_tokenizer import WordTokenizer
from tokenization.bpe_tokenizer import BPETokenizer
from tokenization.char_tokenizer import CharTokenizer

TOKENIZER_REGISTRY = {
    "word": WordTokenizer,
    "bpe": BPETokenizer,
    "char": CharTokenizer,
}


def build_tokenizer(name: str):
    if name not in TOKENIZER_REGISTRY:
        raise ValueError(f"Unknown tokenizer '{name}', expected one of {list(TOKENIZER_REGISTRY)}")
    return TOKENIZER_REGISTRY[name]()
