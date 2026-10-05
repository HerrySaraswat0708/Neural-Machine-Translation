from tokenization.word_tokenizer import WordTokenizer
from tokenization.bpe_tokenizer import BPETokenizer
from tokenization.wordpiece_tokenizer import WordPieceTokenizer
from tokenization.char_tokenizer import CharTokenizer

TOKENIZER_REGISTRY = {
    "word": WordTokenizer,
    "bpe": BPETokenizer,
    "wordpiece": WordPieceTokenizer,
    "char": CharTokenizer,
    # Same BPE algorithm/class as "bpe" -- the only difference is *what it's
    # trained on* (English+French jointly vs. two separate vocabs), handled
    # in tokenization/build_vocab.py. A shared vocabulary lets rare tokens
    # (names, numbers) get a consistent representation whether they appear
    # on the encoder or decoder side, instead of relying on two
    # independently-trained vocabularies coincidentally overlapping.
    "bpe_joint": BPETokenizer,
}


def build_tokenizer(name: str):
    if name not in TOKENIZER_REGISTRY:
        raise ValueError(f"Unknown tokenizer '{name}', expected one of {list(TOKENIZER_REGISTRY)}")
    return TOKENIZER_REGISTRY[name]()
