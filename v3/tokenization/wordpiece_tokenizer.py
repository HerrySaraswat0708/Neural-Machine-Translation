import os

from tokenizers import Tokenizer, decoders, pre_tokenizers, trainers
from tokenizers.models import WordPiece

from tokenization.base import BaseTokenizer, SPECIAL_TOKENS

DEFAULT_VOCAB_SIZE = 8000
CONTINUING_PREFIX = "##"


class WordPieceTokenizer(BaseTokenizer):
    """WordPiece (BERT-style): greedy longest-match-first subword splitting,
    continuation pieces prefixed with '##'. Differs from BPE (bpe_tokenizer.py)
    in both the merge-selection objective during training (likelihood gain vs.
    frequency) and the pre-tokenizer (splits on whitespace+punctuation like
    BERT rather than raw byte-level), so the two are a genuinely distinct
    comparison point, not the same algorithm wearing a different name."""

    def __init__(self, vocab_size: int = DEFAULT_VOCAB_SIZE):
        self.vocab_size_ = vocab_size
        self._tok = Tokenizer(WordPiece(unk_token="<unk>", continuing_subword_prefix=CONTINUING_PREFIX))
        self._tok.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
        self._tok.decoder = decoders.WordPiece(prefix=CONTINUING_PREFIX)

    def train(self, sentences: list) -> None:
        trainer = trainers.WordPieceTrainer(
            vocab_size=self.vocab_size_,
            special_tokens=SPECIAL_TOKENS,
            continuing_subword_prefix=CONTINUING_PREFIX,
        )
        self._tok.train_from_iterator(sentences, trainer=trainer)

    def encode(self, sentence: str) -> list:
        return self._tok.encode(sentence.lower().strip()).ids

    def decode(self, ids: list) -> str:
        # Not self._tok.decode(): the built-in WordPiece decoder applies
        # BERT-style cleanup (no space before punctuation), but this
        # corpus's target text treats terminal punctuation as its own
        # space-delimited token (matching every other tokenizer here) --
        # using the built-in decoder silently mismatches every hypothesis
        # against the reference at evaluation time, inflating TER far more
        # than it degrades BLEU/chrF (n-gram-overlap metrics partially
        # tolerate it; edit-distance-based TER does not).
        content_ids = [i for i in ids if i not in (0, 1, 2)]
        tokens = [self._tok.id_to_token(i) for i in content_ids if self._tok.id_to_token(i) is not None]
        words = []
        for t in tokens:
            if t.startswith(CONTINUING_PREFIX):
                piece = t[len(CONTINUING_PREFIX):]
                if words:
                    words[-1] += piece
                else:
                    words.append(piece)
            else:
                words.append(t)
        return " ".join(words)

    def save(self, dir_path: str) -> None:
        os.makedirs(dir_path, exist_ok=True)
        self._tok.save(os.path.join(dir_path, "tokenizer.json"))

    @classmethod
    def load(cls, dir_path: str) -> "WordPieceTokenizer":
        obj = cls.__new__(cls)
        obj._tok = Tokenizer.from_file(os.path.join(dir_path, "tokenizer.json"))
        obj.vocab_size_ = obj._tok.get_vocab_size()
        return obj

    @property
    def vocab_size(self) -> int:
        return self._tok.get_vocab_size()
