import os

from tokenizers import Tokenizer, decoders, pre_tokenizers, trainers
from tokenizers.models import BPE

from tokenization.base import BaseTokenizer, SPECIAL_TOKENS

DEFAULT_VOCAB_SIZE = 8000


class BPETokenizer(BaseTokenizer):
    def __init__(self, vocab_size: int = DEFAULT_VOCAB_SIZE):
        self.vocab_size_ = vocab_size
        self._tok = Tokenizer(BPE(unk_token="<unk>"))
        self._tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        self._tok.decoder = decoders.ByteLevel()

    def train(self, sentences: list) -> None:
        trainer = trainers.BpeTrainer(
            vocab_size=self.vocab_size_,
            special_tokens=SPECIAL_TOKENS,
        )
        self._tok.train_from_iterator(sentences, trainer=trainer)

    def encode(self, sentence: str) -> list:
        return self._tok.encode(sentence.lower().strip()).ids

    def decode(self, ids: list) -> str:
        content_ids = [i for i in ids if i not in (0, 1, 2)]
        return self._tok.decode(content_ids)

    def save(self, dir_path: str) -> None:
        os.makedirs(dir_path, exist_ok=True)
        self._tok.save(os.path.join(dir_path, "tokenizer.json"))

    @classmethod
    def load(cls, dir_path: str) -> "BPETokenizer":
        obj = cls.__new__(cls)
        obj._tok = Tokenizer.from_file(os.path.join(dir_path, "tokenizer.json"))
        obj.vocab_size_ = obj._tok.get_vocab_size()
        return obj

    @property
    def vocab_size(self) -> int:
        return self._tok.get_vocab_size()
