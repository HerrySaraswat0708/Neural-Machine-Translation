from abc import ABC, abstractmethod

PAD_ID = 0
SOS_ID = 1
EOS_ID = 2
UNK_ID = 3
SPECIAL_TOKENS = ["<pad>", "<sos>", "<eos>", "<unk>"]


class BaseTokenizer(ABC):
    """Common interface for word/BPE/char tokenizers.

    encode() returns content token ids only -- SOS/EOS/PAD are added
    uniformly by the training/decoding code, never inside a tokenizer,
    so all three implementations share exactly one special-token
    convention (ids 0-3 as defined above).
    """

    @abstractmethod
    def train(self, sentences: list) -> None: ...

    @abstractmethod
    def encode(self, sentence: str) -> list: ...

    @abstractmethod
    def decode(self, ids: list) -> str: ...

    @abstractmethod
    def save(self, dir_path: str) -> None: ...

    @classmethod
    @abstractmethod
    def load(cls, dir_path: str) -> "BaseTokenizer": ...

    @property
    @abstractmethod
    def vocab_size(self) -> int: ...
