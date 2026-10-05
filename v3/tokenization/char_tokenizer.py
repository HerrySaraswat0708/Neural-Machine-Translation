import json
import os

from tokenization.base import BaseTokenizer, PAD_ID, SOS_ID, EOS_ID, UNK_ID


class CharTokenizer(BaseTokenizer):
    def __init__(self):
        self.char2index = {"<pad>": PAD_ID, "<sos>": SOS_ID, "<eos>": EOS_ID, "<unk>": UNK_ID}
        self.index2char = {v: k for k, v in self.char2index.items()}

    def train(self, sentences: list) -> None:
        for sentence in sentences:
            for ch in sentence:
                if ch not in self.char2index:
                    idx = len(self.char2index)
                    self.char2index[ch] = idx
                    self.index2char[idx] = ch

    def encode(self, sentence: str) -> list:
        return [self.char2index.get(c, UNK_ID) for c in sentence.lower().strip()]

    def decode(self, ids: list) -> str:
        chars = [
            self.index2char.get(i, "")
            for i in ids
            if i not in (PAD_ID, SOS_ID, EOS_ID)
        ]
        return "".join(chars)

    def save(self, dir_path: str) -> None:
        os.makedirs(dir_path, exist_ok=True)
        with open(os.path.join(dir_path, "char2index.json"), "w", encoding="utf-8") as f:
            json.dump(self.char2index, f, ensure_ascii=False)

    @classmethod
    def load(cls, dir_path: str) -> "CharTokenizer":
        tok = cls()
        with open(os.path.join(dir_path, "char2index.json"), encoding="utf-8") as f:
            tok.char2index = json.load(f)
        tok.index2char = {v: k for k, v in tok.char2index.items()}
        return tok

    @property
    def vocab_size(self) -> int:
        return len(self.char2index)
