import json
import os

from tokenization.base import BaseTokenizer, PAD_ID, SOS_ID, EOS_ID, UNK_ID


class WordTokenizer(BaseTokenizer):
    def __init__(self):
        self.word2index = {"<pad>": PAD_ID, "<sos>": SOS_ID, "<eos>": EOS_ID, "<unk>": UNK_ID}
        self.index2word = {v: k for k, v in self.word2index.items()}

    def train(self, sentences: list) -> None:
        for sentence in sentences:
            for word in sentence.strip().split():
                if word not in self.word2index:
                    idx = len(self.word2index)
                    self.word2index[word] = idx
                    self.index2word[idx] = word

    def encode(self, sentence: str) -> list:
        return [self.word2index.get(w, UNK_ID) for w in sentence.lower().strip().split()]

    def decode(self, ids: list) -> str:
        words = [
            self.index2word.get(i, "<unk>")
            for i in ids
            if i not in (PAD_ID, SOS_ID, EOS_ID)
        ]
        return " ".join(words)

    def save(self, dir_path: str) -> None:
        os.makedirs(dir_path, exist_ok=True)
        with open(os.path.join(dir_path, "word2index.json"), "w", encoding="utf-8") as f:
            json.dump(self.word2index, f, ensure_ascii=False)

    @classmethod
    def load(cls, dir_path: str) -> "WordTokenizer":
        tok = cls()
        with open(os.path.join(dir_path, "word2index.json"), encoding="utf-8") as f:
            tok.word2index = json.load(f)
        tok.index2word = {v: k for k, v in tok.word2index.items()}
        return tok

    @property
    def vocab_size(self) -> int:
        return len(self.word2index)
