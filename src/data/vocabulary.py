import json
from typing import List

class Vocabulary:
    def __init__(self):
        self.word2idx = {"<PAD>": 0, "<UNK>": 1}
        self.tag2idx = {"<PAD>": 0}
        self.idx2tag = {}

    def build(self, sentences: List[List[str]], tags: List[List[str]]):
        for sent in sentences:
            for w in sent:
                if w not in self.word2idx:
                    self.word2idx[w] = len(self.word2idx)

        for tag_seq in tags:
            for t in tag_seq:
                if t not in self.tag2idx:
                    self.tag2idx[t] = len(self.tag2idx)

        self.idx2tag = {v: k for k, v in self.tag2idx.items()}

    def save(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            json.dump({
                "word2idx": self.word2idx,
                "tag2idx": self.tag2idx,
                "idx2tag": self.idx2tag
            }, f, indent=2)

    def load(self, path: str):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        self.word2idx = data["word2idx"]
        self.tag2idx = data["tag2idx"]
        self.idx2tag = {int(k): v for k, v in data["idx2tag"].items()}
