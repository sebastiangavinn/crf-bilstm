import torch
from torch.utils.data import Dataset
from .vocabulary import Vocabulary

class NERDataset(Dataset):
    def __init__(self, sentences, tags, vocab: Vocabulary, max_len=100):
        self.vocab = vocab
        self.max_len = max_len
        self.data = [self._encode(s, t) for s, t in zip(sentences, tags)]

    def _encode(self, sentence, tags):
        word_ids = [self.vocab.word2idx.get(w, 1) for w in sentence][:self.max_len]
        tag_ids = [self.vocab.tag2idx[t] for t in tags][:self.max_len]

        pad_len = self.max_len - len(word_ids)
        word_ids += [0] * pad_len
        tag_ids += [0] * pad_len

        return torch.tensor(word_ids), torch.tensor(tag_ids)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
