import torch
from torch.utils.data import Dataset
from preprocessing import clean_text

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"

def load_data(path):
    sentences, labels = [], []
    sent, lab = [], []

    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                if sent:
                    sentences.append(sent)
                    labels.append(lab)
                    sent, lab = [], []
            else:
                token, tag = line.split()
                sent.append(clean_text(token))
                lab.append(tag)

    return sentences, labels


def build_vocab(sentences):
    vocab = {PAD_TOKEN: 0, UNK_TOKEN: 1}
    for sent in sentences:
        for w in sent:
            if w not in vocab:
                vocab[w] = len(vocab)
    return vocab


def build_label_map(labels):
    label_map = {PAD_TOKEN: 0}
    for sent in labels:
        for l in sent:
            if l not in label_map:
                label_map[l] = len(label_map)
    return label_map


class NERDataset(Dataset):
    def __init__(self, sentences, labels, word2idx, label2idx, max_len=50):
        self.X = sentences
        self.y = labels
        self.word2idx = word2idx
        self.label2idx = label2idx
        self.max_len = max_len

    def encode(self, seq, mapping):
        return [mapping.get(x, mapping[UNK_TOKEN]) for x in seq]

    def pad(self, seq, pad_value):
        return seq[:self.max_len] + [pad_value] * (self.max_len - len(seq))

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.pad(self.encode(self.X[idx], self.word2idx), 0)
        y = self.pad(self.encode(self.y[idx], self.label2idx), 0)
        return torch.tensor(x), torch.tensor(y)
