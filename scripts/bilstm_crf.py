import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchcrf import CRF

# ===== STEP 1: LOAD DATA =====
def read_conll(file_path):
    sentences, tags = [], []
    with open(file_path, "r", encoding="utf-8") as f:
        sentence, tag_seq = [], []
        for line in f:
            line = line.strip()
            if not line:
                if sentence:
                    sentences.append(sentence)
                    tags.append(tag_seq)
                    sentence, tag_seq = [], []
            else:
                parts = line.split()
                if len(parts) >= 2:
                    word, tag = parts[0], parts[-1]
                    sentence.append(word)
                    tag_seq.append(tag)
    return sentences, tags


train_sentences, train_tags = read_conll("data/train.txt")
valid_sentences, valid_tags = read_conll("data/valid.txt")

# ===== STEP 2: BUILD VOCAB =====
word2idx = {"<PAD>": 0, "<UNK>": 1}
tag2idx = {"<PAD>": 0}

for s in train_sentences:
    for w in s:
        if w not in word2idx:
            word2idx[w] = len(word2idx)
for ts in train_tags:
    for t in ts:
        if t not in tag2idx:
            tag2idx[t] = len(tag2idx)
idx2tag = {v: k for k, v in tag2idx.items()}


def encode(sent, tags=None, max_len=100):
    wids = [word2idx.get(w, 1) for w in sent][:max_len]
    wids += [0] * (max_len - len(wids))
    if tags:
        tids = [tag2idx[t] for t in tags][:max_len]
        tids += [0] * (max_len - len(tids))
        return torch.tensor(wids), torch.tensor(tids)
    return torch.tensor(wids)

# ===== STEP 3: MODEL BiLSTM + CRF =====
class BiLSTM_CRF(nn.Module):
    def __init__(self, vocab_size, tagset_size, emb_dim=128, hidden_dim=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            emb_dim, hidden_dim // 2, num_layers=1,
            bidirectional=True, batch_first=True
        )
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
        self.crf = CRF(tagset_size, batch_first=True)

    def forward(self, words, tags, mask):
        emissions = self._get_features(words)
        # torchcrf expects log-likelihood (negative for loss)
        loss = -self.crf(emissions, tags, mask=mask, reduction="mean")
        return loss

    def _get_features(self, words):
        embeds = self.embedding(words)
        lstm_out, _ = self.lstm(embeds)
        feats = self.hidden2tag(lstm_out)
        return feats

    def predict(self, words, mask):
        emissions = self._get_features(words)
        return self.crf.decode(emissions, mask=mask)