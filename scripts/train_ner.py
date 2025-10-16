import torch
import json
import torch.nn as nn
from torchcrf import CRF
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tqdm import tqdm

# =============================
# 1️⃣ LOAD DAN PARSE DATA
# =============================

def load_data(filepath):
    sentences = []
    tags = []
    with open(filepath, "r", encoding="utf-8") as f:
        sentence, tag_seq = [], []
        for line in f:
            line = line.strip()
            if not line:  # kalau baris kosong → selesai satu kalimat
                if sentence:
                    sentences.append(sentence)
                    tags.append(tag_seq)
                    sentence, tag_seq = [], []
            else:
                token, label = line.split()
                sentence.append(token)
                tag_seq.append(label)
        # tambahkan kalimat terakhir
        if sentence:
            sentences.append(sentence)
            tags.append(tag_seq)
    return sentences, tags

# =============================
# 2️⃣ ENCODER
# =============================

def build_vocab(sequences):
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for seq in sequences:
        for tok in seq:
            tok_lower = tok.lower()
            if tok_lower not in vocab:
                vocab[tok_lower] = len(vocab)
    return vocab

def build_tagset(tags):
    tag2idx = {}
    for seq in tags:
        for tag in seq:
            if tag not in tag2idx:
                tag2idx[tag] = len(tag2idx)
    return tag2idx

# =============================
# 3️⃣ DATASET CLASS
# =============================

class NERDataset(Dataset):
    def __init__(self, sentences, tags, word2idx, tag2idx, max_len=100):
        self.sentences = sentences
        self.tags = tags
        self.word2idx = word2idx
        self.tag2idx = tag2idx
        self.max_len = max_len
        self.pad_tag = "O"

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        tag_seq = self.tags[idx]

        X = [self.word2idx.get(w.lower(), self.word2idx["<UNK>"]) for w in sentence]
        y = [self.tag2idx[t] for t in tag_seq]

        if len(X) < self.max_len:
            pad_len = self.max_len - len(X)
            X += [self.word2idx["<PAD>"]] * pad_len
            y += [self.tag2idx[self.pad_tag]] * pad_len
        else:
            X = X[:self.max_len]
            y = y[:self.max_len]

        return torch.tensor(X), torch.tensor(y)

# =============================
# 4️⃣ MODEL BiLSTM + CRF
# =============================

class BiLSTM_CRF(nn.Module):
    def __init__(self, vocab_size, tagset_size, embedding_dim=100, hidden_dim=128):
        super(BiLSTM_CRF, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2,
                            num_layers=1, bidirectional=True, batch_first=True)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
        self.crf = CRF(tagset_size, batch_first=True)

    def forward(self, sentences, tags=None):
        mask = sentences != 0 
        embeds = self.embedding(sentences)
        lstm_out, _ = self.lstm(embeds)
        emissions = self.hidden2tag(lstm_out)

        if tags is not None:
            loss = -self.crf(emissions, tags, mask=mask, reduction="mean")
            return loss
        else:
            pred = self.crf.decode(emissions, mask=mask)
            return pred

# =============================
# 5️⃣ TRAINING PIPELINE
# =============================

def train_model(model, train_loader, optimizer, epochs=5):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for X, y in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            optimizer.zero_grad()
            loss = model(X, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1} Loss: {total_loss / len(train_loader):.4f}")

# =============================
# 6️⃣ EVALUATION
# =============================

def evaluate_model(model, val_loader, idx2tag):
    model.eval()
    true_tags, pred_tags = [], []

    with torch.no_grad():
        for X, y in val_loader:
            preds = model(X)
            for pred_seq, true_seq in zip(preds, y):
                true_seq = true_seq.tolist()
                for p, t in zip(pred_seq, true_seq):
                    if t == -100:
                        continue
                    pred_tags.append(idx2tag[p])
                    true_tags.append(idx2tag[t])

    print(classification_report(true_tags, pred_tags, digits=4))

# =============================
# 7️⃣ MAIN
# =============================

if __name__ == "__main__":
    # load data
    sentences, tags = load_data("data/processed/ner_data.conll")

    # build vocab
    word2idx = build_vocab(sentences)
    tag2idx = build_tagset(tags)
    if "O" not in tag2idx:
        tag2idx["O"] = len(tag2idx)
    idx2tag = {v: k for k, v in tag2idx.items()}

    # split train/val
    train_sents, val_sents, train_tags, val_tags = train_test_split(
        sentences, tags, test_size=0.3, random_state=42
    )

    # dataset dan dataloader
    train_data = NERDataset(train_sents, train_tags, word2idx, tag2idx)
    val_data = NERDataset(val_sents, val_tags, word2idx, tag2idx)

    train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=16)

    # model
    model = BiLSTM_CRF(vocab_size=len(word2idx), tagset_size=len(tag2idx))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    torch.save(word2idx, "models/word2idx.pt")
    torch.save(tag2idx, "models/tag2idx.pt")
    torch.save(model.state_dict(), "models/bilstm_crf_best.pt")

    # train
    train_model(model, train_loader, optimizer, epochs=20)

    # eval
    evaluate_model(model, val_loader, idx2tag)
    

