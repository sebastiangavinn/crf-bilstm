import torch.nn as nn
from torchcrf import CRF

class BiLSTM_CRF(nn.Module):
    def __init__(
        self,
        vocab_size,
        tagset_size,
        emb_dim=128,
        hidden_dim=128,
        dropout=0,
        class_weights=None
    ):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)

        self.lstm = nn.LSTM(
            emb_dim,
            hidden_dim // 2,
            bidirectional=True,
            batch_first=True
        )

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, tagset_size)
        self.crf = CRF(tagset_size, batch_first=True)

        self.class_weights = class_weights

    def forward(self, words, tags, mask):
        emissions = self._get_emissions(words)
        return -self.crf(emissions, tags, mask=mask, reduction="mean")

    def predict(self, words, mask):
        emissions = self._get_emissions(words)
        return self.crf.decode(emissions, mask)

    def _get_emissions(self, words):
        x = self.embedding(words)
        x, _ = self.lstm(x)
        x = self.dropout(x)

        emissions = self.fc(x)

        if self.class_weights is not None:
            w = self.class_weights.view(1, 1, -1)
            emissions = emissions * w

        return emissions
