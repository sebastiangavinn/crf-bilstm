import torch
import torch.nn as nn
from torchcrf import CRF
from typing import List

class BiLSTM_CRF(nn.Module):
    def __init__(self, vocab_size: int, tagset_size: int, 
                 emb_dim: int = 128, hidden_dim: int = 128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            emb_dim, 
            hidden_dim // 2,
            num_layers=1,
            bidirectional=True,
            batch_first=True
        )
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
        self.crf = CRF(tagset_size, batch_first=True)
    
    def _get_features(self, words: torch.Tensor) -> torch.Tensor:
        embeds = self.embedding(words)
        lstm_out, _ = self.lstm(embeds)
        features = self.hidden2tag(lstm_out)
        return features
    
    def forward(self, words: torch.Tensor, tags: torch.Tensor, 
                mask: torch.Tensor) -> torch.Tensor:
        emissions = self._get_features(words)
        loss = -self.crf(emissions, tags, mask=mask, reduction="mean")
        return loss
    
    def predict(self, words: torch.Tensor, mask: torch.Tensor) -> List[List[int]]:
        emissions = self._get_features(words)
        return self.crf.decode(emissions, mask=mask)