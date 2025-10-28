import json
from pathlib import Path
from typing import List, Tuple
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report
from torchcrf import CRF


class DataReader:
    """Handle reading and parsing CoNLL format files."""
    
    @staticmethod
    def read_conll(file_path: str) -> Tuple[List[List[str]], List[List[str]]]:
        """Read CoNLL format file and return sentences and tags."""
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


class Vocabulary:
    """Build and manage vocabulary mappings."""
    
    def __init__(self):
        self.word2idx = {"<PAD>": 0, "<UNK>": 1}
        self.tag2idx = {"<PAD>": 0}
        self.idx2tag = {}
    
    def build_vocab(self, sentences: List[List[str]], tags: List[List[str]]):
        """Build vocabulary from training data."""
        for sentence in sentences:
            for word in sentence:
                if word not in self.word2idx:
                    self.word2idx[word] = len(self.word2idx)
        
        for tag_seq in tags:
            for tag in tag_seq:
                if tag not in self.tag2idx:
                    self.tag2idx[tag] = len(self.tag2idx)
        
        self.idx2tag = {v: k for k, v in self.tag2idx.items()}
    
    def save(self, file_path: str):
        """Save vocabulary to JSON file."""
        vocab_dict = {
            "word2idx": self.word2idx,
            "tag2idx": self.tag2idx,
            "idx2tag": self.idx2tag
        }
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(vocab_dict, f, ensure_ascii=False, indent=2)
    
    def load(self, file_path: str):
        """Load vocabulary from JSON file."""
        with open(file_path, "r", encoding="utf-8") as f:
            vocab_dict = json.load(f)
        self.word2idx = vocab_dict["word2idx"]
        self.tag2idx = vocab_dict["tag2idx"]
        self.idx2tag = {int(k): v for k, v in vocab_dict["idx2tag"].items()}


class NERDataset(Dataset):
    """PyTorch Dataset for NER task."""
    
    def __init__(self, sentences: List[List[str]], tags: List[List[str]], 
                 vocab: Vocabulary, max_len: int = 100):
        self.vocab = vocab
        self.max_len = max_len
        self.samples = [self._encode(s, t) for s, t in zip(sentences, tags)]
    
    def _encode(self, sentence: List[str], tags: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode sentence and tags to tensor indices."""
        # Encode words
        word_ids = [self.vocab.word2idx.get(w, 1) for w in sentence][:self.max_len]
        word_ids += [0] * (self.max_len - len(word_ids))
        
        # Encode tags
        tag_ids = [self.vocab.tag2idx[t] for t in tags][:self.max_len]
        tag_ids += [0] * (self.max_len - len(tag_ids))
        
        return torch.tensor(word_ids), torch.tensor(tag_ids)
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.samples[idx]


class BiLSTM_CRF(nn.Module):
    """BiLSTM-CRF model for sequence labeling."""
    
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
        """Extract features from input words."""
        embeds = self.embedding(words)
        lstm_out, _ = self.lstm(embeds)
        features = self.hidden2tag(lstm_out)
        return features
    
    def forward(self, words: torch.Tensor, tags: torch.Tensor, 
                mask: torch.Tensor) -> torch.Tensor:
        """Forward pass returning negative log-likelihood loss."""
        emissions = self._get_features(words)
        loss = -self.crf(emissions, tags, mask=mask, reduction="mean")
        return loss
    
    def predict(self, words: torch.Tensor, mask: torch.Tensor) -> List[List[int]]:
        """Predict tag sequences using Viterbi decoding."""
        emissions = self._get_features(words)
        return self.crf.decode(emissions, mask=mask)


class Trainer:
    """Trainer class for model training and evaluation."""
    
    def __init__(self, model: nn.Module, optimizer: torch.optim.Optimizer, 
                 device: str):
        self.model = model
        self.optimizer = optimizer
        self.device = device
    
    def train_epoch(self, train_loader: DataLoader) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        
        for words, tags in train_loader:
            words, tags = words.to(self.device), tags.to(self.device)
            mask = self._create_mask(words)
            
            loss = self.model(words, tags, mask)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(train_loader)
    
    def evaluate(self, data_loader: DataLoader, vocab: Vocabulary) -> Tuple[List[str], List[str]]:
        """Evaluate model and return true and predicted labels."""
        self.model.eval()
        y_true, y_pred = [], []
        
        with torch.no_grad():
            for words, tags in data_loader:
                words, tags = words.to(self.device), tags.to(self.device)
                mask = self._create_mask(words)
                predictions = self.model.predict(words, mask)
                
                for i in range(len(predictions)):
                    for j, tag_idx in enumerate(predictions[i]):
                        y_true.append(vocab.idx2tag[tags[i][j].item()])
                        y_pred.append(vocab.idx2tag[tag_idx])
        
        return y_true, y_pred
    
    @staticmethod
    def _create_mask(batch: torch.Tensor) -> torch.Tensor:
        """Create padding mask."""
        return batch != 0


def main():
    """Main training and evaluation pipeline."""
    
    # Configuration
    DATA_DIR = Path("data")
    BATCH_SIZE = 16
    EMB_DIM = 128
    HIDDEN_DIM = 128
    LEARNING_RATE = 0.001
    EPOCHS = 30
    MAX_LEN = 100
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load data
    print("Loading data...")
    reader = DataReader()
    train_sentences, train_tags = reader.read_conll(DATA_DIR / "train.txt")
    valid_sentences, valid_tags = reader.read_conll(DATA_DIR / "valid.txt")
    test_sentences, test_tags = reader.read_conll(DATA_DIR / "test.txt")
    
    # Build vocabulary
    print("Building vocabulary...")
    vocab = Vocabulary()
    vocab.build_vocab(train_sentences, train_tags)
    print(f"Vocab size: {len(vocab.word2idx)}, Tag size: {len(vocab.tag2idx)}")
    
    # Create datasets and dataloaders
    train_dataset = NERDataset(train_sentences, train_tags, vocab, MAX_LEN)
    valid_dataset = NERDataset(valid_sentences, valid_tags, vocab, MAX_LEN)
    test_dataset = NERDataset(test_sentences, test_tags, vocab, MAX_LEN)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
    
    # Initialize model
    print("Initializing model...")
    model = BiLSTM_CRF(
        vocab_size=len(vocab.word2idx),
        tagset_size=len(vocab.tag2idx),
        emb_dim=EMB_DIM,
        hidden_dim=HIDDEN_DIM
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    trainer = Trainer(model, optimizer, device)
    
    # Training loop
    print("\nStarting training...")
    for epoch in range(EPOCHS):
        avg_loss = trainer.train_epoch(train_loader)
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.4f}")
    
    # Final evaluation on test set
    print("\n=== Final Evaluation on Test Set ===")
    y_true, y_pred = trainer.evaluate(test_loader, vocab)
    print(classification_report(y_true, y_pred))
    
    # Save model and vocabulary
    print("\nSaving model and vocabulary...")
    torch.save(model.state_dict(), "bilstm_crf_model.pth")
    vocab.save("vocab.json")
    print("✅ Model saved as bilstm_crf_model.pth")
    print("✅ Vocabulary saved as vocab.json")


if __name__ == "__main__":
    main()