from pathlib import Path
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score

from src.data.reader import DataReader
from src.data.vocabulary import Vocabulary
from src.data.dataset import NERDataset
from src.model.bilstm_crf import BiLSTM_CRF
from src.train.trainer import Trainer
from src.utils.logger import setup_logger
from src.utils.seed import set_seed

# ======================
# CONFIG
# ======================
DATA_DIR = Path("data")
LOG_FILE = "logs/training.log"
EPOCHS = 20
BATCH_SIZE = 16
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ======================
# INIT
# ======================
logger = setup_logger(LOG_FILE)
set_seed(42)

logger.info("=== EXPERIMENT STARTED ===")
logger.info(f"Device: {DEVICE}")

# ======================
# LOAD DATA
# ======================
train_sent, train_tags = DataReader.read_conll(DATA_DIR / "train.txt")
valid_sent, valid_tags = DataReader.read_conll(DATA_DIR / "valid.txt")

logger.info(f"Train sentences: {len(train_sent)}")
logger.info(f"Valid sentences: {len(valid_sent)}")

# ======================
# VOCAB
# ======================
vocab = Vocabulary()
vocab.build(train_sent, train_tags)

logger.info(f"Vocabulary size: {len(vocab.word2idx)}")
logger.info(f"Tag size: {len(vocab.tag2idx)}")

# ======================
# DATASET
# ======================
train_dataset = NERDataset(train_sent, train_tags, vocab)
valid_dataset = NERDataset(valid_sent, valid_tags, vocab)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE)

# ======================
# MODEL
# ======================
model = BiLSTM_CRF(
    vocab_size=len(vocab.word2idx),
    tagset_size=len(vocab.tag2idx)
).to(DEVICE)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
trainer = Trainer(model, optimizer, DEVICE)

# ======================
# TRAINING LOOP
# ======================
best_f1 = 0.0

for epoch in range(EPOCHS):
    train_loss = trainer.train_epoch(train_loader)

    # VALIDATION
    model.eval()
    y_true_all, y_pred_all = [], []

    from src.utils.mask import create_mask
    from src.utils.metrics import collect_predictions

    with torch.no_grad():
        for words, tags in valid_loader:
            words, tags = words.to(DEVICE), tags.to(DEVICE)
            mask = create_mask(words)

            preds = model.predict(words, mask)
            y_true, y_pred = collect_predictions(preds, tags, mask, vocab)

            y_true_all.extend(y_true)
            y_pred_all.extend(y_pred)

    val_f1 = f1_score(y_true_all, y_pred_all, average="macro")

    logger.info(
        f"Epoch {epoch+1}/{EPOCHS} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Val F1: {val_f1:.4f}"
    )

    if val_f1 > best_f1:
        best_f1 = val_f1
        torch.save(model.state_dict(), "bilstm_crf.pth")
        vocab.save("vocab.json")
        logger.info(">> Best model updated")

# ======================
# END
# ======================
logger.info("=== TRAINING FINISHED ===")
logger.info(f"Best Validation F1: {best_f1:.4f}")
