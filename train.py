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
from src.utils.methodology_logger import log_methodology_examples
from src.utils.seed import set_seed
from src.utils.mask import create_mask
from src.utils.metrics import collect_predictions

# ======================
# CONFIG
# ======================
DATA_DIR = Path("data")
LOG_FILE = "logs/training.log"
EPOCHS = 30
BATCH_SIZE = 16
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

PATIENCE = 5

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

# ======================
# VOCAB
# ======================
vocab = Vocabulary()
vocab.build(train_sent, train_tags)
log_methodology_examples(logger=logger, vocab=vocab)

# ======================
# DATASET
# ======================
train_dataset = NERDataset(train_sent, train_tags, vocab)
valid_dataset = NERDataset(valid_sent, valid_tags, vocab)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE)

# ======================
# CLASS WEIGHTS
# ======================
weights = []
for tag in vocab.tag2idx:
    if tag == "O":
        weights.append(0.2)
    else:
        weights.append(1.0)

class_weights = torch.tensor(weights, device=DEVICE)

# ======================
# MODEL
# ======================
model = BiLSTM_CRF(
    vocab_size=len(vocab.word2idx),
    tagset_size=len(vocab.tag2idx),
    dropout=0.5,
    class_weights=class_weights
).to(DEVICE)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
trainer = Trainer(model, optimizer, DEVICE)

# ======================
# TRAINING LOOP
# ======================
best_f1 = 0.0
counter = 0

for epoch in range(EPOCHS):
    train_loss = trainer.train_epoch(train_loader)

    # ===== VALIDATION =====
    model.eval()
    y_true_all, y_pred_all = [], []

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

    # ===== EARLY STOPPING LOGIC =====
    if val_f1 > best_f1:
        best_f1 = val_f1
        counter = 0
        torch.save(model.state_dict(), "bilstm_crf.pth")
        vocab.save("vocab.json")
        logger.info(">> Best model updated")
    else:
        counter += 1
        logger.info(f"No improvement. EarlyStopping counter {counter}/{PATIENCE}")

    if counter >= PATIENCE:
        logger.info(">> Early stopping triggered")
        break

logger.info("=== TRAINING FINISHED ===")
logger.info(f"Best Validation F1: {best_f1:.4f}")
