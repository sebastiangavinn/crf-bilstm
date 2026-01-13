from pathlib import Path
import torch
from sklearn.metrics import classification_report, confusion_matrix

from src.data.reader import DataReader
from src.data.vocabulary import Vocabulary
from src.data.dataset import NERDataset
from src.model.bilstm_crf import BiLSTM_CRF
from src.utils.logger import setup_logger
from src.utils.mask import create_mask
from src.utils.metrics import collect_predictions

LOG_FILE = "logs/evaluation.log"
DATA_DIR = Path("data")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16

logger = setup_logger(LOG_FILE)

logger.info("=== EVALUATION STARTED ===")

# Load data
test_sent, test_tags = DataReader.read_conll(DATA_DIR / "test.txt")
logger.info(f"Test sentences: {len(test_sent)}")

# Load vocab & model
vocab = Vocabulary()
vocab.load("vocab.json")

model = BiLSTM_CRF(
    vocab_size=len(vocab.word2idx),
    tagset_size=len(vocab.tag2idx)
).to(DEVICE)

model.load_state_dict(torch.load("bilstm_crf.pth", map_location=DEVICE))
model.eval()

# Dataset
test_dataset = NERDataset(test_sent, test_tags, vocab)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE)

y_true_all, y_pred_all = [], []

with torch.no_grad():
    for words, tags in test_loader:
        words, tags = words.to(DEVICE), tags.to(DEVICE)
        mask = create_mask(words)

        preds = model.predict(words, mask)
        y_true, y_pred = collect_predictions(preds, tags, mask, vocab)

        y_true_all.extend(y_true)
        y_pred_all.extend(y_pred)

report = classification_report(y_true_all, y_pred_all)
logger.info("=== FINAL TEST RESULT ===")
logger.info("\n" + report)

labels = list(vocab.tag2idx.keys())
cm = confusion_matrix(y_true_all, y_pred_all, labels=labels)

logger.info("Labels order:")
logger.info(labels)

logger.info("=== CONFUSION MATRIX ===")
logger.info("\n" + str(cm))


logger.info("=== EVALUATION FINISHED ===")
