from typing import List
from src.data.vocabulary import Vocabulary

def collect_predictions(
    preds: List[List[int]],
    golds,
    mask,
    vocab: Vocabulary
):
    y_true, y_pred = [], []

    for i in range(len(preds)):
        for j in range(len(preds[i])):
            if mask[i][j]:
                y_true.append(vocab.idx2tag[golds[i][j].item()])
                y_pred.append(vocab.idx2tag[preds[i][j]])

    return y_true, y_pred
