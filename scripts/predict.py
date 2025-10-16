import torch

from train_ner import BiLSTM_CRF

word2idx = torch.load("models/word2idx.pt")
tag2idx = torch.load("models/tag2idx.pt")
idx2tag = {v: k for k, v in tag2idx.items()}
idx2tag = {v: k for k, v in tag2idx.items()}

model_path = "models/bilstm_crf_best.pt"
model = BiLSTM_CRF(vocab_size=len(word2idx), tagset_size=len(tag2idx)) 
model.load_state_dict(torch.load(model_path))
model.eval()

# =============================
# 2️⃣ Fungsi prediksi
# =============================
def predict_sentence(model, sentence, word2idx, idx2tag, max_len=100):
    # Ubah kata jadi index
    X = [word2idx.get(w.lower(), word2idx["<UNK>"]) for w in sentence]
    if len(X) < max_len:
        X += [word2idx["<PAD>"]] * (max_len - len(X))
    else:
        X = X[:max_len]

    X_tensor = torch.tensor([X])
    pred_indices = model(X_tensor)[0]  # ambil prediksi sequence pertama

    # Kembalikan list tuple (token, pred_tag)
    result = []
    for token, idx in zip(sentence, pred_indices[:len(sentence)]):
        result.append((token, idx2tag[idx]))
    return result

# =============================
# 3️⃣ Input dan prediksi
# =============================
if __name__ == "__main__":
    sentence = input("Masukkan kalimat: ").strip().split()
    predictions = predict_sentence(model, sentence, word2idx, idx2tag)

    # Cetak hanya entitas (skip 'O')
    print("\nHasil prediksi entitas:")
    for token, tag in predictions:
        if tag != "O":
            print(f"{token}\t{tag}")