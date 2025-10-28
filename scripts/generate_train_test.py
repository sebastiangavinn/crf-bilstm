from pathlib import Path
import random

input_file_path = Path("data/processed/ner_data.conll")

with open(input_file_path, "r", encoding="utf-8") as f:
    data = f.read().strip().split("\n\n")

random.shuffle(data)

train_split = 0.8
valid_split = 0.1

n = len(data)
n_train = int(n * train_split)
n_valid = int(n * valid_split)

train_data = data[:n_train]
valid_data = data[n_train:n_train+n_valid]
test_data  = data[n_train+n_valid:]

with open("train.txt", "w", encoding="utf-8") as f:
    f.write("\n\n".join(train_data))

with open("valid.txt", "w", encoding="utf-8") as f:
    f.write("\n\n".join(valid_data))

with open("test.txt", "w", encoding="utf-8") as f:
    f.write("\n\n".join(test_data))

print(f"Total: {n}, Train: {len(train_data)}, Valid: {len(valid_data)}, Test: {len(test_data)}")
