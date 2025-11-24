from pathlib import Path
import random
from collections import Counter

input_file_path = Path("data/processed/ner_dataset_500.conll")

# Load dataset (split per kalimat)
with open(input_file_path, "r", encoding="utf-8") as f:
    raw_blocks = f.read().strip().split("\n\n")

# Extract entity types per sentence
def extract_entities(block):
    ents = set()
    for line in block.splitlines():
        if "\t" in line:
            _, tag = line.split("\t")
            if tag != "O":
                ents.add(tag.replace("B-","").replace("I-",""))
    return ents

data = [(block, extract_entities(block)) for block in raw_blocks]

# Shuffle (reproducible)
random.seed(42)
random.shuffle(data)

# Target split sizes
total = len(data)
train_size = int(total * 0.70)
valid_size = int(total * 0.20)

train = data[:train_size]
valid = data[train_size : train_size + valid_size]
test  = data[train_size + valid_size :]

# Pastikan semua entity types muncul di train
all_labels = set(e for _, ents in data for e in ents)
train_labels = set(e for _, ents in train for e in ents)

missing = list(all_labels - train_labels)

for label in missing:
    for i, (block, ents) in enumerate(valid + test):
        if label in ents:
            train.append((block, ents))
            if i < len(valid): valid.pop(i)
            else: test.pop(i - len(valid))
            break

# Jika train kelebihan → pindahkan kembali ke valid
while len(train) > train_size:
    valid.append(train.pop())

# Save ke file
def write_conll(path, split):
    with open(path, "w", encoding="utf-8") as f:
        for block, _ in split:
            f.write(block + "\n\n")

write_conll("train.txt", train)
write_conll("valid.txt", valid)
write_conll("test.txt",  test)

print(f"Total: {total}")
print(f"Train: {len(train)}, Valid: {len(valid)}, Test: {len(test)}")

# Statistik distribusi label
def label_stats(split):
    return Counter(lbl for _, ents in split for lbl in ents)

print("\nDistribusi Label:")
print("Train:", label_stats(train))
print("Valid:", label_stats(valid))
print("Test :", label_stats(test))
