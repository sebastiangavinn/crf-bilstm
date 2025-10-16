import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

def auto_label(text: str, gazetteer: Dict[str, List[str]]) -> List[Tuple[str, str]]:
    tokens = re.findall(r'\b\w+\b|[.,!?;]', text)
    labels = ["O"] * len(tokens)
    
    char_to_token = {}
    current_pos = 0
    for idx, token in enumerate(tokens):
        token_start = text.find(token, current_pos)
        if token_start != -1:
            for i in range(token_start, token_start + len(token)):
                char_to_token[i] = idx
            current_pos = token_start + len(token)
    
    all_terms = []
    for label, terms in gazetteer.items():
        for term in terms:
            all_terms.append((term, label))
    all_terms.sort(key=lambda x: len(x[0]), reverse=True)
    
    labeled_positions = set()
    
    for term, label in all_terms:
        escaped_term = re.escape(term)
        pattern = re.compile(rf"\b{escaped_term}\b", re.IGNORECASE)
        
        for match in pattern.finditer(text):
            start_char = match.start()
            end_char = match.end()
            
            token_indices = set()
            for char_pos in range(start_char, end_char):
                if char_pos in char_to_token:
                    token_indices.add(char_to_token[char_pos])
            
            token_indices = sorted(token_indices)
            
            if not any(idx in labeled_positions for idx in token_indices):
                for i, token_idx in enumerate(token_indices):
                    if i == 0:
                        labels[token_idx] = f"B-{label}"
                    else:
                        labels[token_idx] = f"I-{label}"
                    labeled_positions.add(token_idx)
    
    return list(zip(tokens, labels))


def save_conll_format(labeled_data: List[Tuple[str, str]], output_path: str):
    """
    Simpan dalam format CoNLL (token per line, blank line untuk sentence boundary)
    """
    with open(output_path, "w", encoding="utf-8") as f:
        for token, label in labeled_data:
            if token in ['.', '!', '?']:
                f.write(f"{token}\t{label}\n\n")
            else:
                f.write(f"{token}\t{label}\n")


def save_spacy_format(labeled_data: List[Tuple[str, str]], text: str, output_path: str):
    """
    Simpan dalam format Spacy training data (JSON)
    """
    entities = []
    current_entity = None
    char_position = 0
    
    for token, label in labeled_data:
        token_start = text.find(token, char_position)
        token_end = token_start + len(token)
        
        if label.startswith("B-"):
            if current_entity:
                entities.append(current_entity)
            entity_type = label[2:]
            current_entity = [token_start, token_end, entity_type]
        elif label.startswith("I-") and current_entity:
            current_entity[1] = token_end
        else:
            if current_entity:
                entities.append(current_entity)
                current_entity = None
        
        char_position = token_end
    
    if current_entity:
        entities.append(current_entity)
    
    training_data = {
        "text": text,
        "entities": [[start, end, label] for start, end, label in entities]
    }
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(training_data, f, ensure_ascii=False, indent=2)


def generate_statistics(labeled_data: List[Tuple[str, str]]) -> Dict:
    """
    Generate statistik dari hasil labeling
    """
    stats = {
        "total_tokens": len(labeled_data),
        "labeled_tokens": 0,
        "entity_counts": {}
    }
    
    for token, label in labeled_data:
        if label != "O":
            stats["labeled_tokens"] += 1
            entity_type = label.split("-")[1]
            stats["entity_counts"][entity_type] = stats["entity_counts"].get(entity_type, 0) + 1
    
    stats["coverage_percentage"] = (stats["labeled_tokens"] / stats["total_tokens"]) * 100
    
    return stats


def main():
    text_path = Path("data/ner_labeling/dataset.txt")
    gazetteer_path = Path("data/processed/gazetteer.json")
    output_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not text_path.exists():
        print(f"[ERROR] File tidak ditemukan: {text_path}")
        print("Gunakan web scraper terlebih dahulu untuk mendapatkan text!")
        return
    
    text = text_path.read_text(encoding="utf-8")
    
    if gazetteer_path.exists():
        gazetteer = json.loads(gazetteer_path.read_text(encoding="utf-8"))
    
    print(f"\n{'='*60}")
    print("AUTO LABELING NER")
    print(f"{'='*60}")
    
    labeled = auto_label(text, gazetteer)
    
    stats = generate_statistics(labeled)
    print(f"\n📊 Statistik Labeling:")
    print(f"  Total tokens: {stats['total_tokens']}")
    print(f"  Tokens terlabel: {stats['labeled_tokens']}")
    print(f"  Coverage: {stats['coverage_percentage']:.2f}%")
    print(f"\n  Entity counts:")
    for entity_type, count in stats['entity_counts'].items():
        print(f"    {entity_type}: {count}")
    
    conll_path = output_dir / "ner_data.conll"
    save_conll_format(labeled, str(conll_path))
    print(f"\n✅ CoNLL format: {conll_path}")
    
    spacy_path = output_dir / "ner_data_spacy.json"
    save_spacy_format(labeled, text, str(spacy_path))
    print(f"✅ Spacy format: {spacy_path}")
    
    simple_path = output_dir / "ner_data.txt"
    with open(simple_path, "w", encoding="utf-8") as f:
        for token, label in labeled:
            f.write(f"{token}\t{label}\n")
    print(f"✅ Simple format: {simple_path}")
    
    stats_path = output_dir / "labeling_stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"✅ Statistics: {stats_path}")
    
    print(f"\n{'='*60}")
    print("Preview (20 token pertama):")
    print(f"{'='*60}")
    for i, (token, label) in enumerate(labeled[:20]):
        print(f"{token:20} {label}")
    
    print(f"\n✨ Selesai! Dataset NER siap digunakan.")


if __name__ == "__main__":
    main()