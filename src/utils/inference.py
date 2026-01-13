"""
Inference dengan reasoning menggunakan Knowledge Graph
"""

import torch
import re
from typing import List, Tuple, Dict, Optional
from src.utils.mask import create_mask

NON_MEDICAL_TERMS = {
    "penyebab", "sebab", "alasan", "faktor",
    "mengapa", "kenapa"
}

GEJALA_LEXICON = {
    "menguning", "kuning", "kekuningan", "klorosis",
    "layu", "kering", "busuk",
    "bercak", "bercak coklat", "bercak cokelat",
    "berlubang", "gugur", "mati"
}

# Question words untuk filtering
QUESTION_WORDS = {
    "kenapa", "mengapa", "apa", "bagaimana",
    "kapan", "dimana", "di", "mana",
    "siapa", "gimana", "kok", "apakah",
    "gejala", "penyakit", "hama", "tanaman"
}

# Synonym map untuk normalisasi
SYNONYM_MAP = {
    # gejala kuning
    "kuning": "menguning",
    "kekuningan": "menguning",
    "warna kuning": "menguning",
    "klorosis": "menguning",
    
    # gejala bercak
    "bercak coklat": "bercak coklat",
    "bercak cokelat": "bercak coklat",
    
    # bagian tanaman
    "helai daun": "daun",
    "daun padi": "daun",
    "batang padi": "batang",
    "malai padi": "malai",
}

def recover_gejala_from_text(text: str, entities: Dict[str, List[str]]) -> Dict[str, List[str]]:
    """
    Fallback jika NER gagal mendeteksi GEJALA.
    """
    text_l = text.lower()
    found = set(entities.get("GEJALA", []))

    for g in GEJALA_LEXICON:
        if g in text_l:
            found.add(SYNONYM_MAP.get(g, g))

    if found:
        entities["GEJALA"] = list(found)

    return entities

def predict_sentence(model, sentence: str, vocab, device: str, max_len: int = 100) -> List[Tuple[str, str]]:
    """
    Prediksi NER untuk satu kalimat
    
    Args:
        model: Model BiLSTM-CRF
        sentence: Kalimat yang akan diprediksi
        vocab: Vocabulary object
        device: Device untuk inference
        max_len: Maximum length untuk padding
        
    Returns:
        List of (token, tag) tuples
    """
    model.eval()
    
    words = [vocab.word2idx.get(w.lower(), 1) for w in sentence.split()]
    words = words[:max_len] + [0] * (max_len - len(words))
    
    tensor = torch.tensor([words]).to(device)
    mask = create_mask(tensor)
    
    with torch.no_grad():
        preds = model.predict(tensor, mask)[0]
    
    tokens = sentence.split()
    tags = [vocab.idx2tag[p] for p in preds[:len(tokens)]]
    
    return list(zip(tokens, tags))


def preprocess_text(text: str) -> str:
    """
    Preprocess teks untuk inference
    
    Args:
        text: Input text
        
    Returns:
        Preprocessed text
    """
    text = re.sub(r"[^\w\s]", "", text.lower())
    return text.strip()


def remove_question_tokens(token_tags: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
    filtered = []
    for token, tag in token_tags:
        tok = token.lower()

        if tok in QUESTION_WORDS:
            continue

        if tok in NON_MEDICAL_TERMS:
            continue

        filtered.append((token, tag))

    return filtered


def normalize_entities(entities: Dict[str, List[str]]) -> Dict[str, List[str]]:
    """
    Normalisasi teks entitas menggunakan SYNONYM_MAP
    
    Args:
        entities: Dictionary dengan format {entity_type: [list of entities]}
        
    Returns:
        Normalized entities dictionary
    """
    norm = {}
    for ent_type, values in entities.items():
        norm_vals = []
        for v in values:
            key = v.lower()
            if key in SYNONYM_MAP:
                norm_vals.append(SYNONYM_MAP[key])
            else:
                norm_vals.append(key)
        if norm_vals:
            norm[ent_type] = list(set(norm_vals))  # unik
    return norm


def extract_entities(token_tags: List[Tuple[str, str]]) -> Dict[str, List[str]]:
    """
    Ekstrak entitas dari hasil prediksi NER
    
    Args:
        token_tags: List of (token, tag) tuples
        
    Returns:
        Dictionary dengan format {entity_type: [list of entities]}
    """
    entities = {}
    current_entity = []
    current_type = None
    
    for token, tag in token_tags:
        if tag.startswith("B-"):
            # Simpan entity sebelumnya jika ada
            if current_entity:
                entity_text = " ".join(current_entity)
                if current_type not in entities:
                    entities[current_type] = []
                entities[current_type].append(entity_text)
            
            # Mulai entity baru
            current_type = tag[2:]  # Hapus "B-"
            current_entity = [token]
            
        elif tag.startswith("I-") and current_entity:
            # Lanjutkan entity yang sama
            ent_type = tag[2:]  # Hapus "I-"
            if ent_type == current_type:
                current_entity.append(token)
            else:
                # Type berbeda, simpan yang lama dan mulai baru
                if current_entity:
                    entity_text = " ".join(current_entity)
                    if current_type not in entities:
                        entities[current_type] = []
                    entities[current_type].append(entity_text)
                current_type = ent_type
                current_entity = [token]
        else:
            # Tag = 'O' atau tag lain, simpan entity jika ada
            if current_entity:
                entity_text = " ".join(current_entity)
                if current_type not in entities:
                    entities[current_type] = []
                entities[current_type].append(entity_text)
                current_entity = []
                current_type = None
    
    # Simpan entity terakhir jika ada
    if current_entity:
        entity_text = " ".join(current_entity)
        if current_type not in entities:
            entities[current_type] = []
        entities[current_type].append(entity_text)
    
    return entities


def detect_intent(entities: Dict[str, List[str]]) -> str:
    if "GEJALA" in entities or "BAGIAN_TANAMAN" in entities:
        return "diagnosis"

    if "PENYAKIT" in entities or "HAMA" in entities:
        return "definition"

    return "unknown"

class QAPipeline:
    """
    Pipeline untuk Question Answering dengan reasoning
    """
    
    def __init__(self, model, vocab, device: str, knowledge_graph=None):
        """
        Initialize QA Pipeline
        
        Args:
            model: Model BiLSTM-CRF
            vocab: Vocabulary object
            device: Device untuk inference
            knowledge_graph: KnowledgeGraph object (optional)
        """
        self.model = model
        self.vocab = vocab
        self.device = device
        self.kg = knowledge_graph
    
    def predict_ner(self, text: str) -> List[Tuple[str, str]]:
        """
        Prediksi NER untuk teks
        
        Args:
            text: Input text
            
        Returns:
            List of (token, tag) tuples
        """
        clean_text = preprocess_text(text)
        return predict_sentence(self.model, clean_text, self.vocab, self.device)
    
    def extract_and_normalize_entities(self, text: str) -> Dict[str, List[str]]:
        ner_output = self.predict_ner(text)

        filtered_tokens = remove_question_tokens(ner_output)

        entities = extract_entities(filtered_tokens)

        entities = normalize_entities(entities)

        entities = recover_gejala_from_text(text, entities)

        return entities
    
    def answer_with_reasoning(self, question: str) -> Dict:
        """
        Jawab pertanyaan dengan reasoning menggunakan Knowledge Graph
        
        Args:
            question: Pertanyaan user
            
        Returns:
            Dictionary dengan hasil reasoning
        """
        # Ekstrak dan normalisasi entitas
        entities = self.extract_and_normalize_entities(question)
        
        # Deteksi intent
        intent = detect_intent(entities)
        
        result = {
            "question": question,
            "entities": entities,
            "intent": intent,
            "reasoning": None
        }
        
        # Jika tidak ada knowledge graph, return tanpa reasoning
        if self.kg is None:
            result["reasoning"] = {
                "status": "no_kg",
                "message": "Knowledge graph not available"
            }
            return result
        
        # Reasoning berdasarkan intent
        if intent == "diagnosis":
            # Diagnosis berdasarkan gejala & organ
            gejala = entities.get("GEJALA", [])
            organ = entities.get("BAGIAN_TANAMAN", [])
            
            if gejala or organ:
                reasoning_results = self.kg.query_full_reasoning(
                    symptoms=gejala,
                    organs=organ
                )
                result["reasoning"] = {
                    "type": "diagnosis",
                    "symptoms": gejala,
                    "organs": organ,
                    "results": reasoning_results
                }
            else:
                result["reasoning"] = {
                    "type": "diagnosis",
                    "status": "no_symptoms_or_organs",
                    "message": "Tidak ditemukan gejala atau bagian tanaman yang jelas"
                }
        
        elif intent == "definition":
            # Definisi penyakit/hama
            found = False
            definition_results = []
            
            for ent_type in ["PENYAKIT", "HAMA"]:
                for ent in entities.get(ent_type, []):
                    info = self.kg.query_entity_details(ent)
                    if not info:
                        info = self.kg.query_by_scientific_name(ent)
                    if info:
                        found = True
                        definition_results.append(info)
            
            if found:
                result["reasoning"] = {
                    "type": "definition",
                    "results": definition_results
                }
            else:
                result["reasoning"] = {
                    "type": "definition",
                    "status": "not_found",
                    "message": "Entitas tidak ditemukan di knowledge graph"
                }
        
        else:
            result["reasoning"] = {
                "type": "unknown",
                "message": "Intent tidak dikenali"
            }
        
        return result
