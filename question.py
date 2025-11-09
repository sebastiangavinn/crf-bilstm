import json
import re
import torch
from neo4j import GraphDatabase
from main import BiLSTM_CRF

# =====================================
# 1️⃣ MODEL & NER HELPER
# =====================================

class NERPredictor:
    def __init__(self, model_path, word_to_ix, tag_to_ix):
        self.word_to_ix = word_to_ix
        self.tag_to_ix = tag_to_ix
        self.ix_to_tag = {v: k for k, v in tag_to_ix.items()}

        self.model = BiLSTM_CRF(len(word_to_ix), len(tag_to_ix), 128, 128)
        self.model.load_state_dict(torch.load(model_path, map_location="cpu"))
        self.model.eval()

    def tokenize(self, text: str):
        return text.split()

    def text_to_tensor(self, tokens):
        idxs = [self.word_to_ix.get(w, self.word_to_ix.get("<UNK>", 0)) for w in tokens]
        return torch.tensor(idxs).unsqueeze(0)

    def predict(self, text: str):
        tokens = self.tokenize(text)
        input_tensor = self.text_to_tensor(tokens)
        mask = input_tensor != 0  
        with torch.no_grad():
            preds = self.model.predict(input_tensor, mask)[0]  
        tags = [self.ix_to_tag[i] for i in preds]
        return list(zip(tokens, tags))

    def extract_entities(self, token_tags):
        entities = {}
        current_entity = None
        current_type = None

        for token, tag in token_tags:
            if tag.startswith("B-"):
                if current_entity:
                    entities.setdefault(current_type, []).append(" ".join(current_entity))
                current_type = tag.split("-")[1]
                current_entity = [token]
            elif tag.startswith("I-") and current_entity:
                current_entity.append(token)
            else:
                if current_entity:
                    entities.setdefault(current_type, []).append(" ".join(current_entity))
                    current_entity, current_type = None, None

        if current_entity:
            entities.setdefault(current_type, []).append(" ".join(current_entity))
        return entities


# =====================================
# 2️⃣ PREPROCESSING
# =====================================

def preprocess_text(text: str):
    text = re.sub(r"[^\w\s]", "", text)  # hapus tanda baca
    text = text.lower()
    return text.strip()


# =====================================
# 3️⃣ KNOWLEDGE GRAPH HANDLER
# =====================================

class KnowledgeGraph:
    def __init__(self, uri, user, password, database="neo4j"):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.database = database

    def close(self):
        self.driver.close()

    def query_disease_by_symptom(self, symptom):
        query = """
        MATCH (p:Penyakit)-[:MEMILIKI_GEJALA]->(g:Gejala {nama: $symptom})
        RETURN DISTINCT p.nama AS penyakit
        """
        with self.driver.session(database=self.database) as session:
            results = session.run(query, symptom=symptom)
            return [r["penyakit"] for r in results]


# =====================================
# 4️⃣ QA PIPELINE
# =====================================

class QAPipeline:
    def __init__(self, predictor, kg):
        self.predictor = predictor
        self.kg = kg

    def answer(self, question: str):
        clean_text = preprocess_text(question)
        results = self.predictor.predict(clean_text)
        entities = self.predictor.extract_entities(results)

        print(f"Entities ditemukan: {entities}")

        if "GEJALA" not in entities:
            print("Saya tidak menemukan gejala pada pertanyaan Anda.")
            return

        for symptom in entities["GEJALA"]:
            diseases = self.kg.query_disease_by_symptom(symptom)
            if diseases:
                print(f"Gejala '{symptom}' dapat disebabkan oleh: {', '.join(diseases)}")
            else:
                print(f"Tidak ditemukan penyakit yang terkait dengan gejala '{symptom}'.")


# =====================================
# 5️⃣ MAIN LOOP
# =====================================

if __name__ == "__main__":
    with open("vocab.json") as f:
        vocab = json.load(f)

    word_to_ix = vocab["word2idx"]
    tag_to_ix = vocab["tag2idx"]
    model_path = "bilstm_crf_model.pth"

    predictor = NERPredictor(model_path, word_to_ix, tag_to_ix)
    kg = KnowledgeGraph("bolt://localhost:7687", "neo4j", "password", database="hamapenyakit")

    qa = QAPipeline(predictor, kg)

    while True:
        q = input("\nTanya (ketik 'exit' untuk keluar): ")
        if q.lower() == "exit":
            break
        qa.answer(q)
