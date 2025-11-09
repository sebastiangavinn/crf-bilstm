import json
import re
import torch
from neo4j import GraphDatabase
from main import BiLSTM_CRF
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

factory = StemmerFactory()
stemmer = factory.create_stemmer()

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
    text = re.sub(r"[^\w\s]", "", text.lower())
    return stemmer.stem(text.strip())


# =====================================
# 3️⃣ KNOWLEDGE GRAPH HANDLER
# =====================================

class KnowledgeGraph:
    def __init__(self, uri, user, password, database="neo4j"):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.database = database

    def close(self):
        self.driver.close()

    def query_full_reasoning(self, symptoms: list, organs: list):
        query = """
        // Mulai dari penyakit yang cocok dengan gejala / organ
        MATCH (p:Penyakit)
        OPTIONAL MATCH (p)-[r1:MEMILIKI_GEJALA]->(g:Gejala)
        OPTIONAL MATCH (p)-[r2:MENYERANG]->(o:OrganTanaman)
        OPTIONAL MATCH (p)-[:TERJADI_PADA]->(l:Lokasi)
        OPTIONAL MATCH (virus:Penyakit)-[:MENYEBABKAN]->(p)
        OPTIONAL MATCH (h:Hama)-[:MENYEBABKAN]->(virus)
        WHERE (size($symptoms) = 0 OR g.nama IN $symptoms)
          AND (size($organs) = 0 OR o.nama IN $organs)

        WITH p, g, o, l, virus, h,
             size(collect(DISTINCT g.nama)) AS jml_gejala,
             size(collect(DISTINCT o.nama)) AS jml_organ,
             (CASE WHEN virus IS NOT NULL THEN 0.1 ELSE 0 END) +
             (CASE WHEN h IS NOT NULL THEN 0.1 ELSE 0 END) AS bonus

        RETURN DISTINCT 
            p.nama AS penyakit,
            collect(DISTINCT g.nama) AS gejala,
            collect(DISTINCT o.nama) AS organ,
            collect(DISTINCT l.nama) AS lokasi,
            virus.nama AS virus,
            h.nama AS hama,
            (0.6 * jml_gejala + 0.3 * jml_organ + bonus) AS skor
        ORDER BY skor DESC
        LIMIT 10
        """
        with self.driver.session(database=self.database) as session:
            results = session.run(query, symptoms=symptoms, organs=organs)
            return [dict(r) for r in results]

    def query_entity_details(self, name: str):
        query = """
        MATCH (e)
        WHERE toLower(e.nama) = toLower($name)
        OPTIONAL MATCH (e)-[:MEMILIKI_GEJALA]->(g:Gejala)
        OPTIONAL MATCH (e)-[:MENYERANG]->(o:OrganTanaman)
        OPTIONAL MATCH (e)-[:TERJADI_PADA]->(l:Lokasi)
        OPTIONAL MATCH (cause)-[:MENYEBABKAN]->(e)
        OPTIONAL MATCH (e)-[:MENYEBABKAN]->(d:Penyakit)
        RETURN labels(e)[0] AS tipe,
               e.nama AS nama,
               collect(DISTINCT g.nama) AS gejala,
               collect(DISTINCT o.nama) AS organ,
               collect(DISTINCT l.nama) AS lokasi,
               collect(DISTINCT cause.nama) AS penyebab,
               collect(DISTINCT d.nama) AS penyakit_disebabkan
        """
        with self.driver.session(database=self.database) as session:
            result = session.run(query, name=name).single()
            return dict(result) if result else None


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

        if "PENYAKIT" in entities or "HAMA" in entities:
            for ent_type in ["PENYAKIT", "HAMA"]:
                for ent in entities.get(ent_type, []):
                    info = self.kg.query_entity_details(ent)
                    if not info:
                        print(f"Tidak ditemukan informasi untuk {ent_type.lower()} '{ent}'.")
                        continue

                    print(f"\n📘 Detail {ent_type.lower()}: {info['nama']}")
                    if info["gejala"]:
                        print(f"  • Gejala: {', '.join(info['gejala'])}")
                    if info["organ"]:
                        print(f"  • Menyerang: {', '.join(info['organ'])}")
                    if info["lokasi"]:
                        print(f"  • Sering terjadi pada: {', '.join(info['lokasi'])}")
                    if info["penyebab"]:
                        print(f"  • Disebabkan oleh: {', '.join(info['penyebab'])}")
                    if info["penyakit_disebabkan"]:
                        print(f"  • Menyebabkan: {', '.join(info['penyakit_disebabkan'])}")
            return

        gejala = entities.get("GEJALA", [])
        organ = entities.get("ORGAN", [])

        if not gejala and not organ:
            print("Saya tidak menemukan gejala, organ, atau penyakit/hama dalam pertanyaan Anda.")
            return

        reasoning_results = self.kg.query_full_reasoning(symptoms=gejala, organs=organ)

        if not reasoning_results:
            print("Tidak ditemukan penyakit atau hama terkait.")
            return

        print("\n🧬 Kemungkinan hasil reasoning:")
        for r in reasoning_results:
            line = f"• {r['penyakit']} (skor: {r['skor']:.2f})"
            if r['virus']:
                line += f" → disebabkan oleh virus {r['virus']}"
            if r['hama']:
                line += f", dibawa oleh hama {r['hama']}"
            if r['lokasi']:
                locs = ', '.join(r['lokasi'])
                line += f", sering terjadi pada {locs}"
            print(line)


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
