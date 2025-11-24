import json
import re
import logging
import torch
from neo4j import GraphDatabase
from main import BiLSTM_CRF
# from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# factory = StemmerFactory()
# stemmer = factory.create_stemmer()

# =====================================
# LOGGING SETUP
# =====================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
logger = logging.getLogger("QA-HamaPenyakit")


# =====================================
# GLOBAL CONFIG
# =====================================

QUESTION_WORDS = {
    "kenapa", "mengapa", "apa", "bagaimana",
    "kapan", "dimana", "di", "mana",
    "siapa", "gimana", "kok", "apakah",
    "gejala", "penyakit", "hama", "tanaman"
}

# Sederhana, bisa kamu perluas sendiri
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
        logger.info("NER model loaded.")

    def tokenize(self, text: str):
        # Sesuaikan dengan tokenisasi saat training
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
        current_entity = []
        current_type = None

        for token, tag in token_tags:
            if tag.startswith("B-"):
                # Close entity if exists
                if current_entity:
                    entities.setdefault(current_type, []).append(" ".join(current_entity))

                current_type = tag.split("-", 1)[1]
                current_entity = [token]

            elif tag.startswith("I-"):
                ent_type = tag.split("-", 1)[1]
                # Only continue if same type
                if current_entity and ent_type == current_type:
                    current_entity.append(token)
                else:
                    # Start new entity if mismatch
                    if current_entity:
                        entities.setdefault(current_type, []).append(" ".join(current_entity))
                    current_type = ent_type
                    current_entity = [token]

            else:  # Tag = 'O'
                if current_entity:
                    entities.setdefault(current_type, []).append(" ".join(current_entity))
                    current_entity, current_type = None, None

        # Close last entity if exists
        if current_entity:
            entities.setdefault(current_type, []).append(" ".join(current_entity))

        return entities


# =====================================
# 2️⃣ PREPROCESSING
# =====================================

def preprocess_text(text: str):
    # Kalau training pakai lowercase, ini benar.
    # Kalau tidak, hilangkan .lower()
    text = re.sub(r"[^\w\s]", "", text.lower())
    # return stemmer.stem(text.strip())
    return text.strip()


def remove_question_tokens(token_tags):
    """Hapus token kata tanya dari hasil NER (token+tag)."""
    filtered = []
    for token, tag in token_tags:
        if token.lower() in QUESTION_WORDS:
            continue
        filtered.append((token, tag))
    return filtered


def normalize_entities(entities: dict):
    """Normalisasi teks entitas pakai SYNONYM_MAP (lower-case)."""
    norm = {}
    for ent_type, values in entities.items():
        norm_vals = []
        for v in values:
            key = v.lower()
            # Kalau full phrase tidak ada di map, cek token per kata (opsional)
            if key in SYNONYM_MAP:
                norm_vals.append(SYNONYM_MAP[key])
            else:
                norm_vals.append(key)
        if norm_vals:
            norm[ent_type] = list(set(norm_vals))  # unik
    return norm


# =====================================
# 3️⃣ KNOWLEDGE GRAPH HANDLER
# =====================================

class KnowledgeGraph:
    def __init__(self, uri, user, password, database="neo4j"):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.database = database
        logger.info(f"Connected to Neo4j database: {database}")

    def close(self):
        self.driver.close()
        logger.info("Neo4j connection closed.")

    def query_full_reasoning(self, symptoms: list, organs: list):
        """
        Reasoning: cari Hama/Penyakit berdasarkan kecocokan gejala & organ,
        dengan skor berbasis rasio kecocokan (precision-like).
        """
        query = """
        MATCH (entity)
        WHERE entity:Penyakit OR entity:Hama
        
        OPTIONAL MATCH (entity)-[:MEMILIKI_GEJALA]->(g:Gejala)
        WITH entity, collect(DISTINCT g) AS all_gejala
        
        OPTIONAL MATCH (entity)-[:MENYERANG]->(o:BagianTanaman)
        WITH entity, all_gejala, collect(DISTINCT o) AS all_organ
        
        // Filter gejala & organ yang cocok
        WITH entity,
             [g IN all_gejala WHERE g.nama IN $symptoms] AS gejala_cocok,
             [o IN all_organ WHERE o.nama IN $organs] AS organ_cocok,
             all_gejala, all_organ
        
        WITH entity,
             gejala_cocok, organ_cocok,
             size(gejala_cocok) AS matched_gejala,
             size(organ_cocok) AS matched_organ,
             size(all_gejala) AS total_gejala,
             size(all_organ) AS total_organ
        
        // Hanya ambil yang ada kecocokan
        WHERE matched_gejala > 0 OR matched_organ > 0
        
        WITH entity,
             gejala_cocok,
             organ_cocok,
             CASE 
                WHEN total_gejala = 0 THEN 0.0 
                ELSE 1.0 * matched_gejala / total_gejala 
             END AS score_gejala,
             CASE 
                WHEN total_organ = 0 THEN 0.0
                ELSE 1.0 * matched_organ / total_organ
             END AS score_organ
        
        // hitung skor akhir (70% gejala, 30% organ)
        WITH entity,
             gejala_cocok,
             organ_cocok,
             (0.7 * score_gejala + 0.3 * score_organ) AS skor
        
        OPTIONAL MATCH (penyebab)-[:MENYEBABKAN]->(entity)
        OPTIONAL MATCH (entity)-[:MENYEBABKAN]->(penyakit:Penyakit)
        
        WITH entity,
             gejala_cocok,
             organ_cocok,
             skor,
             collect(DISTINCT penyebab.nama) AS penyebab_list,
             collect(DISTINCT penyakit.nama) AS penyakit_list
        
        RETURN DISTINCT 
            labels(entity)[0] AS tipe,
            entity.nama AS nama,
            [g IN gejala_cocok | g.nama] AS gejala,
            [o IN organ_cocok | o.nama] AS organ,
            penyebab_list AS penyebab,
            penyakit_list AS penyakit_disebabkan,
            skor
        ORDER BY skor DESC, nama ASC
        LIMIT 10
        """
        with self.driver.session(database=self.database) as session:
            results = session.run(query, symptoms=symptoms, organs=organs)
            res = [dict(r) for r in results]
            logger.info(
                "Reasoning query returned %d candidates for symptoms=%s, organs=%s",
                len(res), symptoms, organs
            )
            return res

    def query_entity_details(self, name: str):
        query = """
        MATCH (e)
        WHERE toLower(e.nama) = toLower($name) AND (e:Penyakit OR e:Hama)
        
        OPTIONAL MATCH (e)-[:MEMILIKI_GEJALA]->(g:Gejala)
        OPTIONAL MATCH (e)-[:MENYERANG]->(o:BagianTanaman)
        OPTIONAL MATCH (penyebab)-[:MENYEBABKAN]->(e)
        OPTIONAL MATCH (e)-[:MENYEBABKAN]->(penyakit:Penyakit)
        OPTIONAL MATCH (e)-[:NAMA_ILMIAH]->(ilmiah)
        
        RETURN labels(e)[0] AS tipe,
               e.nama AS nama,
               collect(DISTINCT g.nama) AS gejala,
               collect(DISTINCT o.nama) AS organ,
               collect(DISTINCT penyebab.nama) AS penyebab,
               collect(DISTINCT penyakit.nama) AS penyakit_disebabkan,
               collect(DISTINCT ilmiah.nama) AS nama_ilmiah
        """
        with self.driver.session(database=self.database) as session:
            result = session.run(query, name=name).single()
            if result:
                logger.info("Entity details found for '%s'", name)
            else:
                logger.info("No entity details for '%s'", name)
            return dict(result) if result else None

    def query_by_scientific_name(self, name: str):
        """Query berdasarkan nama ilmiah"""
        query = """
        MATCH (ilmiah)-[:NAMA_ILMIAH]->(e)
        WHERE toLower(ilmiah.nama) = toLower($name)
        
        OPTIONAL MATCH (e)-[:MEMILIKI_GEJALA]->(g:Gejala)
        OPTIONAL MATCH (e)-[:MENYERANG]->(o:BagianTanaman)
        OPTIONAL MATCH (penyebab)-[:MENYEBABKAN]->(e)
        OPTIONAL MATCH (e)-[:MENYEBABKAN]->(penyakit:Penyakit)
        
        RETURN labels(e)[0] AS tipe,
               e.nama AS nama,
               ilmiah.nama AS nama_ilmiah,
               collect(DISTINCT g.nama) AS gejala,
               collect(DISTINCT o.nama) AS organ,
               collect(DISTINCT penyebab.nama) AS penyebab,
               collect(DISTINCT penyakit.nama) AS penyakit_disebabkan
        """
        with self.driver.session(database=self.database) as session:
            result = session.run(query, name=name).single()
            if result:
                logger.info("Entity found by scientific name '%s'", name)
            else:
                logger.info("No entity by scientific name '%s'", name)
            return dict(result) if result else None


# =====================================
# 4️⃣ QA PIPELINE
# =====================================

class QAPipeline:
    def __init__(self, predictor, kg):
        self.predictor = predictor
        self.kg = kg

    def detect_intent(self, entities: dict):
        """Diagnosis vs definition vs unknown."""
        if "GEJALA" in entities:
            return "diagnosis"
        if "PENYAKIT" in entities or "HAMA" in entities:
            return "definition"
        return "unknown"

    def format_entity_info(self, info):
        """Format informasi entitas untuk ditampilkan"""
        output = []
        tipe = info['tipe'].lower()
        
        output.append(f"\n📘 Detail {tipe}: {info['nama']}")
        
        if info.get('nama_ilmiah') and any(info['nama_ilmiah']):
            output.append(f"  • Nama ilmiah: {', '.join(filter(None, info['nama_ilmiah']))}")
        
        if info.get('gejala') and any(info['gejala']):
            output.append(f"  • Gejala: {', '.join(filter(None, info['gejala']))}")
        
        if info.get('organ') and any(info['organ']):
            output.append(f"  • Menyerang bagian: {', '.join(filter(None, info['organ']))}")
        
        if info.get('penyebab') and any(info['penyebab']):
            output.append(f"  • Disebabkan oleh: {', '.join(filter(None, info['penyebab']))}")
        
        if info.get('penyakit_disebabkan') and any(info['penyakit_disebabkan']):
            output.append(f"  • Dapat menyebabkan: {', '.join(filter(None, info['penyakit_disebabkan']))}")
        
        return "\n".join(output)

    def fallback_answer(self, question: str, entities: dict):
        """Fallback jawaban kalau KG tidak punya info yang cukup."""
        parts = []

        if entities:
            ent_str = []
            for t, vals in entities.items():
                ent_str.append(f"{t}: {', '.join(vals)}")
            parts.append("Saya mendeteksi beberapa entitas:\n- " + "\n- ".join(ent_str))

        parts.append(
            "Namun, saya belum menemukan informasi yang cocok di basis pengetahuan.\n"
            "Untuk kepastian, sebaiknya konsultasikan dengan penyuluh pertanian atau ahli terkait, "
            "dan amati kembali gejala yang muncul pada tanaman."
        )

        print("\n".join(parts))

    def answer(self, question: str):
        logger.info("Incoming question: %s", question)
        clean_text = preprocess_text(question)
        ner_output = self.predictor.predict(clean_text)

        # Filter kata tanya
        filtered_tokens = remove_question_tokens(ner_output)

        entities = self.predictor.extract_entities(filtered_tokens)
        entities = normalize_entities(entities)

        logger.info("Entities detected: %s", entities)
        print(f"Entities ditemukan: {entities}")

        intent = self.detect_intent(entities)
        logger.info("Detected intent: %s", intent)

        # 1) User tanya tentang penyakit/hama tertentu → definisi
        if intent == "definition":
            found = False
            for ent_type in ["PENYAKIT", "HAMA"]:
                for ent in entities.get(ent_type, []):
                    info = self.kg.query_entity_details(ent)
                    if not info:
                        info = self.kg.query_by_scientific_name(ent)
                    if not info:
                        logger.info("No info for entity '%s'", ent)
                        continue
                    found = True
                    print(self.format_entity_info(info))
            if found:
                return
            else:
                self.fallback_answer(question, entities)
                return

        # 2) Diagnosis berdasarkan gejala & organ
        if intent == "diagnosis":
            gejala = entities.get("GEJALA", [])
            organ = entities.get("BAGIAN_TANAMAN", [])

            if not gejala and not organ:
                print("Saya tidak menemukan gejala atau bagian tanaman yang jelas dalam pertanyaan Anda.")
                return

            reasoning_results = self.kg.query_full_reasoning(symptoms=gejala, organs=organ)

            if not reasoning_results:
                print("Tidak ditemukan penyakit atau hama yang cocok di basis pengetahuan.")
                self.fallback_answer(question, entities)
                return

            print("\n🧬 Kemungkinan diagnosis berdasarkan gejala dan organ yang diserang:")
            for i, r in enumerate(reasoning_results, 1):
                tipe = r['tipe'].lower()
                line = f"{i}. {r['nama']} ({tipe}, skor: {r['skor']:.2f})"
                
                if r.get('penyebab') and any(r['penyebab']):
                    penyebab_str = ', '.join(filter(None, r['penyebab']))
                    line += f"\n   → Disebabkan oleh: {penyebab_str}"
                
                if r.get('penyakit_disebabkan') and any(r['penyakit_disebabkan']):
                    penyakit_str = ', '.join(filter(None, r['penyakit_disebabkan']))
                    line += f"\n   → Dapat menyebabkan penyakit: {penyakit_str}"
                
                if r.get('gejala') and any(r['gejala']):
                    gejala_str = ', '.join(filter(None, r['gejala']))
                    line += f"\n   → Gejala yang cocok: {gejala_str}"
                
                if r.get('organ') and any(r['organ']):
                    organ_str = ', '.join(filter(None, r['organ']))
                    line += f"\n   → Menyerang: {organ_str}"
                
                print(line)
            return

        # 3) Unknown intent → coba fallback
        print("Saya belum bisa memahami maksud pertanyaan dengan jelas.")
        self.fallback_answer(question, entities)


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

    print("="*60)
    print("🌾 SISTEM DIAGNOSIS HAMA DAN PENYAKIT PADI")
    print("="*60)
    print("\nContoh pertanyaan:")
    print("1. Daun padi menguning dan menggulung")
    print("2. Apa itu wereng coklat?")
    print("3. Bagaimana gejala penyakit blas?")
    print("4. Batang padi berlubang dan anakan mati")
    print("5. Pyricularia oryzae")
    print("\nKetik 'exit' untuk keluar\n")

    try:
        while True:
            q = input("Tanya: ")
            if q.lower().strip() == "exit":
                print("\nTerima kasih telah menggunakan sistem diagnosis!")
                break
            if q.strip():
                qa.answer(q)
            print()
    finally:
        kg.close()
