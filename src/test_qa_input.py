"""
File test untuk Question Answering dengan user input
Menggunakan modul-modul dari src/
"""

import torch
import json
import sys
from pathlib import Path

root_dir = Path(__file__).parent.parent.absolute()
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from src.data.vocabulary import Vocabulary
from src.model.bilstm_crf import BiLSTM_CRF
from src.utils.inference import (
    predict_sentence,
    extract_entities,
    QAPipeline
)
from src.utils.knowledge_graph import KnowledgeGraph


# extract_entities sudah diimport dari src.utils.inference


def format_output(token_tags, show_all=False):
    """
    Format output prediksi NER untuk ditampilkan
    
    Args:
        token_tags: List of (token, tag) tuples
        show_all: Jika True, tampilkan semua token termasuk yang tag-nya 'O'
        
    Returns:
        String yang sudah diformat
    """
    output = []
    for token, tag in token_tags:
        if not show_all and tag == "O":
            continue
        output.append(f"{token:20} -> {tag}")
    return "\n".join(output)


def load_model_and_vocab(model_path="bilstm_crf.pth", vocab_path="vocab.json", device=None,
                         kg_uri=None, kg_user="neo4j", kg_password="password", kg_database="hamapenyakit"):
    """
    Load model, vocabulary, dan knowledge graph
    
    Args:
        model_path: Path ke file model
        vocab_path: Path ke file vocabulary
        device: Device untuk inference ('cpu' atau 'cuda'), jika None akan auto-detect
        kg_uri: Neo4j URI (optional)
        kg_user: Neo4j username
        kg_password: Neo4j password
        kg_database: Neo4j database name
        
    Returns:
        Tuple (model, vocab, device, qa_pipeline, kg)
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"📦 Loading vocabulary from {vocab_path}...")
    vocab = Vocabulary()
    vocab.load(vocab_path)
    print(f"   ✅ Vocabulary loaded: {len(vocab.word2idx)} words, {len(vocab.tag2idx)} tags")
    
    print(f"📦 Loading model from {model_path}...")
    model = BiLSTM_CRF(
        vocab_size=len(vocab.word2idx),
        tagset_size=len(vocab.tag2idx),
        emb_dim=128,
        hidden_dim=128
    ).to(device)
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"   ✅ Model loaded on device: {device}")
    
    # Load Knowledge Graph jika URI diberikan
    kg = None
    if kg_uri:
        try:
            print(f"📦 Connecting to Neo4j: {kg_uri}...")
            kg = KnowledgeGraph(kg_uri, kg_user, kg_password, kg_database)
            print(f"   ✅ Knowledge Graph connected: {kg_database}")
        except Exception as e:
            print(f"   ⚠️  Warning: Could not connect to Knowledge Graph: {e}")
            kg = None
    else:
        print("   ℹ️  Knowledge Graph not configured (no URI provided)")
    
    # Initialize QA Pipeline
    qa_pipeline = QAPipeline(model, vocab, device, kg)
    print("   ✅ QA Pipeline initialized")
    
    return model, vocab, device, qa_pipeline, kg


def predict_and_display(model, vocab, device, sentence, qa_pipeline=None, use_reasoning=False):
    """
    Prediksi NER untuk sebuah kalimat dan tampilkan hasilnya
    
    Args:
        model: Model BiLSTM-CRF
        vocab: Vocabulary object
        device: Device untuk inference
        sentence: Kalimat yang akan diprediksi
        qa_pipeline: QAPipeline object (optional, untuk reasoning)
        use_reasoning: Jika True, gunakan reasoning dengan Knowledge Graph
    """
    print("\n" + "="*60)
    print(f"Input: {sentence}")
    print("="*60)
    
    if use_reasoning and qa_pipeline:
        # Gunakan QA Pipeline dengan reasoning
        result = qa_pipeline.answer_with_reasoning(sentence)
        
        # Tampilkan entities
        print("\n🏷️  Entities found:")
        if result["entities"]:
            for entity_type, entity_list in result["entities"].items():
                print(f"   {entity_type}:")
                for entity in entity_list:
                    print(f"      - {entity}")
        else:
            print("   ⚠️  No entities found")
        
        # Tampilkan intent
        print(f"\n🎯 Intent: {result['intent']}")
        
        # Tampilkan reasoning
        if result["reasoning"]:
            reasoning = result["reasoning"]
            print(f"\n🧠 Reasoning Type: {reasoning.get('type', 'N/A')}")
            
            if reasoning.get("type") == "diagnosis":
                print(f"   Symptoms: {reasoning.get('symptoms', [])}")
                print(f"   Organs: {reasoning.get('organs', [])}")
                
                if "results" in reasoning and reasoning["results"]:
                    print(f"\n   📊 Diagnosis Results ({len(reasoning['results'])} found):")
                    for i, res in enumerate(reasoning["results"][:5], 1):  # Show top 5
                        print(f"\n   {i}. {res.get('nama', 'N/A')} ({res.get('tipe', 'N/A')})")
                        print(f"      Score: {res.get('skor', 0):.2f}")
                        if res.get('gejala'):
                            print(f"      Gejala cocok: {', '.join(res['gejala'])}")
                        if res.get('organ'):
                            print(f"      Organ cocok: {', '.join(res['organ'])}")
                        if res.get('penyebab'):
                            print(f"      Penyebab: {', '.join(res['penyebab'])}")
                else:
                    print("   ⚠️  No matching diseases/pests found")
            
            elif reasoning.get("type") == "definition":
                if "results" in reasoning and reasoning["results"]:
                    print(f"\n   📖 Definition Results:")
                    for i, res in enumerate(reasoning["results"], 1):
                        print(f"\n   {i}. {res.get('nama', 'N/A')} ({res.get('tipe', 'N/A')})")
                        if res.get('nama_ilmiah'):
                            ilmiah = [n for n in res['nama_ilmiah'] if n]
                            if ilmiah:
                                print(f"      Nama ilmiah: {', '.join(ilmiah)}")
                        if res.get('gejala'):
                            gejala = [g for g in res['gejala'] if g]
                            if gejala:
                                print(f"      Gejala: {', '.join(gejala)}")
                        if res.get('organ'):
                            organ = [o for o in res['organ'] if o]
                            if organ:
                                print(f"      Menyerang: {', '.join(organ)}")
                        if res.get('penyebab'):
                            penyebab = [p for p in res['penyebab'] if p]
                            if penyebab:
                                print(f"      Penyebab: {', '.join(penyebab)}")
                else:
                    print("   ⚠️  Entity not found in knowledge graph")
            
            elif reasoning.get("type") == "unknown":
                print(f"   ⚠️  {reasoning.get('message', 'Intent tidak dikenali')}")
        
        return result
    else:
        # Mode biasa tanpa reasoning
        # Prediksi
        token_tags = predict_sentence(model, sentence, vocab, device)
        
        # Tampilkan hasil token-level
        print("\n📋 Token-level predictions:")
        print(format_output(token_tags, show_all=True))
        
        # Ekstrak entitas
        entities = extract_entities(token_tags)
        
        # Tampilkan entitas yang ditemukan
        if entities:
            print("\n🏷️  Entities found:")
            for entity_type, entity_list in entities.items():
                print(f"   {entity_type}:")
                for entity in entity_list:
                    print(f"      - {entity}")
        else:
            print("\n⚠️  No entities found")
        
        return token_tags, entities


def interactive_mode(model, vocab, device, qa_pipeline=None, use_reasoning=False):
    """
    Mode interaktif untuk user input
    
    Args:
        model: Model BiLSTM-CRF
        vocab: Vocabulary object
        device: Device untuk inference
        qa_pipeline: QAPipeline object (optional)
        use_reasoning: Jika True, gunakan reasoning
    """
    print("\n" + "="*60)
    print("🎯 INTERACTIVE MODE")
    if use_reasoning:
        print("   (with Reasoning)")
    print("="*60)
    print("\nMasukkan pertanyaan atau kalimat untuk dianalisis.")
    print("Ketik 'quit' atau 'exit' untuk keluar.")
    print("Ketik 'help' untuk melihat contoh pertanyaan.")
    if use_reasoning:
        print("Ketik 'toggle' untuk switch antara reasoning dan normal mode.\n")
    else:
        print()
    
    example_questions = [
        "Daun padi menguning dan muncul bercak coklat",
        "Apa itu penyakit blas?",
        "Batang padi berlubang dan anakan mati",
        "Bagaimana gejala wereng coklat?",
        "Tanaman padi diserang hama"
    ]
    
    current_use_reasoning = use_reasoning
    
    while True:
        try:
            user_input = input("\n>>> ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Terima kasih! Sampai jumpa!")
                break
            
            if user_input.lower() == 'help':
                print("\n📝 Contoh pertanyaan:")
                for i, example in enumerate(example_questions, 1):
                    print(f"   {i}. {example}")
                continue
            
            if user_input.lower() == 'toggle' and qa_pipeline:
                current_use_reasoning = not current_use_reasoning
                mode = "with Reasoning" if current_use_reasoning else "Normal Mode"
                print(f"🔄 Switched to {mode}")
                continue
            
            # Prediksi dan tampilkan hasil
            predict_and_display(model, vocab, device, user_input, qa_pipeline, current_use_reasoning)
            
        except KeyboardInterrupt:
            print("\n\n👋 Terima kasih! Sampai jumpa!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()


def batch_mode(model, vocab, device, sentences, qa_pipeline=None, use_reasoning=False):
    """
    Mode batch untuk test beberapa kalimat sekaligus
    
    Args:
        model: Model BiLSTM-CRF
        vocab: Vocabulary object
        device: Device untuk inference
        sentences: List of sentences
        qa_pipeline: QAPipeline object (optional)
        use_reasoning: Jika True, gunakan reasoning
    """
    print("\n" + "="*60)
    print("📦 BATCH MODE")
    if use_reasoning:
        print("   (with Reasoning)")
    print("="*60)
    
    all_entities = {}
    
    for i, sentence in enumerate(sentences, 1):
        print(f"\n[{i}/{len(sentences)}]")
        result = predict_and_display(model, vocab, device, sentence, qa_pipeline, use_reasoning)
        
        # Kumpulkan entitas
        if use_reasoning and isinstance(result, dict):
            entities = result.get("entities", {})
        else:
            entities = result[1] if isinstance(result, tuple) else {}
        
        for entity_type, entity_list in entities.items():
            if entity_type not in all_entities:
                all_entities[entity_type] = []
            all_entities[entity_type].extend(entity_list)
    
    # Summary
    if all_entities:
        print("\n" + "="*60)
        print("📊 SUMMARY - All Entities Found")
        print("="*60)
        for entity_type, entity_list in all_entities.items():
            unique_entities = list(set(entity_list))
            print(f"\n{entity_type} ({len(unique_entities)} unique):")
            for entity in unique_entities:
                print(f"   - {entity}")


def main():
    """Main function"""
    print("="*60)
    print("🌾 TEST QUESTION ANSWERING SYSTEM")
    print("="*60)
    
    # Dapatkan root directory
    root_dir = Path(__file__).parent.parent.absolute()
    
    # Cek file yang diperlukan (relatif terhadap root directory)
    model_path = root_dir / "bilstm_crf.pth"
    vocab_path = root_dir / "vocab.json"
    
    if not model_path.exists():
        print(f"❌ Error: Model file '{model_path}' not found!")
        print("   Pastikan file model sudah ada di direktori root.")
        sys.exit(1)
    
    if not vocab_path.exists():
        print(f"❌ Error: Vocabulary file '{vocab_path}' not found!")
        print("   Pastikan file vocabulary sudah ada di direktori root.")
        sys.exit(1)
    
    # Tanya apakah ingin menggunakan Knowledge Graph
    print("\n" + "="*60)
    print("KNOWLEDGE GRAPH CONFIGURATION")
    print("="*60)
    use_kg = input("Gunakan Knowledge Graph untuk reasoning? (y/n, default=n): ").strip().lower()
    
    kg_uri = None
    if use_kg == 'y':
        kg_uri = input("Neo4j URI (default: bolt://localhost:7687): ").strip() or "bolt://localhost:7687"
        kg_user = input("Neo4j Username (default: neo4j): ").strip() or "neo4j"
        kg_password = input("Neo4j Password (default: password): ").strip() or "password"
        kg_database = input("Neo4j Database (default: hamapenyakit): ").strip() or "hamapenyakit"
    else:
        kg_user = kg_password = kg_database = None
    
    # Load model, vocabulary, dan knowledge graph
    try:
        if use_kg == 'y':
            model, vocab, device, qa_pipeline, kg = load_model_and_vocab(
                str(model_path), str(vocab_path),
                kg_uri=kg_uri, kg_user=kg_user, kg_password=kg_password, kg_database=kg_database
            )
        else:
            model, vocab, device, qa_pipeline, kg = load_model_and_vocab(
                str(model_path), str(vocab_path)
            )
    except Exception as e:
        print(f"\n❌ Error loading model/vocabulary: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Menu
    print("\n" + "="*60)
    print("PILIH MODE:")
    print("="*60)
    print("1. Interactive Mode (input sendiri)")
    print("2. Batch Mode (test beberapa contoh)")
    print("3. Single Test (test satu kalimat)")
    print("="*60)
    
    choice = input("\nPilih mode (1/2/3): ").strip()
    
    # Tanya apakah ingin menggunakan reasoning
    use_reasoning = False
    if qa_pipeline and kg:
        use_reasoning_input = input("\nGunakan reasoning dengan Knowledge Graph? (y/n, default=y): ").strip().lower()
        use_reasoning = use_reasoning_input != 'n'
    
    if choice == "1":
        # Interactive mode
        interactive_mode(model, vocab, device, qa_pipeline, use_reasoning)
        
    elif choice == "2":
        # Batch mode dengan contoh
        example_sentences = [
            "Daun padi menguning dan muncul bercak coklat",
            "Apa itu penyakit blas?",
            "Batang padi berlubang dan anakan mati",
            "Bagaimana gejala wereng coklat?",
            "Tanaman padi diserang hama"
        ]
        batch_mode(model, vocab, device, example_sentences, qa_pipeline, use_reasoning)
        
    elif choice == "3":
        # Single test
        sentence = input("\nMasukkan kalimat: ").strip()
        if sentence:
            predict_and_display(model, vocab, device, sentence, qa_pipeline, use_reasoning)
        else:
            print("❌ Kalimat kosong!")
    
    else:
        print("❌ Pilihan tidak valid!")
    
    # Cleanup
    if kg:
        kg.close()


if __name__ == "__main__":
    main()

