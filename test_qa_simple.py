"""
Script sederhana untuk test question answering dengan contoh pertanyaan
Dapat dijalankan langsung untuk menguji sistem QA
"""

import json
import sys
from question import (
    NERPredictor,
    QAPipeline,
    KnowledgeGraph,
    preprocess_text
)


def test_qa_system():
    """Test sistem QA dengan beberapa contoh pertanyaan"""
    
    print("="*60)
    print("TEST SISTEM QUESTION ANSWERING")
    print("="*60)
    print()
    
    # Load vocabulary
    try:
        with open("vocab.json", "r") as f:
            vocab = json.load(f)
        word_to_ix = vocab["word2idx"]
        tag_to_ix = vocab["tag2idx"]
        print("✅ Vocabulary loaded")
    except FileNotFoundError:
        print("❌ Error: vocab.json not found")
        return False
    except Exception as e:
        print(f"❌ Error loading vocabulary: {e}")
        return False
    
    # Load model
    try:
        model_path = "bilstm_crf_model.pth"
        predictor = NERPredictor(model_path, word_to_ix, tag_to_ix)
        print("✅ NER Model loaded")
    except FileNotFoundError:
        print("❌ Error: bilstm_crf_model.pth not found")
        return False
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return False
    
    # Connect to Neo4j
    try:
        kg = KnowledgeGraph(
            "bolt://localhost:7687",
            "neo4j",
            "password",
            database="hamapenyakit"
        )
        print("✅ Neo4j connection established")
    except Exception as e:
        print(f"⚠️  Warning: Could not connect to Neo4j: {e}")
        print("   Some tests will be skipped")
        kg = None
    
    # Initialize QA pipeline
    if kg:
        qa = QAPipeline(predictor, kg)
    else:
        print("⚠️  QA Pipeline initialized without Knowledge Graph")
        qa = None
    
    print()
    print("="*60)
    print("TEST CASES")
    print("="*60)
    print()
    
    # Test questions
    test_questions = [
        {
            "question": "Apa itu penyakit blas?",
            "description": "Test definisi penyakit"
        },
        {
            "question": "Daun padi menguning dan muncul bercak coklat",
            "description": "Test diagnosis berdasarkan gejala"
        },
        {
            "question": "Bagaimana gejala wereng coklat?",
            "description": "Test definisi hama"
        },
        {
            "question": "Batang padi berlubang dan anakan mati",
            "description": "Test diagnosis dengan bagian tanaman"
        },
        {
            "question": "Pyricularia oryzae",
            "description": "Test query dengan nama ilmiah"
        },
        {
            "question": "Gejala apa yang muncul pada daun padi?",
            "description": "Test pertanyaan umum"
        }
    ]
    
    results = []
    
    for i, test_case in enumerate(test_questions, 1):
        question = test_case["question"]
        description = test_case["description"]
        
        print(f"\n[{i}/{len(test_questions)}] {description}")
        print(f"Pertanyaan: {question}")
        print("-" * 60)
        
        try:
            # Test preprocessing
            clean_text = preprocess_text(question)
            print(f"Preprocessed: {clean_text}")
            
            # Test NER
            ner_output = predictor.predict(clean_text)
            print(f"NER Output: {ner_output[:5]}...")  # Show first 5 tokens
            
            # Extract entities
            entities = predictor.extract_entities(ner_output)
            print(f"Entities: {entities}")
            
            # Test QA if available
            if qa:
                print("\nAnswer:")
                qa.answer(question)
            
            results.append({
                "question": question,
                "status": "success",
                "entities": entities
            })
            
        except Exception as e:
            print(f"❌ Error: {e}")
            results.append({
                "question": question,
                "status": "error",
                "error": str(e)
            })
        
        print()
    
    # Summary
    print("="*60)
    print("SUMMARY")
    print("="*60)
    
    success_count = sum(1 for r in results if r["status"] == "success")
    error_count = len(results) - success_count
    
    print(f"Total questions: {len(results)}")
    print(f"Success: {success_count}")
    print(f"Errors: {error_count}")
    
    # Show entities found
    print("\nEntities detected across all questions:")
    all_entities = {}
    for r in results:
        if r["status"] == "success" and r.get("entities"):
            for ent_type, ent_list in r["entities"].items():
                if ent_type not in all_entities:
                    all_entities[ent_type] = []
                all_entities[ent_type].extend(ent_list)
    
    for ent_type, ent_list in all_entities.items():
        unique_ents = list(set(ent_list))
        print(f"  {ent_type}: {unique_ents}")
    
    # Cleanup
    if kg:
        kg.close()
    
    return error_count == 0


if __name__ == "__main__":
    success = test_qa_system()
    
    if success:
        print("\n✅ All tests completed successfully!")
        sys.exit(0)
    else:
        print("\n⚠️  Some tests had errors (check Neo4j connection)")
        sys.exit(1)

