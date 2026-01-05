"""
Test file untuk sistem Question Answering Hama dan Penyakit Padi
Menguji berbagai skenario pertanyaan dan respons sistem
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import json
import torch
import sys
from io import StringIO

# Import modul yang akan diuji
from question import (
    NERPredictor,
    QAPipeline,
    KnowledgeGraph,
    preprocess_text,
    remove_question_tokens,
    normalize_entities,
    QUESTION_WORDS,
    SYNONYM_MAP
)


class TestTextPreprocessing(unittest.TestCase):
    """Test untuk fungsi preprocessing teks"""
    
    def test_preprocess_text(self):
        """Test preprocessing teks"""
        # Test lowercase
        result = preprocess_text("Daun Padi Menguning")
        self.assertEqual(result, "daun padi menguning")
        
        # Test remove punctuation
        result = preprocess_text("Daun padi, menguning!")
        self.assertEqual(result, "daun padi menguning")
        
        # Test strip whitespace
        result = preprocess_text("  daun padi menguning  ")
        self.assertEqual(result, "daun padi menguning")
    
    def test_remove_question_tokens(self):
        """Test menghapus kata tanya dari token"""
        token_tags = [
            ("apa", "O"),
            ("gejala", "B-GEJALA"),
            ("penyakit", "B-PENYAKIT"),
            ("blas", "I-PENYAKIT"),
            ("bagaimana", "O")
        ]
        
        filtered = remove_question_tokens(token_tags)
        self.assertEqual(len(filtered), 3)
        self.assertNotIn(("apa", "O"), filtered)
        self.assertNotIn(("bagaimana", "O"), filtered)
        self.assertIn(("gejala", "B-GEJALA"), filtered)
    
    def test_normalize_entities(self):
        """Test normalisasi entitas"""
        entities = {
            "GEJALA": ["kuning", "kekuningan", "bercak coklat"],
            "BAGIAN_TANAMAN": ["helai daun", "batang padi"]
        }
        
        normalized = normalize_entities(entities)
        
        # Check synonym mapping
        self.assertIn("menguning", normalized.get("GEJALA", []))
        self.assertIn("daun", normalized.get("BAGIAN_TANAMAN", []))


class TestNERPredictor(unittest.TestCase):
    """Test untuk NER Predictor"""
    
    def setUp(self):
        """Setup mock model dan vocabulary"""
        self.word_to_ix = {
            "<PAD>": 0, "<UNK>": 1,
            "daun": 2, "padi": 3, "menguning": 4,
            "bercak": 5, "coklat": 6, "penyakit": 7
        }
        self.tag_to_ix = {
            "<PAD>": 0, "O": 1,
            "B-GEJALA": 2, "I-GEJALA": 3,
            "B-PENYAKIT": 4, "I-PENYAKIT": 5,
            "B-BAGIAN_TANAMAN": 6, "I-BAGIAN_TANAMAN": 7
        }
    
    @patch('question.torch.load')
    @patch('question.BiLSTM_CRF')
    def test_extract_entities(self, mock_model_class, mock_load):
        """Test ekstraksi entitas dari token tags"""
        # Mock model
        mock_model = MagicMock()
        mock_model_class.return_value = mock_model
        mock_load.return_value = {}
        
        predictor = NERPredictor("dummy_path.pth", self.word_to_ix, self.tag_to_ix)
        
        # Test case 1: Single entity
        token_tags = [
            ("daun", "B-BAGIAN_TANAMAN"),
            ("menguning", "B-GEJALA")
        ]
        entities = predictor.extract_entities(token_tags)
        self.assertIn("BAGIAN_TANAMAN", entities)
        self.assertIn("GEJALA", entities)
        self.assertEqual(entities["BAGIAN_TANAMAN"], ["daun"])
        self.assertEqual(entities["GEJALA"], ["menguning"])
        
        # Test case 2: Multi-token entity
        token_tags = [
            ("penyakit", "B-PENYAKIT"),
            ("blas", "I-PENYAKIT")
        ]
        entities = predictor.extract_entities(token_tags)
        self.assertEqual(entities["PENYAKIT"], ["penyakit blas"])
        
        # Test case 3: Multiple entities of same type
        token_tags = [
            ("bercak", "B-GEJALA"),
            ("coklat", "I-GEJALA"),
            ("menguning", "B-GEJALA")
        ]
        entities = predictor.extract_entities(token_tags)
        self.assertEqual(len(entities["GEJALA"]), 2)
        self.assertIn("bercak coklat", entities["GEJALA"])
        self.assertIn("menguning", entities["GEJALA"])


class TestKnowledgeGraph(unittest.TestCase):
    """Test untuk Knowledge Graph queries"""
    
    def setUp(self):
        """Setup mock Neo4j driver"""
        self.mock_driver = MagicMock()
        self.mock_session = MagicMock()
        self.mock_driver.session.return_value.__enter__.return_value = self.mock_session
    
    @patch('question.GraphDatabase')
    def test_query_entity_details(self, mock_graph_db):
        """Test query detail entitas"""
        mock_graph_db.driver.return_value = self.mock_driver
        
        # Mock result
        mock_result = {
            'tipe': 'Penyakit',
            'nama': 'Blas',
            'gejala': ['bercak coklat', 'menguning'],
            'organ': ['daun'],
            'penyebab': [],
            'penyakit_disebabkan': [],
            'nama_ilmiah': []
        }
        self.mock_session.run.return_value.single.return_value = mock_result
        
        kg = KnowledgeGraph("bolt://localhost:7687", "neo4j", "password")
        result = kg.query_entity_details("Blas")
        
        self.assertIsNotNone(result)
        self.assertEqual(result['nama'], 'Blas')
        self.assertEqual(result['tipe'], 'Penyakit')
        kg.close()
    
    @patch('question.GraphDatabase')
    def test_query_full_reasoning(self, mock_graph_db):
        """Test query reasoning untuk diagnosis"""
        mock_graph_db.driver.return_value = self.mock_driver
        
        # Mock results
        mock_results = [
            {
                'tipe': 'Penyakit',
                'nama': 'Blas',
                'gejala': ['bercak coklat'],
                'organ': ['daun'],
                'penyebab': ['Pyricularia oryzae'],
                'penyakit_disebabkan': [],
                'skor': 0.85
            },
            {
                'tipe': 'Hama',
                'nama': 'Wereng Coklat',
                'gejala': ['menguning'],
                'organ': ['daun'],
                'penyebab': [],
                'penyakit_disebabkan': [],
                'skor': 0.70
            }
        ]
        self.mock_session.run.return_value = iter([dict(r) for r in mock_results])
        
        kg = KnowledgeGraph("bolt://localhost:7687", "neo4j", "password")
        results = kg.query_full_reasoning(
            symptoms=["bercak coklat", "menguning"],
            organs=["daun"]
        )
        
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]['nama'], 'Blas')
        self.assertGreater(results[0]['skor'], results[1]['skor'])
        kg.close()


class TestQAPipeline(unittest.TestCase):
    """Test untuk QA Pipeline"""
    
    def setUp(self):
        """Setup mock predictor dan knowledge graph"""
        self.mock_predictor = MagicMock()
        self.mock_kg = MagicMock()
        self.qa = QAPipeline(self.mock_predictor, self.mock_kg)
    
    def test_detect_intent(self):
        """Test deteksi intent dari entitas"""
        # Test diagnosis intent
        entities = {"GEJALA": ["menguning"]}
        intent = self.qa.detect_intent(entities)
        self.assertEqual(intent, "diagnosis")
        
        # Test definition intent
        entities = {"PENYAKIT": ["blas"]}
        intent = self.qa.detect_intent(entities)
        self.assertEqual(intent, "definition")
        
        # Test unknown intent
        entities = {"BAGIAN_TANAMAN": ["daun"]}
        intent = self.qa.detect_intent(entities)
        self.assertEqual(intent, "unknown")
    
    def test_format_entity_info(self):
        """Test format informasi entitas"""
        info = {
            'tipe': 'Penyakit',
            'nama': 'Blas',
            'gejala': ['bercak coklat', 'menguning'],
            'organ': ['daun'],
            'penyebab': ['Pyricularia oryzae'],
            'penyakit_disebabkan': [],
            'nama_ilmiah': ['Pyricularia oryzae']
        }
        
        formatted = self.qa.format_entity_info(info)
        self.assertIn("Blas", formatted)
        self.assertIn("bercak coklat", formatted)
        self.assertIn("Pyricularia oryzae", formatted)
    
    @patch('sys.stdout', new_callable=StringIO)
    def test_answer_definition_intent(self, mock_stdout):
        """Test menjawab pertanyaan definisi"""
        # Mock NER prediction
        self.mock_predictor.predict.return_value = [
            ("apa", "O"),
            ("penyakit", "B-PENYAKIT"),
            ("blas", "I-PENYAKIT")
        ]
        self.mock_predictor.extract_entities.return_value = {
            "PENYAKIT": ["penyakit blas"]
        }
        
        # Mock KG query
        mock_info = {
            'tipe': 'Penyakit',
            'nama': 'Blas',
            'gejala': ['bercak coklat'],
            'organ': ['daun'],
            'penyebab': [],
            'penyakit_disebabkan': [],
            'nama_ilmiah': []
        }
        self.mock_kg.query_entity_details.return_value = mock_info
        
        self.qa.answer("Apa itu penyakit blas?")
        
        output = mock_stdout.getvalue()
        self.assertIn("Blas", output)
        self.assertIn("bercak coklat", output)
    
    @patch('sys.stdout', new_callable=StringIO)
    def test_answer_diagnosis_intent(self, mock_stdout):
        """Test menjawab pertanyaan diagnosis"""
        # Mock NER prediction
        self.mock_predictor.predict.return_value = [
            ("daun", "B-BAGIAN_TANAMAN"),
            ("padi", "I-BAGIAN_TANAMAN"),
            ("menguning", "B-GEJALA"),
            ("bercak", "B-GEJALA"),
            ("coklat", "I-GEJALA")
        ]
        self.mock_predictor.extract_entities.return_value = {
            "GEJALA": ["menguning", "bercak coklat"],
            "BAGIAN_TANAMAN": ["daun padi"]
        }
        
        # Mock KG reasoning query
        mock_results = [
            {
                'tipe': 'Penyakit',
                'nama': 'Blas',
                'gejala': ['bercak coklat'],
                'organ': ['daun'],
                'penyebab': ['Pyricularia oryzae'],
                'penyakit_disebabkan': [],
                'skor': 0.85
            }
        ]
        self.mock_kg.query_full_reasoning.return_value = mock_results
        
        self.qa.answer("Daun padi menguning dan muncul bercak coklat")
        
        output = mock_stdout.getvalue()
        self.assertIn("Blas", output)
        self.assertIn("diagnosis", output.lower() or "kemungkinan" in output.lower())


class TestQuestionAnsweringIntegration(unittest.TestCase):
    """Test integrasi untuk berbagai skenario pertanyaan"""
    
    def setUp(self):
        """Setup untuk test integrasi"""
        self.test_questions = [
            {
                "question": "Apa itu penyakit blas?",
                "expected_intent": "definition",
                "expected_entities": ["PENYAKIT"]
            },
            {
                "question": "Daun padi menguning dan muncul bercak coklat",
                "expected_intent": "diagnosis",
                "expected_entities": ["GEJALA", "BAGIAN_TANAMAN"]
            },
            {
                "question": "Bagaimana gejala wereng coklat?",
                "expected_intent": "definition",
                "expected_entities": ["HAMA"]
            },
            {
                "question": "Batang padi berlubang dan anakan mati",
                "expected_intent": "diagnosis",
                "expected_entities": ["GEJALA", "BAGIAN_TANAMAN"]
            },
            {
                "question": "Pyricularia oryzae",
                "expected_intent": "definition",
                "expected_entities": []
            }
        ]
    
    def test_question_types(self):
        """Test berbagai jenis pertanyaan"""
        for test_case in self.test_questions:
            question = test_case["question"]
            clean_text = preprocess_text(question)
            
            # Verify preprocessing
            self.assertIsInstance(clean_text, str)
            self.assertEqual(clean_text, clean_text.lower())
            
            # Verify question words detection
            words = question.lower().split()
            has_question_word = any(w in QUESTION_WORDS for w in words)
            
            # Some questions should have question words
            if "apa" in question.lower() or "bagaimana" in question.lower():
                self.assertTrue(has_question_word)


class TestEdgeCases(unittest.TestCase):
    """Test untuk edge cases dan error handling"""
    
    def test_empty_question(self):
        """Test pertanyaan kosong"""
        clean_text = preprocess_text("")
        self.assertEqual(clean_text, "")
    
    def test_question_with_only_punctuation(self):
        """Test pertanyaan hanya punctuation"""
        clean_text = preprocess_text("!!!???")
        self.assertEqual(clean_text, "")
    
    def test_question_with_special_characters(self):
        """Test pertanyaan dengan karakter khusus"""
        clean_text = preprocess_text("Daun padi @#$% menguning!")
        self.assertEqual(clean_text, "daun padi menguning")
    
    def test_normalize_unknown_entities(self):
        """Test normalisasi entitas yang tidak ada di synonym map"""
        entities = {
            "GEJALA": ["gejala_baru_yang_tidak_ada"],
            "PENYAKIT": ["penyakit_unknown"]
        }
        normalized = normalize_entities(entities)
        
        # Should still return the lowercase version
        self.assertIn("gejala_baru_yang_tidak_ada", normalized.get("GEJALA", []))
    
    def test_extract_entities_no_entities(self):
        """Test ekstraksi entitas ketika tidak ada entitas"""
        token_tags = [("ini", "O"), ("adalah", "O"), ("test", "O")]
        
        # Create a minimal predictor instance to test extract_entities
        word_to_ix = {"<PAD>": 0, "<UNK>": 1}
        tag_to_ix = {"<PAD>": 0, "O": 1}
        
        # Use the extract_entities method directly
        with patch('question.torch.load'), patch('question.BiLSTM_CRF'):
            predictor = NERPredictor("dummy.pth", word_to_ix, tag_to_ix)
            entities = predictor.extract_entities(token_tags)
            self.assertEqual(entities, {})


def run_tests():
    """Fungsi untuk menjalankan semua test"""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestTextPreprocessing))
    suite.addTests(loader.loadTestsFromTestCase(TestNERPredictor))
    suite.addTests(loader.loadTestsFromTestCase(TestKnowledgeGraph))
    suite.addTests(loader.loadTestsFromTestCase(TestQAPipeline))
    suite.addTests(loader.loadTestsFromTestCase(TestQuestionAnsweringIntegration))
    suite.addTests(loader.loadTestsFromTestCase(TestEdgeCases))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.failures:
        print("\nFailures:")
        for test, traceback in result.failures:
            print(f"  - {test}")
    
    if result.errors:
        print("\nErrors:")
        for test, traceback in result.errors:
            print(f"  - {test}")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    print("="*60)
    print("TESTING QUESTION ANSWERING SYSTEM")
    print("="*60)
    print()
    
    success = run_tests()
    
    if success:
        print("\n✅ All tests passed!")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed!")
        sys.exit(1)

