"""
Inference script for BiLSTM-CRF NER model.
Usage: python inference.py
"""

import torch
from typing import List, Tuple
from main import BiLSTM_CRF, Vocabulary


class NERPredictor:
    """Class for making predictions with trained NER model."""
    
    def __init__(self, model_path: str, vocab_path: str, device: str = "cpu"):
        """
        Initialize predictor with trained model and vocabulary.
        
        Args:
            model_path: Path to saved model weights
            vocab_path: Path to vocabulary JSON file
            device: Device to run inference on ('cpu' or 'cuda')
        """
        self.device = device
        
        # Load vocabulary
        self.vocab = Vocabulary()
        self.vocab.load(vocab_path)
        
        # Initialize and load model
        self.model = BiLSTM_CRF(
            vocab_size=len(self.vocab.word2idx),
            tagset_size=len(self.vocab.tag2idx),
            emb_dim=128,  # Must match training config
            hidden_dim=128
        ).to(self.device)
        
        self.model.load_state_dict(
            torch.load(model_path, map_location=self.device)
        )
        self.model.eval()
    
    def tokenize(self, sentence: str) -> List[str]:
        """Simple whitespace tokenization."""
        return sentence.strip().split()
    
    def encode_sentence(self, tokens: List[str]) -> List[int]:
        """Convert tokens to indices using vocabulary."""
        return [
            self.vocab.word2idx.get(token, self.vocab.word2idx["<UNK>"]) 
            for token in tokens
        ]
    
    def predict(self, sentence: str) -> List[Tuple[str, str]]:
        """
        Predict NER tags for input sentence.
        
        Args:
            sentence: Input text string
            
        Returns:
            List of (token, tag) tuples
        """
        # Tokenize and encode
        tokens = self.tokenize(sentence)
        input_ids = self.encode_sentence(tokens)
        
        # Convert to tensor
        tensor = torch.tensor([input_ids], dtype=torch.long).to(self.device)
        mask = tensor != 0
        
        # Get predictions
        with torch.no_grad():
            predictions = self.model.predict(tensor, mask)
            pred_tags = [self.vocab.idx2tag[idx] for idx in predictions[0]]
        
        return list(zip(tokens, pred_tags))
    
    def predict_batch(self, sentences: List[str]) -> List[List[Tuple[str, str]]]:
        """
        Predict NER tags for multiple sentences.
        
        Args:
            sentences: List of input text strings
            
        Returns:
            List of prediction results for each sentence
        """
        return [self.predict(sent) for sent in sentences]
    
    def format_output(self, results: List[Tuple[str, str]], 
                     show_only_entities: bool = False) -> str:
        """
        Format prediction results for display.
        
        Args:
            results: List of (token, tag) tuples
            show_only_entities: If True, only show non-O tags
            
        Returns:
            Formatted string
        """
        output = []
        for token, tag in results:
            if show_only_entities and tag == "O":
                continue
            output.append(f"{token:20} -> {tag}")
        return "\n".join(output)
    
    def extract_entities(self, results: List[Tuple[str, str]]) -> dict:
        """
        Extract named entities from prediction results.
        
        Args:
            results: List of (token, tag) tuples
            
        Returns:
            Dictionary mapping entity types to lists of entities
        """
        entities = {}
        current_entity = []
        current_type = None
        
        for token, tag in results:
            if tag.startswith("B-"):
                # Save previous entity if exists
                if current_entity:
                    entity_text = " ".join(current_entity)
                    if current_type not in entities:
                        entities[current_type] = []
                    entities[current_type].append(entity_text)
                
                # Start new entity
                current_type = tag[2:]
                current_entity = [token]
                
            elif tag.startswith("I-") and current_entity:
                current_entity.append(token)
                
            else:
                # Save entity and reset
                if current_entity:
                    entity_text = " ".join(current_entity)
                    if current_type not in entities:
                        entities[current_type] = []
                    entities[current_type].append(entity_text)
                current_entity = []
                current_type = None
        
        # Don't forget last entity
        if current_entity:
            entity_text = " ".join(current_entity)
            if current_type not in entities:
                entities[current_type] = []
            entities[current_type].append(entity_text)
        
        return entities


def main():
    """Example usage of NER predictor."""
    
    # Initialize predictor
    print("Loading model...")
    predictor = NERPredictor(
        model_path="bilstm_crf_model.pth",
        vocab_path="vocab.json",
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    print("Model loaded successfully!\n")
    
    # Example 1: Single sentence prediction
    print("=" * 50)
    print("Example 1: Single Sentence Prediction")
    print("=" * 50)
    
    sentence = "daun padi menguning dan muncul bercak coklat"
    print(f"Input: {sentence}\n")
    
    results = predictor.predict(sentence)
    print("Token-level predictions:")
    print(predictor.format_output(results))
    
    print("\nExtracted entities:")
    entities = predictor.extract_entities(results)
    for entity_type, entity_list in entities.items():
        print(f"  {entity_type}: {entity_list}")
    
    # Example 2: Multiple sentences
    print("\n" + "=" * 50)
    print("Example 2: Batch Prediction")
    print("=" * 50)
    
    sentences = [
        "daun padi menguning dan muncul bercak coklat",
        "tanaman jagung diserang hama belalang",
        "buah tomat membusuk karena penyakit busuk daun"
    ]
    
    batch_results = predictor.predict_batch(sentences)
    
    for i, (sent, results) in enumerate(zip(sentences, batch_results), 1):
        print(f"\nSentence {i}: {sent}")
        entities = predictor.extract_entities(results)
        if entities:
            for entity_type, entity_list in entities.items():
                print(f"  {entity_type}: {entity_list}")
        else:
            print("  No entities found")
    
    # Example 3: Interactive mode
    print("\n" + "=" * 50)
    print("Example 3: Interactive Mode")
    print("=" * 50)
    print("Enter sentences for NER prediction (type 'quit' to exit):\n")
    
    while True:
        try:
            user_input = input(">>> ").strip()
            
            if user_input.lower() in ["quit", "exit", "q"]:
                print("Goodbye!")
                break
            
            if not user_input:
                continue
            
            results = predictor.predict(user_input)
            print("\nPredictions:")
            print(predictor.format_output(results))
            
            entities = predictor.extract_entities(results)
            if entities:
                print("\nEntities:")
                for entity_type, entity_list in entities.items():
                    print(f"  {entity_type}: {entity_list}")
            print()
            
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    main()