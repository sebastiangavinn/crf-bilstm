"""
API Backend untuk Question Answering System
Menggunakan FastAPI dan modul-modul dari src/
"""

import torch
import sys
from pathlib import Path
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Setup path untuk import modul src
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

# Initialize FastAPI app
app = FastAPI(
    title="Question Answering API",
    description="API untuk Named Entity Recognition (NER) pada sistem Question Answering Hama dan Penyakit Padi",
    version="1.0.0"
)

# CORS middleware untuk frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Dalam production, ganti dengan domain spesifik
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables untuk model, vocabulary, dan knowledge graph
model = None
vocab = None
device = None
qa_pipeline = None
kg = None


# =====================================
# Pydantic Models untuk Request/Response
# =====================================

class QuestionInput(BaseModel):
    question: str

class TextInput(BaseModel):
    """Model untuk input teks tunggal"""
    text: str


class BatchTextInput(BaseModel):
    """Model untuk input batch teks"""
    texts: List[str]


class TokenTag(BaseModel):
    """Model untuk token dan tag"""
    token: str
    tag: str


class EntityResult(BaseModel):
    """Model untuk hasil entitas"""
    entity_type: str
    entities: List[str]


class NERResponse(BaseModel):
    """Response untuk prediksi NER"""
    text: str
    token_tags: List[TokenTag]
    entities: List[EntityResult]


class BatchNERResponse(BaseModel):
    """Response untuk batch prediction"""
    results: List[NERResponse]


class HealthResponse(BaseModel):
    """Response untuk health check"""
    status: str
    model_loaded: bool
    device: str
    vocab_size: Optional[int] = None
    tag_size: Optional[int] = None
    kg_connected: bool = False


# =====================================
# Helper Functions
# =====================================

# extract_entities sudah diimport dari src.utils.inference


def load_model_and_vocab(model_path: str = "bilstm_crf.pth", vocab_path: str = "vocab.json",
                         kg_uri: Optional[str] = None, kg_user: str = "neo4j", 
                         kg_password: str = "password", kg_database: str = "hamapenyakit"):
    """
    Load model, vocabulary, dan knowledge graph ke memory
    
    Args:
        model_path: Path ke file model
        vocab_path: Path ke file vocabulary
        kg_uri: Neo4j URI (optional, jika None maka KG tidak akan dimuat)
        kg_user: Neo4j username
        kg_password: Neo4j password
        kg_database: Neo4j database name
    """
    global model, vocab, device, qa_pipeline, kg
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load vocabulary
    vocab = Vocabulary()
    vocab.load(vocab_path)
    
    # Load model
    model = BiLSTM_CRF(
        vocab_size=len(vocab.word2idx),
        tagset_size=len(vocab.tag2idx),
        emb_dim=128,
        hidden_dim=128
    ).to(device)
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    print(f"✅ Model loaded successfully on {device}")
    print(f"✅ Vocabulary loaded: {len(vocab.word2idx)} words, {len(vocab.tag2idx)} tags")
    
    # Load Knowledge Graph jika URI diberikan
    if kg_uri:
        try:
            kg = KnowledgeGraph(kg_uri, kg_user, kg_password, kg_database)
            print(f"✅ Knowledge Graph connected: {kg_database}")
        except Exception as e:
            print(f"⚠️  Warning: Could not connect to Knowledge Graph: {e}")
            kg = None
    else:
        kg = None
        print("ℹ️  Knowledge Graph not configured (no URI provided)")
    
    # Initialize QA Pipeline
    qa_pipeline = QAPipeline(model, vocab, device, kg)
    print("✅ QA Pipeline initialized")


# =====================================
# API Endpoints
# =====================================

@app.on_event("startup")
async def startup_event():
    """Load model dan knowledge graph saat aplikasi startup"""
    import os
    
    root_dir = Path(__file__).parent.parent.absolute()
    model_path = root_dir / "bilstm_crf.pth"
    vocab_path = root_dir / "vocab.json"
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not vocab_path.exists():
        raise FileNotFoundError(f"Vocabulary file not found: {vocab_path}")
    
    # Get Neo4j config from environment variables or use defaults
    kg_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    kg_user = os.getenv("NEO4J_USER", "neo4j")
    kg_password = os.getenv("NEO4J_PASSWORD", "password")
    kg_database = os.getenv("NEO4J_DATABASE", "hamapenyakit")
    
    # Set kg_uri to None if you want to disable KG (for testing without Neo4j)
    # kg_uri = None
    
    load_model_and_vocab(
        str(model_path), 
        str(vocab_path),
        kg_uri=kg_uri,
        kg_user=kg_user,
        kg_password=kg_password,
        kg_database=kg_database
    )


@app.get("/", response_model=HealthResponse)
async def root():
    """Root endpoint - health check"""
    return HealthResponse(
        status="ok",
        model_loaded=model is not None,
        device=device or "unknown",
        vocab_size=len(vocab.word2idx) if vocab else None,
        tag_size=len(vocab.tag2idx) if vocab else None,
        kg_connected=kg is not None
    )


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="ok",
        model_loaded=model is not None,
        device=device or "unknown",
        vocab_size=len(vocab.word2idx) if vocab else None,
        tag_size=len(vocab.tag2idx) if vocab else None,
        kg_connected=kg is not None
    )


@app.post("/predict", response_model=NERResponse)
async def predict_ner(input_data: TextInput):
    """
    Prediksi NER untuk satu kalimat
    
    Args:
        input_data: TextInput dengan field 'text'
        
    Returns:
        NERResponse dengan token_tags dan entities
    """
    if model is None or vocab is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if not input_data.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    
    try:
        # Prediksi
        token_tags = predict_sentence(model, input_data.text, vocab, device)
        
        # Ekstrak entitas
        entities_dict = extract_entities(token_tags)
        
        # Format response
        token_tag_list = [TokenTag(token=token, tag=tag) for token, tag in token_tags]
        entity_results = [
            EntityResult(entity_type=ent_type, entities=entities)
            for ent_type, entities in entities_dict.items()
        ]
        
        return NERResponse(
            text=input_data.text,
            token_tags=token_tag_list,
            entities=entity_results
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/predict/batch", response_model=BatchNERResponse)
async def predict_batch(input_data: BatchTextInput):
    """
    Prediksi NER untuk beberapa kalimat sekaligus (batch)
    
    Args:
        input_data: BatchTextInput dengan field 'texts' (list of strings)
        
    Returns:
        BatchNERResponse dengan list of NERResponse
    """
    if model is None or vocab is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if not input_data.texts:
        raise HTTPException(status_code=400, detail="Texts list cannot be empty")
    
    if len(input_data.texts) > 100:  # Limit batch size
        raise HTTPException(status_code=400, detail="Batch size cannot exceed 100")
    
    try:
        results = []
        
        for text in input_data.texts:
            if not text.strip():
                continue
            
            # Prediksi
            token_tags = predict_sentence(model, text, vocab, device)
            
            # Ekstrak entitas
            entities_dict = extract_entities(token_tags)
            
            # Format response
            token_tag_list = [TokenTag(token=token, tag=tag) for token, tag in token_tags]
            entity_results = [
                EntityResult(entity_type=ent_type, entities=entities)
                for ent_type, entities in entities_dict.items()
            ]
            
            results.append(NERResponse(
                text=text,
                token_tags=token_tag_list,
                entities=entity_results
            ))
        
        return BatchNERResponse(results=results)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")


@app.post("/extract-entities")
async def extract_entities_only(input_data: TextInput):
    """
    Hanya ekstrak entitas tanpa detail token-tag
    
    Args:
        input_data: TextInput dengan field 'text'
        
    Returns:
        Dictionary dengan entities saja
    """
    if model is None or vocab is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if not input_data.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    
    try:
        # Prediksi
        token_tags = predict_sentence(model, input_data.text, vocab, device)
        
        # Ekstrak entitas
        entities_dict = extract_entities(token_tags)
        
        return {
            "text": input_data.text,
            "entities": entities_dict
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Extraction error: {str(e)}")


@app.post("/qa")
async def qa_single_sentence(input_data: QuestionInput):
    """
    Question Answering untuk satu kalimat
    """
    if qa_pipeline is None:
        raise HTTPException(status_code=503, detail="QA Pipeline not initialized")

    question = input_data.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    try:
        result = qa_pipeline.answer_with_reasoning(question)

        return {
            "question": question,
            "intent": result.get("intent"),
            "entities": result.get("entities"),
            "answer": result.get("answer"),
            "confidence": result.get("confidence"),
            "evidence": result.get("reasoning", {}).get("results", [])
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

