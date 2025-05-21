# main.py
import os
from typing import List, Dict, Any
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import numpy as np
import chromadb

# ===== CORE CONFIG =====
class Settings:
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    VECTOR_DB_PATH: str = "./chroma_db"
    MAX_CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 50

settings = Settings()

# ===== MODELS =====
class DocumentModel(BaseModel):
    id: str
    content: str
    metadata: Dict[str, Any] = {}

class EmbeddingModel(BaseModel):
    text: str
    vector: List[float]

class ChatModel(BaseModel):
    query: str
    context: List[str] = []

# ===== UTILS =====
class TextSplitter:
    @staticmethod
    def split_text(text: str, max_chunk_size: int = 500, chunk_overlap: int = 50) -> List[str]:
        chunks = []
        for i in range(0, len(text), max_chunk_size - chunk_overlap):
            chunk = text[i:i + max_chunk_size]
            chunks.append(chunk)
        return chunks

class VectorStore:
    def __init__(self, path: str):
        self.client = chromadb.PersistentClient(path=path)
        self.collection = self.client.get_or_create_collection("documents")

    def add_documents(self, documents: List[DocumentModel], embeddings: List[EmbeddingModel]):
        for doc, emb in zip(documents, embeddings):
            self.collection.add(
                ids=[doc.id],
                embeddings=[emb.vector],
                documents=[doc.content],
                metadatas=[doc.metadata]
            )

    def search(self, query_embedding: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )
        return results

# ===== SERVICES =====
class EmbeddingService:
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)

    def encode(self, texts: List[str]) -> List[EmbeddingModel]:
        embeddings = self.model.encode(texts)
        return [
            EmbeddingModel(text=text, vector=emb.tolist()) 
            for text, emb in zip(texts, embeddings)
        ]

class DocumentService:
    def __init__(self, embedding_service: EmbeddingService, vector_store: VectorStore):
        self.embedding_service = embedding_service
        self.vector_store = vector_store

    def process_document(self, document: str) -> List[DocumentModel]:
        chunks = TextSplitter.split_text(document)
        doc_models = [
            DocumentModel(id=f"chunk_{i}", content=chunk) 
            for i, chunk in enumerate(chunks)
        ]
        embeddings = self.embedding_service.encode([doc.content for doc in doc_models])
        
        self.vector_store.add_documents(doc_models, embeddings)
        return doc_models

class ChatService:
    def __init__(self, embedding_service: EmbeddingService, vector_store: VectorStore):
        self.embedding_service = embedding_service
        self.vector_store = vector_store

    def generate_response(self, query: ChatModel) -> str:
        query_embedding = self.embedding_service.encode([query.query])[0]
        context = self.vector_store.search(query_embedding.vector)
        
        # Đơn giản hóa - bạn có thể thay thế bằng LLM thực tế
        return f"Response based on: {context}"

# ===== API =====
app = FastAPI()

# Dependency Injection
embedding_service = EmbeddingService(settings.EMBEDDING_MODEL)
vector_store = VectorStore(settings.VECTOR_DB_PATH)
document_service = DocumentService(embedding_service, vector_store)
chat_service = ChatService(embedding_service, vector_store)

@app.post("/documents/")
async def upload_document(document: str):
    try:
        processed_docs = document_service.process_document(document)
        return {"status": "success", "chunks": len(processed_docs)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/chat/")
async def chat(query: ChatModel):
    try:
        response = chat_service.generate_response(query)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# ===== MAIN =====
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
