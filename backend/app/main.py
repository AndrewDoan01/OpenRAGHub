# main.py
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from sentence_transformers import SentenceTransformer
import chromadb
import re

# ===== CORE CONFIG =====
class Settings:
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    VECTOR_DB_PATH: str = "./chroma_db"
    MAX_CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 50
    MAX_COLLECTIONS: int = 10

settings = Settings()

# ===== ENHANCED BASE MODEL =====
class EnhancedBaseModel(BaseModel):
    @validator('*', pre=True)
    def empty_str_to_none(cls, v):
        return None if v == '' else v

# ===== MODELS =====
class DocumentModel(EnhancedBaseModel):
    id: str = Field(..., min_length=1, max_length=100)
    content: str = Field(..., min_length=10, max_length=10000)
    metadata: Optional[Dict[str, Any]] = {}

    @validator('content')
    def validate_content(cls, v):
        if len(v.split()) < 3:
            raise ValueError('Content is too short')
        return v

class EmbeddingModel(BaseModel):
    text: str
    vector: List[float]

class ChatModel(BaseModel):
    query: str
    context: List[str] = []

# ===== ADVANCED TEXT SPLITTER =====
class AdvancedTextSplitter:
    @staticmethod
    def split_text(
        text: str, 
        max_chunk_size: int = 500, 
        chunk_overlap: int = 50,
        min_chunk_length: int = 50
    ) -> List[str]:
        if not text:
            return []

        chunks = []
        for i in range(0, len(text), max_chunk_size - chunk_overlap):
            chunk = text[i:i + max_chunk_size]
            if len(chunk) >= min_chunk_length:
                chunks.append(chunk)
        
        return chunks

    @staticmethod
    def clean_text(text: str) -> str:
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

# ===== ENHANCED VECTOR STORE =====
class EnhancedVectorStore:
    def __init__(self, path: str, max_collections: int = 10):
        self.client = chromadb.PersistentClient(path=path)
        self.max_collections = max_collections
        self._validate_collections()

    def _validate_collections(self):
        collections = self.client.list_collections()
        if len(collections) >= self.max_collections:
            raise ValueError(f"Exceeded max collections limit of {self.max_collections}")

    def add_documents(self, documents: List[DocumentModel], embeddings: List[EmbeddingModel]):
        try:
            collection_name = f"documents_{len(self.client.list_collections()) + 1}"
            collection = self.client.create_collection(name=collection_name)
            
            for doc, emb in zip(documents, embeddings):
                collection.add(
                    ids=[doc.id],
                    embeddings=[emb.vector],
                    documents=[doc.content],
                    metadatas=[doc.metadata or {}]
                )
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Document storage failed: {str(e)}"
            )

    def search(self, query_embedding: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        try:
            results = []
            for collection in self.client.list_collections():
                collection_results = collection.query(
                    query_embeddings=[query_embedding],
                    n_results=top_k
                )
                results.extend(collection_results)
            return results
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Search failed: {str(e)}"
            )

# ===== SERVICES =====
class EmbeddingService:
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)

    def encode(self, texts: List[str]) -> List[EmbeddingModel]:
        # Làm sạch văn bản trước khi encode
        cleaned_texts = [AdvancedTextSplitter.clean_text(text) for text in texts]
        embeddings = self.model.encode(cleaned_texts)
        return [
            EmbeddingModel(text=text, vector=emb.tolist()) 
            for text, emb in zip(cleaned_texts, embeddings)
        ]

class DocumentService:
    def __init__(self, embedding_service: EmbeddingService, vector_store: EnhancedVectorStore):
        self.embedding_service = embedding_service
        self.vector_store = vector_store

    def process_document(self, document: str) -> List[DocumentModel]:
        # Làm sạch văn bản
        cleaned_document = AdvancedTextSplitter.clean_text(document)
        
        # Chia văn bản thành các chunk
        chunks = AdvancedTextSplitter.split_text(cleaned_document)
        
        # Tạo document models
        doc_models = [
            DocumentModel(id=f"chunk_{i}", content=chunk) 
            for i, chunk in enumerate(chunks)
        ]
        
        # Tạo embeddings
        embeddings = self.embedding_service.encode([doc.content for doc in doc_models])
        
        # Lưu vào vector store
        self.vector_store.add_documents(doc_models, embeddings)
        
        return doc_models

class ChatService:
    def __init__(self, embedding_service: EmbeddingService, vector_store: EnhancedVectorStore):
        self.embedding_service = embedding_service
        self.vector_store = vector_store

    def generate_response(self, query: ChatModel) -> str:
        # Làm sạch query
        cleaned_query = AdvancedTextSplitter.clean_text(query.query)
        
        # Tạo embedding cho query
        query_embedding = self.embedding_service.encode([cleaned_query])[0]
        
        # Tìm kiếm context
        context = self.vector_store.search(query_embedding.vector)
        
        # Đơn giản hóa response (có thể thay thế bằng LLM thực tế)
        context_texts = [str(ctx) for ctx in context]
        return f"Response based on: {' | '.join(context_texts)}"

# ===== API =====
app = FastAPI(
    title="Enhanced Document Processing API",
    description="Advanced backend for document embedding and retrieval",
    version="0.2.0"
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global Exception Handler
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "Internal Server Error",
            "detail": str(exc)
        }
    )

# Dependency Injection
embedding_service = EmbeddingService(settings.EMBEDDING_MODEL)
vector_store = EnhancedVectorStore(settings.VECTOR_DB_PATH, settings.MAX_COLLECTIONS)
document_service = DocumentService(embedding_service, vector_store)
chat_service = ChatService(embedding_service, vector_store)

# Endpoints
@app.post("/documents/")
async def upload_document(document: str):
    try:
        processed_docs = document_service.process_document(document)
        return {
            "status": "success", 
            "chunks": len(processed_docs),
            "details": [{"id": doc.id, "length": len(doc.content)} for doc in processed_docs]
        }
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
