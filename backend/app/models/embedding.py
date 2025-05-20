from pydantic import BaseModel
from typing import List

class EmbeddingRequest(BaseModel):
    text: str

class EmbeddingResponse(BaseModel):
    success: bool
    embedding: List[float]

class VectorSearchRequest(BaseModel):
    embedding: List[float]
    top_k: int = 5

class VectorSearchResponse(BaseModel):
    success: bool
    results: list
