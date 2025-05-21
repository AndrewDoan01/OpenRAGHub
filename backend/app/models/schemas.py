from pydantic import BaseModel
from typing import List, Optional

class EnhancedBaseModel(BaseModel):
    class Config:
        orm_mode = True

class DocumentModel(EnhancedBaseModel):
    id: str
    content: str
    metadata: Optional[dict] = None

class EmbeddingModel(EnhancedBaseModel):
    id: str
    embedding: List[float]

class ChatModel(EnhancedBaseModel):
    query: str
    n_results: int = 3
