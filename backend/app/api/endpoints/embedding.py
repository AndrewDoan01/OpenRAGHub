from fastapi import APIRouter, HTTPException
from models.embedding import EmbeddingRequest, EmbeddingResponse
from services.embedding_service import EmbeddingService

router = APIRouter(prefix="/embedding", tags=["embedding"])
embedding_service = EmbeddingService()

@router.post("/", response_model=EmbeddingResponse)
async def get_embedding(request: EmbeddingRequest):
    try:
        vector = await embedding_service.create_embedding(request.text)
        return EmbeddingResponse(success=True, embedding=vector)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
