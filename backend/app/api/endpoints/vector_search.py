from fastapi import APIRouter, HTTPException
from models.embedding import VectorSearchRequest, VectorSearchResponse
from services.vector_service import VectorService

router = APIRouter(prefix="/vector", tags=["vector"])
vector_service = VectorService()

@router.post("/search", response_model=VectorSearchResponse)
async def search_vector(request: VectorSearchRequest):
    try:
        results = await vector_service.search(request.embedding, request.top_k)
        return VectorSearchResponse(success=True, results=results)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
