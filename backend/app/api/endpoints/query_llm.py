from fastapi import APIRouter, HTTPException
from models.query_llm import QueryLLMRequest, QueryLLMResponse
from services.llm_service import LLMService

router = APIRouter(prefix="/llm", tags=["llm"])
llm_service = LLMService()

@router.post("/query", response_model=QueryLLMResponse)
async def query_llm(request: QueryLLMRequest):
    try:
        result = await llm_service.query(request.prompt, request.context)
        return QueryLLMResponse(success=True, response=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
