from fastapi import APIRouter, HTTPException
from models.response import ModelResponseRequest, ModelResponse
from services.llm_service import LLMService

router = APIRouter(prefix="/model", tags=["model"])
llm_service = LLMService()

@router.post("/response", response_model=ModelResponse)
async def get_model_response(request: ModelResponseRequest):
    try:
        answer = await llm_service.model_response(request.input)
        return ModelResponse(success=True, answer=answer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
