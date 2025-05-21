from fastapi import APIRouter, HTTPException
from app.models.schemas import ChatModel
from app.services.chat_service import chat_service

router = APIRouter()

@router.post("/chat/")
async def chat(query: ChatModel):
    try:
        response = chat_service.generate_response(query)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
