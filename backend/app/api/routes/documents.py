from fastapi import APIRouter, HTTPException
from app.services.document_service import document_service

router = APIRouter()

@router.post("/documents/")
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
