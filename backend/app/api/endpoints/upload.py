from fastapi import APIRouter, File, UploadFile, HTTPException
from services.file_service import FileService
from models.upload import UploadResponse

router = APIRouter(prefix="/upload", tags=["upload"])
file_service = FileService()

@router.post("/", response_model=UploadResponse)
async def upload_file(file: UploadFile = File(...)):
    try:
        file_info = await file_service.save_file(file)
        return UploadResponse(success=True, file_info=file_info)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
