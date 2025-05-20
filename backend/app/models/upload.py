from pydantic import BaseModel

class UploadResponse(BaseModel):
    success: bool
    file_info: dict
