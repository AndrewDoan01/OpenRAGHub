from pydantic import BaseModel

class ModelResponseRequest(BaseModel):
    input: str

class ModelResponse(BaseModel):
    success: bool
    answer: str
