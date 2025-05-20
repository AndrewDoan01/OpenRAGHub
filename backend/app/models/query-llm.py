from pydantic import BaseModel
from typing import Optional

class QueryLLMRequest(BaseModel):
    prompt: str
    context: Optional[str] = None

class QueryLLMResponse(BaseModel):
    success: bool
    response: str
