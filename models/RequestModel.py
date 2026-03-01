from pydantic import BaseModel
from typing import Optional


class ManualContentRequest(BaseModel):
    target_job: Optional[str]
    skills: str

class ResumeRequest(BaseModel):
    target_job: Optional[str]
    content: str

class QueryRequest(BaseModel):
    text: str
    entity_type: Optional[str] = None
    n: int = 5