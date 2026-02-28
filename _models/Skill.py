from typing import Optional
from pydantic import BaseModel

class Skill(BaseModel):
    id: str
    esco_uri: str
    label: str
    relation_type: str
    score: float
    source_text: str