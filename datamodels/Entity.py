from pydantic import BaseModel
from typing import Optional


class Entity(BaseModel):
    text: str
    entity_type: Optional[str] = None