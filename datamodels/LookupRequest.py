from pydantic import BaseModel
from typing import List
from .Entity import Entity


class LookupRequest(BaseModel):
    entities: List[Entity]
    top_k_per_input: int = 10
    extra_skill_limit: int = 25