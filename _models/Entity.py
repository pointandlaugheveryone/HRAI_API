from typing import Literal
from pydantic import BaseModel

class Entity(BaseModel):
    text: str
    entity_type:  Literal[
        'occupation',
        'skill',
        'isco_group',
        'skill_group',
        'all'
    ]