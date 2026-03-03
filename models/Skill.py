from dataclasses import dataclass
from typing import Optional


@dataclass
class Skill:
    id: str
    esco_uri: str
    label: str
    relation_type: str
    score: Optional[float]
    description: str