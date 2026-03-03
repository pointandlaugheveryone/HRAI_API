from dataclasses import dataclass

@dataclass
class EntityResult:
    id: str
    cosine_score: float
    entity_type: str
    esco_uri: str
    label: str
    code: str
    description: str
