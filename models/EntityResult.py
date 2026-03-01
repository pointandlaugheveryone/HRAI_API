from dataclasses import dataclass

@dataclass
class EntityResult:
    id: int
    cosine_score: float
    entity_type: str
    esco_uri: str
    label: str
    code: str
    isco_code: str
    description: str

