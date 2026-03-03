from dataclasses import dataclass

@dataclass
class Occupation:
    id: str
    cosine_score: float
    esco_uri: str
    label: str
    code: str
    description: str