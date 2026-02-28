from dataclasses import dataclass

@dataclass
class Occupation:
    id: str
    esco_uri: str
    label: str
    code: str
    isco_code: str
    score: float