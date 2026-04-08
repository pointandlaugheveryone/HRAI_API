from dataclasses import dataclass


@dataclass
class Skill:
    id: str
    esco_uri: str
    label: str
    relation_type: str
    description: str