from pydantic import BaseModel
from typing import List, Optional


class Entity(BaseModel):
    text: str
    entity_type: Optional[str] = None


class Skill(BaseModel):
    id: str
    esco_uri: str
    label: str
    relation_type: Optional[str]
    score: float


class OccupationInfo(BaseModel):
    id: str
    esco_uri: str
    label: str
    code: str
    isco_code: str
    score: float


class DomainInfo(BaseModel):
    domain_code: str
    occupation: OccupationInfo
    extra_skills: List[Skill]
