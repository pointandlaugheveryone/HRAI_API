from typing import List
from _models.Skill import Skill
from pydantic import BaseModel

class Occupation(BaseModel):
    id: str
    esco_uri: str
    label: str
    code: str
    isco_code: str
    score: float

class OccupationResponse(BaseModel):
    occupation: Occupation
    extra_skills: List[Skill]