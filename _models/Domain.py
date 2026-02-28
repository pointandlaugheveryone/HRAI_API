from typing import List
from _models.Occupation import Occupation
from _models.Skill import Skill
from pydantic import BaseModel


class Domain(BaseModel):
    domain_code: str
    occupation: Occupation
    extra_skills: List[Skill]