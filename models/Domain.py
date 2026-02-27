from typing import List
from models import Occupation, Skill
from pydantic import BaseModel


class Domain(BaseModel):
    domain_code: str
    occupation: Occupation
    extra_skills: List[Skill]