from typing import List, Optional
import Skill
import Domain
from Occupation import OccupationResponse
import Entity
from pydantic import BaseModel

from settings import config


class LookupRequest(BaseModel):
    entities: List[Entity]
    top_n: int = config.top_result_n
    extra_skill_limit: int = 25


class ResumeLookup(BaseModel):
    resume_text: Optional[str] = None
    job_title: Optional[str] = None
    skills: Optional[str] = None
    occupations: Optional[str] = None
    top_n: int = 10

class LookupResponse(BaseModel):
    extracted_skills: List[Skill]
    domain_recommendations: List[Domain]
    occupation_suggestions: List[OccupationResponse]
