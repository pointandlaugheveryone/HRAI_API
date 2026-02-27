from pydantic import BaseModel
from typing import List, Optional
from .Entity import Entity


class LookupRequest(BaseModel):
    entities: List[Entity]
    top_n: int = 10
    extra_skill_limit: int = 25


class ResumeLookup(BaseModel):
    resume_text: Optional[str] = None
    job_title: Optional[str] = None
    use_spacy: bool = False
    skills: Optional[str] = None
    occupations: Optional[str] = None
    top_n: int = 10