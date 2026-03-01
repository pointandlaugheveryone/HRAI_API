from pydantic import BaseModel
from typing import List, Optional

from models.Skill import Skill


class ManualContentRequest(BaseModel):
    target_job: Optional[str]
    skills: List[Skill]

class ResumeRequest(BaseModel):
    target_job: Optional[str]
    content: str
