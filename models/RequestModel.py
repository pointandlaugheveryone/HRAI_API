from pydantic import BaseModel
from typing import List

from models.Skill import Skill


class ManualContentRequest(BaseModel):
    skills: List[Skill]

class ManualContentLabeledRequest(BaseModel):
    target_job: str
    skills: List[Skill]

class ResumeRequest(BaseModel):
    content: str
