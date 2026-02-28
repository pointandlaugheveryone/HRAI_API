from typing import List
from _models.Domain import Domain
from pydantic import BaseModel

class RequestModel(BaseModel):
    occupations: List[Domain]