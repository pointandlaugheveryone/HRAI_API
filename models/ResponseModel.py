from pydantic import BaseModel
from typing import List, Optional

from models.Suggestion import Suggestion
from models.DomainResult import DomainResult


class SkillsResponse(BaseModel):
    suggestions: List[Suggestion]

    class Config:
        from_attributes = True


class SuggestionResponse(BaseModel):
    top_suggestion: Optional[Suggestion]
    target_suggestion: Optional[Suggestion]

    class Config:
        from_attributes = True


class DomainResponse(BaseModel):
    domains: List[DomainResult]

    class Config:
        from_attributes = True


class TargetJobResponse(BaseModel):
    top_suggestion: Optional[Suggestion]
    target_suggestion: Optional[Suggestion]

    class Config:
        from_attributes = True
