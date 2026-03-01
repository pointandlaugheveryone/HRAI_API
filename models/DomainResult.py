from dataclasses import dataclass
from typing import List
from models.Suggestion import Suggestion


@dataclass
class DomainResult:
    domain: str
    occ_count: int
    occupations: List[Suggestion]