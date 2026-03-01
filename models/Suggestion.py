from dataclasses import dataclass

from models.Occupation import Occupation
from models.Skill import Skill
from typing import List


@dataclass
class Suggestion:
    occupation: Occupation
    missing_skills: List[Skill]
    match_score: float  # proportion of skills of label 'essential'