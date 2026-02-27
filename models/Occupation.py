from typing import List
import Skill


class Occupation:
    id: str
    esco_uri: str
    label: str
    code: str
    isco_code: str
    score: float

class OccupationResponse:
    occupation: Occupation
    extra_skills: List[Skill]