from dataclasses import dataclass
from typing import List, Dict
from models.Suggestion import Suggestion


@dataclass
class DomainResult:
    domain: str
    occupations: List[Suggestion]


CODE_TO_DOMAIN: Dict = {
    "0" : "Zaměstnanci v ozbrojených silách",
    "1" : "Zákonodárci, vedoucí a řídící pracovníci",
    "2" : "Specialisté",
    "3" : "Techničtí a odborní pracovníci",
    "4" : "Úředníci",
    "5" : "Pracovníci ve službách a prodeji",
    "6" : "Kvalifikovaní pracovníci v zemědělství, lesnictví a rybářství",
    "7" : "Řemeslníci a opraváři",
    "8" : "Obsluha strojů a zařízení, montéři",
    "9" : "Pomocní a nekvalifikovaní pracovníci"
}