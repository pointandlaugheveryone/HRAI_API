from models.EntityResult import EntityResult
from models.Skill import Skill
from models.Occupation import Occupation
from typing import Dict, List, Set, Optional
from query import query_type
from models.Suggestion import Suggestion
from models.DomainResult import DomainResult
from setup import get_relations


ISCO_MAP = {
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

def match_occupations(
    skills: List[EntityResult],
    occupations: Optional[List[EntityResult]],
    target_job: Optional[str]
) -> List[Suggestion]:
    """
    score how well extracted skills match the occupation's esco skill list
    """

    relations: Dict[str, List[Dict[str, str]]] = get_relations()
    extracted_uris: Set[str] = {s.esco_uri for s in skills}

    suggestions: List[Suggestion] = []

    target_matches: List[EntityResult] = []
    if target_job:
        # When a target label is supplied, only consider the closest matching occupation
        target_matches = query_type([target_job], 'occupation', search_k=1)
        if not target_matches:
            return []

    selected_occupations = target_matches if target_matches else (occupations or [])
    if not selected_occupations:
        return []

    for occ in selected_occupations:
        # lookup essential/optional skills 
        occ_uri = occ.esco_uri
        related_skills = relations.get(occ_uri, [])
        essential_skills = [s for s in related_skills if s.get('relation_type') == 'essential']

        missing: List[Skill] = []
        matched_count = 0
        for skill in related_skills:
            uri = skill.get('skill_uri', '')
            if uri in extracted_uris: matched_count += 1
            else:
                missing.append(Skill(
                    id=skill.get('id', ''),
                    esco_uri=uri,
                    label=skill.get('preferred_label', ''),
                    relation_type=skill.get('relation_type', ''),
                    score=0.0,
                    source_text='',
                ))

        total_essential = len(essential_skills) if essential_skills else 1
        # Match score is normalized by essential skills so it stays comparable across jobs
        match_score = matched_count / max(total_essential, 1)

        occupation_model = Occupation(
            id=occ.esco_uri,
            esco_uri=occ.esco_uri,
            label=occ.label,
            code=occ.code,
            isco_code=occ.isco_code,
            score=occ.cosine_score,
        )

        suggestions.append(Suggestion(
            occupation=occupation_model,
            missing_skills=missing,
            match_score=match_score,
        ))

    suggestions.sort(key=lambda s: s.match_score, reverse=True)
    return suggestions[:1] if target_job else suggestions


def get_domain_skills(suggestions: List[Suggestion]) -> List[DomainResult]:
    """
    group by top level ISCO domain
    return domain lists ordered by [occupation count ordered by match_score]
    """
    domain_to_suggestions: Dict[str, List[Suggestion]] = {}
    for suggestion in suggestions:
        code = suggestion.occupation.code or ''
        domain_key = code[:1] if code else ''
        domain_name = ISCO_MAP.get(domain_key, 'Unknown')
        domain_to_suggestions.setdefault(domain_name, []).append(suggestion)

    domain_results: List[DomainResult] = []
    for domain_name, domain_suggestions in domain_to_suggestions.items():
        sorted_suggestions = sorted(domain_suggestions, key=lambda s: s.match_score, reverse=True)
        domain_results.append(DomainResult(
            domain=domain_name,
            occ_count=len(domain_suggestions),
            occupations=sorted_suggestions,
        ))

    return sorted(domain_results, key=lambda d: d.occ_count, reverse=True)