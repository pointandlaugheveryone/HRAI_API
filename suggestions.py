from models.EntityResult import EntityResult
from models.Skill import Skill
from models.Occupation import Occupation
from models.Suggestion import Suggestion
from models.DomainResult import DomainResult, CODE_TO_DOMAIN
from load import get_relations

from typing import Dict, List


def _count_essential_skills(skills: List[Skill]) -> int:
    return sum(1 for skill in skills if skill.relation_type == 'essential')


def _is_better_suggestion(new: Suggestion, current: Suggestion) -> bool:
    new_essential = _count_essential_skills(new.missing_skills)
    current_essential = _count_essential_skills(current.missing_skills)
    if new_essential != current_essential:
        return new_essential < current_essential

    if len(new.missing_skills) != len(current.missing_skills):
        return len(new.missing_skills) < len(current.missing_skills)

    return (new.occupation.cosine_score or 0) > (current.occupation.cosine_score or 0)


def deduplicate(suggestions: List[Suggestion]) -> List[Suggestion]:
    best_by_occ: Dict[str, Suggestion] = {}
    for suggestion in suggestions:
        current = best_by_occ.get(suggestion.occupation.id)
        if current is None or _is_better_suggestion(suggestion, current):
            best_by_occ[suggestion.occupation.id] = suggestion
    return list(best_by_occ.values())


def get_expanded_skills(
    metadata:Dict[str,Dict],
    entities: List[EntityResult]
) -> List[Suggestion]:

    skills = [e for e in entities if e.entity_type == 'skill']
    occupations = [e for e in entities if e.entity_type in {'occupation', 'isco_group'}]
    skill_set = set([s.id for s in skills])

    occ_to_skill, skill_to_occ = get_relations()
    suggestions = []

    # expand skills for occupations/jobs that user already has
    for occ in occupations:
        occ_skills = occ_to_skill.get(occ.id, {})
        missing_skills = []

        for skill_id,rel_type in occ_skills.items():
            skill_meta = metadata.get(skill_id, {})
            if skill_id in skill_set: continue
            else:
                missing_skills.append(Skill(
                    id=skill_id,
                    esco_uri=skill_meta.get('esco_uri',''),
                    label=skill_meta.get('preferred_label', ''),
                    relation_type=rel_type,
                    description=skill_meta.get('description',''),
                ))

        # create occ object with information
        occ_meta = metadata.get(occ.id,{})
        job_title = occ_meta.get('preferred_label', '')
        alt_label = occ_meta.get('alt_label', '')
        if alt_label:
            job_title = f"{job_title} / {alt_label}"
        occ_match = Occupation(
            id=occ.id,
            cosine_score=occ.cosine_score,
            esco_uri=occ.esco_uri,
            label=job_title,
            code=occ.code,
            description=occ_meta.get('description',''),
        )
        suggestions.append(Suggestion(
            occupation=occ_match,
            missing_skills=missing_skills
        ))

    return deduplicate(suggestions)


def get_domain_reports(suggestions: List[Suggestion]) -> List[DomainResult]: #TODO: add optional param target_domain (eg. it) and score/return it first
    domain_to_suggestions: Dict[str, List[Suggestion]] = {}
    deduped_suggestions = deduplicate(suggestions)
    for suggestion in deduped_suggestions:
        code = str(suggestion.occupation.code) or ''
        domain_key = code[:2]
        domain_name = CODE_TO_DOMAIN.get(domain_key, 'Other')
        domain_to_suggestions.setdefault(domain_name, []).append(suggestion)

    domain_results: List[DomainResult] = []
    for domain_name, domain_suggestions in domain_to_suggestions.items():
        domain_results.append(DomainResult(
            domain=domain_name,
            occ_count=len(domain_suggestions),
            occupations=domain_suggestions,
        ))

    return sorted(domain_results, key=lambda d: d.occ_count, reverse=True)