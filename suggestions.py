from models.EntityResult import EntityResult
from models.Skill import Skill
from models.Occupation import Occupation
from typing import Dict, List, Set, Optional
from query import query_type
from models.Suggestion import Suggestion
from models.DomainResult import DomainResult, CODE_TO_DOMAIN
from setup import get_relations, get_metadata
from config import conf


def match_occupations(
    skills: List[EntityResult],
    occupations: Optional[List[EntityResult]],
    target_job: Optional[str]
) -> List[Suggestion]:
    occ_to_skill, skill_to_occ = get_relations()
    meta = get_metadata()

    # Build key_NNN → metadata dict for fast skill lookups
    skill_meta_by_key: Dict[str, dict] = {}
    skill_uri_to_key: Dict[str, str] = {}
    for m in meta:
        id = m['id']


        if uri and uri not in skill_uri_to_key:
            skill_uri_to_key[uri] = id

    # Build key_NNN → FAISS index for occupation metadata
    occ_uri_to_key: Dict[str, str] = {}
    for m in occ_meta:
        uri = m.get('esco_uri', '')
        if uri and uri not in occ_uri_to_key:
            occ_uri_to_key[uri] = m['id']

    # Resolve user's skills to key_NNN ids via esco_uri (works regardless of source index)
    skill_keys: Set[str] = set()
    for s in skills:
        key = skill_uri_to_key.get(s.esco_uri, '')
        if key:
            skill_keys.add(key)

    suggestions: List[Suggestion] = []

    if target_job:
        occupations = query_type([target_job], 'occupation', search_k=1)

    if not occupations:
        # Reverse lookup: find all occupations related to user's skills
        candidate_occ_keys: Set[str] = set()
        for sk in skill_keys:
            for occ_key, _ in skill_to_occ.get(sk, []):
                candidate_occ_keys.add(occ_key)

        # Build EntityResult stubs from occupation metadata
        # Need to find the FAISS index for each occ key
        occupations = []
        for i, meta in enumerate(occ_meta):
            if meta['id'] in candidate_occ_keys:
                candidate_occ_keys.discard(meta['id'])  # only take first FAISS entry per key
                occupations.append(EntityResult(
                    id=i,
                    cosine_score=0.0,
                    entity_type='occupation',
                    esco_uri=meta.get('esco_uri', ''),
                    label=meta.get('preferred_label', ''),
                    code=meta.get('code', ''),
                    isco_code=meta.get('isco_code', ''),
                    description='',
                ))

    for occ in occupations:
        occ_key = occ_uri_to_key.get(occ.esco_uri, '')
        if not occ_key:
            continue
        related_skills = occ_to_skill.get(occ_key, [])

        essential_count = 0
        matched_count = 0
        missing: List[Skill] = []

        for skill_key, relation_type in related_skills:
            if relation_type == 'essential':
                essential_count += 1

            if skill_key in skill_keys:
                matched_count += 1
            else:
                sm = skill_meta_by_key.get(skill_key, {})
                missing.append(Skill(
                    id=skill_key,
                    esco_uri=sm.get('esco_uri', ''),
                    label=sm.get('preferred_label', ''),
                    relation_type=relation_type,
                    score=0.0,
                    source_text='',
                ))

        if not target_job and matched_count < config.min_skills:
            continue

        total_essential = essential_count if essential_count else 1
        match_score = round(matched_count / max(total_essential, 1), 5)

        suggestions.append(Suggestion(
            occupation=Occupation(
                id=occ_key,
                esco_uri=occ.esco_uri,
                label=occ.label,
                code=occ.code,
                isco_code=occ.isco_code,
                score=occ.cosine_score,
            ),
            missing_skills=missing,
            match_score=match_score,
        ))

    suggestions.sort(key=lambda s: s.match_score, reverse=True)
    return suggestions


def get_domain_skills(suggestions: List[Suggestion]) -> List[DomainResult]:
    domain_suggestions: Dict[str, List[Suggestion]] = {}

    for suggestion in suggestions:
        domain_name = CODE_TO_DOMAIN[suggestion.occupation.code[:1]]
        if domain_name not in domain_suggestions:
            domain_suggestions[domain_name] = []
        domain_suggestions[domain_name].append(suggestion)

    domain_results: List[DomainResult] = []
    for domain_name, domain_suggestions in domain_suggestions.items():
        domain_results.append(DomainResult(
            domain=domain_name,
            occupations=sorted(domain_suggestions, key=lambda sgn: sgn.match_score, reverse=True)
        ))

    return sorted(domain_results, key=lambda d: len(d.occupations), reverse=True)