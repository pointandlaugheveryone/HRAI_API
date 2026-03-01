from dataclasses import dataclass, field
from models.EntityResult import EntityResult
from models.Skill import Skill
from models.Occupation import Occupation
from typing import Dict, List, Set

from models.Suggestion import Suggestion
from setup import get_relations

def match_occupations(
    extracted_skills: List[EntityResult],
    occupations: List[EntityResult],
) -> List[Suggestion]:
    """
    For each occupation, look up its essential/optional skills via get_relations,
    compare against extracted_skills, and return occupations ranked by how many
    essential skills are already present — along with the missing ones.
    """
    relations: Dict[str, List[Dict[str, str]]] = get_relations()
    extracted_uris: Set[str] = {s.esco_uri for s in extracted_skills}

    suggestions: List[Suggestion] = []

    for occ in occupations:
        occ_uri = occ.esco_uri
        related_skills = relations.get(occ_uri, [])

        essential_skills = [s for s in related_skills if s.get('relation_type') == 'essential']
        optional_skills = [s for s in related_skills if s.get('relation_type') == 'optional']

        all_related = essential_skills + optional_skills
        missing: List[Skill] = []
        matched_count = 0

        for skill_rel in all_related:
            skill_uri = skill_rel.get('skill_uri', '')
            if skill_uri in extracted_uris:
                matched_count += 1
            else:
                missing.append(Skill(
                    id=skill_rel.get('id', ''),
                    esco_uri=skill_uri,
                    label=skill_rel.get('preferred_label', ''),
                    relation_type=skill_rel.get('relation_type', ''),
                    score=0.0,
                    source_text='',
                ))

        total_essential = len(essential_skills) if essential_skills else 1
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
    return suggestions