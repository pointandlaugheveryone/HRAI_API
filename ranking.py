from lookups import load_data, get_meta_by_id, load_relations
from datamodels import Skill
from typing import Dict, List, Optional
import numpy as np


def best_occ_per_domain(occ_scores):
    """
    esco categorises all occupations into top 10 most generic domains
    being part of the project's goal, this function finds an occupation from each of these
    """
    indexes = load_data()
    occ_by_id = get_meta_by_id(indexes['occupation']['metadata'])

    best_domain_occ = {}
    for occ_id, score in occ_scores.items():
        meta = occ_by_id.get(occ_id)
        code = meta.get('code')
        domain = code[0]
        current = best_domain_occ.get(domain)
        if not current or score > current[1]:
            best_domain_occ[domain] = (occ_id, score)

    return best_domain_occ


def skills_to_score(skills, skill_to_occupations) -> Dict[str, float]:
    occ_scores = {}
    for skill in skills:
        for occ_id, relation_type in skill_to_occupations.get(skill.id, []):
            weight = 1.0 if relation_type.lower() == 'essential' else 0.7

            occ_scores[occ_id] = max(occ_scores.get(occ_id,''),skill.score * weight)
    return occ_scores


def expand_skills(
        occ_id: str,
        input_encoded: np.ndarray,  # vectorised skills extracted from occupation
        limit: int = 5,
) -> List[Skill]:
    # get top related skills ranked by cosine similarity against mean of input_encoded

    occupation_to_skills, _ = load_relations()
    indexes = load_data()
    skill_by_id = get_meta_by_id(indexes['skill']['metadata'])
    related = occupation_to_skills.get(occ_id, [])

    # reference vector to rank skills against
    ref_vec = np.mean(np.atleast_2d(input_encoded), axis=0).astype(np.float32)

    # Normalise the reference vector for cosine similarity calculation
    norm = np.linalg.norm(ref_vec)
    if norm > 0: ref_vec /= norm

    scored_skills = []
    for id, relation in related:
        skill_idx = indexes['skill']['id_to_idx'].get(id)
        skill_vec = np.array(
            indexes['skill']['index'].reconstruct(skill_idx),
            dtype=np.float32,
        )
        score = float(np.dot(ref_vec, skill_vec))
        scored_skills.append((id, relation, score))

    scored_skills.sort(key=lambda t: t[2], reverse=True)
    new_skills: List[Skill] = []
    for id, relation, score in scored_skills[:limit]:
        meta = skill_by_id.get(id)
        new_skills.append(
            Skill(
                id=id,
                esco_uri=meta.get('esco_uri', ''),
                label=meta.get('preferred_label', ''),
                relation_type=relation,
                score=score,
            ))

    return new_skills
