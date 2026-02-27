from lookups import load_data, metadata_get_by_id, load_relations
from main import config
from datamodels import Skill
from typing import List
import numpy as np


def pick_best_per_domain(occ_scores):
    indexes = load_data()
    occ_by_id = metadata_get_by_id(indexes["occupation"]["metadata"]) # todo: pass indexes obj or individual index in parameter instead

    best_domain_occ = {}
    for occ_id, score in occ_scores.items():
        meta = occ_by_id.get(occ_id)
        code = meta.get("code")
        domain = code[0] # TODO: map to domain edum to display the domains name
        current = best_domain_occ.get(domain)
        if not current or score > current[1]:
            best_domain_occ[domain] = (occ_id, score)

    return best_domain_occ


def expand_skills(
        occ_id: str,
        input_encoded: np.ndarray,
        limit = config.match_cutoff,
):
    occupation_to_skills, _ = load_relations()
    indexes = load_data()
    skill_by_id = metadata_get_by_id(indexes["skill"]["metadata"])

    related = occupation_to_skills.get(occ_id, [])

    mean_vec = input_encoded.mean(axis=0)
    norm = np.linalg.norm(mean_vec)
    if norm == 0: return []
    mean_vec = mean_vec / norm

    new_skills: List[Skill] = []
    for skill_id, relation_type in related:
        idx = indexes["skill"]["id_to_idx"].get(skill_id)
        skill_vec = np.array(indexes["skill"]["index"].reconstruct(idx)) # lossy reconstruction of vector already stored in an index, does not actually affect performance
        score = float(np.dot(mean_vec, skill_vec))
        if score >= config.match_cutoff:
            continue
        meta = skill_by_id.get(skill_id)
        new_skills.append(
            Skill(
                id=skill_id,
                esco_uri=meta.get("esco_uri", ""),
                label=meta.get("preferred_label", ""),
                relation_type=relation_type,
                score=score,
            )
        )
        if len(new_skills) >= config.extra_skills_n: break

    return new_skills.sort(key=lambda s: s.score, reverse=True)