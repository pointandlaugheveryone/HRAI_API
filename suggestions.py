from typing import Dict, List

from _models.Domain import Domain
from _models.Occupation import OccupationResponse


def build_domain_recommendations(
        occ_scores: Dict[str, float],
        indexes: Dict[str, Dict],
        extra_skill_limit: int,
        reference_vectors: np.ndarray | None,
) -> List[Domain]:
    occ_by_id = get_meta_by_id(indexes["occupation"]["metadata"])
    best_by_domain = best_occ_per_domain(occ_scores)

    recommendations: List[Domain] = []
    for domain, (occ_id, score) in sorted(best_by_domain.items(), key=lambda i: i[0]):
        meta = occ_by_id.get(occ_id)
        if not meta:
            continue
        reference = reference_vectors
        if reference is None:
            idx = indexes['occupation']['id_to_idx'].get(occ_id)
            reference = np.array(indexes['occupation']['index'].reconstruct(idx))
        extra_skills = expand_skills(
            occ_id=occ_id,
            input_encoded=reference,
            limit=extra_skill_limit,
        )
        recommendations.append(
            Domain(
                domain_code=domain,
                occupation=Occupation(
                    id=occ_id,
                    esco_uri=meta.get("esco_uri", ""),
                    label=meta.get("preferred_label", ""),
                    code=meta.get("code", ""),
                    isco_code=meta.get("isco_code", ""),
                    score=score,
                ),
                extra_skills=extra_skills,
            )
        )
    return recommendations


def build_occupation_suggestions(
        occupation_scores: Dict[str, float],
        indexes: Dict[str, Dict],
        extra_skill_limit: int,
) -> List[OccupationResponse]:
    occ_by_id = get_meta_by_id(indexes["occupation"]["metadata"])
    suggestions: List[OccupationResponse] = []
    for occ_id, score in sorted(occupation_scores.items(), key=lambda i: i[1], reverse=True):
        meta = occ_by_id.get(occ_id)
        if not meta:
            continue
        idx = indexes['occupation']['id_to_idx'].get(occ_id)
        occ_vec = np.array(indexes['occupation']['index'].reconstruct(idx))
        extra_skills = expand_skills(occ_id=occ_id, input_encoded=occ_vec, limit=extra_skill_limit)
        suggestions.append(
            OccupationResponse(
                occupation=Occupation(
                    id=occ_id,
                    esco_uri=meta.get("esco_uri", ""),
                    label=meta.get("preferred_label", ""),
                    code=meta.get("code", ""),
                    isco_code=meta.get("isco_code", ""),
                    score=score,
                ),
                extra_skills=extra_skills,
            )
        )
    return suggestions
