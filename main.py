from models import Domain, Entity, LookupResponse, Occupation, OccupationResponse, ResumeLookup, LookupRequest
from lookups import load_data, load_relations, match_any, get_meta_by_id
from ranking import expand_skills, best_occ_per_domain, skills_to_score
from settings import config
from extraction import match_skill_texts

from functools import lru_cache
from typing import Dict, List

import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sentence_transformers import SentenceTransformer
from sentence_transformers.models import Transformer, Pooling

app = FastAPI(title="HRAI demo")
app.add_middleware(CORSMiddleware)


@lru_cache(maxsize=1)
def get_encoder() -> SentenceTransformer:
    # build the huggingface model for embeddings, since its not provided by huggingface config.json
    word_embeddings = Transformer(
        model_name_or_path=config.model_name,
        max_seq_length=256,
    )
    pooling = Pooling(
        word_embeddings.get_word_embedding_dimension(),
        pooling_mode_cls_token=True,
    )
    return SentenceTransformer(modules=[word_embeddings, pooling])


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


@app.post("/post", response_model=List[Domain])
def get_near_occupations(request: LookupRequest) -> List[Domain]:
    encoder = get_encoder()
    occupation_scores = match_any(request.entities, encoder, request.top_n)

    if not occupation_scores:
        return []

    index_data = load_data()
    input_vectors = encoder.encode(
        [e.text for e in request.entities],
        normalize_embeddings=config.normalize_embeddings,
        convert_to_numpy=True,
    )

    return build_domain_recommendations(
        occ_scores=occupation_scores,
        indexes=index_data,
        extra_skill_limit=request.extra_skill_limit,
        reference_vectors=input_vectors,
    )


@app.post("/resume", response_model=LookupResponse)
def resume_lookup(payload: ResumeLookup) -> LookupResponse:
    encoder = get_encoder()
    indexes = load_data()
    _, skill_to_occupations = load_relations()

    skill_source_text = payload.skills or payload.resume_text or ""
    extracted_skills = match_skill_texts(
        skill_source_text,
        encoder,
        top_n=payload.top_n)

    occupation_entities = [Entity(text=payload.job_title, entity_type='occupation')] if payload.job_title else []
    occupation_match = match_any(occupation_entities, encoder, payload.top_n) if occupation_entities else {}

    occ_scores = skills_to_score(extracted_skills, skill_to_occupations)
    for occ_id, score in occupation_match.items():
        occ_scores[occ_id] = max(occ_scores.get(occ_id, 0.0), score)

    recommendations = build_domain_recommendations(
        occ_scores=occ_scores,
        indexes=indexes,
        extra_skill_limit=config.extra_skills_n,
        reference_vectors=None,
    )

    occupation_suggestions = build_occupation_suggestions(
        occupation_scores=occupation_match,
        indexes=indexes,
        extra_skill_limit=config.extra_skills_n,
    )

    return LookupResponse(
        extracted_skills=extracted_skills,
        domain_recommendations=recommendations,
        occupation_suggestions=occupation_suggestions,
    )