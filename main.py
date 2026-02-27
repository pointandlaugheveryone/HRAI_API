from functools import lru_cache
from typing import List
import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sentence_transformers import SentenceTransformer
from sentence_transformers.models import Transformer, Pooling

from datamodels import DomainInfo, LookupRequest, LookupResponse, OccupationInfo, OccupationSuggestion, ResumeLookup
from lookups import load_data, load_relations, match_entities, get_meta_by_id, match_occupations
from ranking import expand_skills, best_occ_per_domain, skills_to_score
from settings import config
from extraction import extract_skill_ents, match_skill_texts

app = FastAPI(title="HRAI demo")
app.add_middleware(CORSMiddleware)


@lru_cache(maxsize=1)
def get_encoder():
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


@app.post("/post", response_model=List[DomainInfo])
def get_near_occupations(request: LookupRequest):
    encoder = get_encoder()
    occupation_scores = match_entities(request.entities, encoder, request.top_n)

    if not occupation_scores: return []

    index_data = load_data()
    occ_meta = index_data["occupation"]["metadata"]
    occ_by_id = get_meta_by_id(occ_meta)

    best_by_domain = best_occ_per_domain(occupation_scores)

    input_vectors = encoder.encode(
        [e.text for e in request.entities],
        normalize_embeddings=config.normalize_embeddings,
        convert_to_numpy=True,
    )

    recommendations: List[DomainInfo] = []
    for domain, (occ_id, score) in sorted(best_by_domain.items(), key=lambda i: i[0]):
        meta = occ_by_id.get(occ_id)
        if not meta:
            continue
        extra_skills = expand_skills(
            occ_id=occ_id,
            input_encoded=input_vectors,
            limit=request.extra_skill_limit,
        )
        recommendations.append(
            DomainInfo(
                domain_code=domain,
                occupation=OccupationInfo(
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


@app.post("/resume", response_model=LookupResponse)
def resume_lookup(payload: ResumeLookup):
    encoder = get_encoder()
    indexes = load_data()
    occupation_to_skills, skill_to_occupations = load_relations()
    occ_by_id = get_meta_by_id(indexes["occupation"]["metadata"])

    manual_skills = payload.skills.split() if payload.skills else []
    target_job = [payload.job_title] if payload.job_title else []

    if manual_skills:
        extracted_skills = match_skill_texts(
            manual_skills,
            encoder,
            top_n=payload.top_n,
            source="manual")

    else:
        extracted_skills = extract_skill_ents(
            payload.resume_text,
            encoder,
            use_spacy=payload.use_spacy,
            top_n=payload.top_n)

    skill_vectors = []
    for skill_id in [s.id for s in extracted_skills]:
        idx = indexes['skill']['id_to_idx'].get(skill_id)
        vec = np.array(indexes['skill']['index'].reconstruct(idx))
        skill_vectors.append(vec)

    provided_occ_matches = match_occupations(target_job, encoder, top_n=1)

    occ_scores = skills_to_score(extracted_skills, skill_to_occupations)
    for occ_id, score in provided_occ_matches:
        occ_scores[occ_id] = max(occ_scores.get(occ_id, 0.0), score)

    best_by_domain = best_occ_per_domain(occ_scores)

    recommendations: List[DomainInfo] = []
    for domain, (occ_id, score) in sorted(best_by_domain.items(), key=lambda i: i[0]):
        meta = occ_by_id.get(occ_id)
        idx = indexes['occupation']['id_to_idx'].get(occ_id)
        occ_vec = np.array(indexes['occupation']['index'].reconstruct(idx))
        extra_skills = expand_skills(occ_id=occ_id, input_encoded=occ_vec)
        recommendations.append(
            DomainInfo(
                domain_code=domain,
                occupation=OccupationInfo(
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

    occupation_suggestions: List[OccupationSuggestion] = []
    for occ_id, score in provided_occ_matches:
        meta = occ_by_id.get(occ_id)
        if not meta: continue
        idx = indexes['occupation']['id_to_idx'].get(occ_id)
        occ_vec = np.array(indexes['occupation']['index'].reconstruct(idx))
        extra_skills = expand_skills(occ_id=occ_id, input_encoded=occ_vec)
        occupation_suggestions.append(
            OccupationSuggestion(
                occupation=OccupationInfo(
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

    return LookupResponse(
        extracted_skills=extracted_skills,
        domain_recommendations=recommendations,
        occupation_suggestions=occupation_suggestions,
    )
