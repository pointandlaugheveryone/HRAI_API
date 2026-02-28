from _models.Entity import Entity
from _models.Domain import Domain
from _models.RequestModel import RequestModel
from _models.Lookup import LookupRequest, LookupResponse, ResumeLookup
from _lookups import load_data, load_relations, match_any
from _ranking import skills_to_score
from config import config
from extraction import match_skill_texts

from typing import List
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sentence_transformers import SentenceTransformer


load_dotenv()
app = FastAPI(title="HRAI demo")
app.add_middleware(CORSMiddleware)
encoder = SentenceTransformer(config.model_name)


@app.post("/resume", response_model=RequestModel)
def doc_post(request: LookupRequest) -> List[Domain]:
    # text extracted from a document -> needs to be parsed for entities
    
    occupation_scores = match_any(request.entities, encoder, request.top_n)

    if not occupation_scores:
        return []

    index_data = load_data()
    input_vectors = encoder.encode(
        [e.text for e in request.entities],
        normalize_embeddings=config.normalize_embeddings,
        convert_to_numpy=True,
        show_progress_bar=False,
    )

    return build_domain_recommendations(
        occ_scores=occupation_scores,
        indexes=index_data,
        extra_skill_limit=request.extra_skill_limit,
        reference_vectors=input_vectors,
    )


@app.post("/text", response_model=LookupResponse)
def text_post(payload: ResumeLookup) -> LookupResponse:
    # user inputs skills manually
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