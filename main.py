from __future__ import annotations

from ranking import expand_skills
from datamodels import Config, DomainInfo, Entity, OccupationInfo, LookupRequest, Skill
from lookups import load_data, load_relations, metadata_get_by_id, match_entities
from typing import Dict, List
import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sentence_transformers import SentenceTransformer
from sentence_transformers.models import Transformer, Pooling
load_dotenv()
config = Config()

def init_encoder():
    word_embeddings = Transformer(
        model_name_or_path=config.model_name,
        max_seq_length=256,
    )
    pooling = Pooling(
        word_embeddings.get_word_embedding_dimension(),
        pooling_mode_cls_token=True,
    )
    return SentenceTransformer(modules=[word_embeddings, pooling])

app = FastAPI(title="HRAI demo", version="1.0.0")
app.add_middleware(CORSMiddleware)

def score_all(
        entities: List[Entity],
        encoder,
        top_n
) -> Dict[str, float]:
    scores = match_entities(entities, encoder, top_n) # TODO add lookup EntityType param
    occupation_to_skills, skill_to_occupations = load_relations()

    for skill_id, score in scores.items():
        for occ_id, relation_type in skill_to_occupations.get(skill_id, []):
            weight = 1.0 if relation_type.lower() == "essential" else 0.7
            scores[occ_id] = scores.get(occ_id, 0.0) + score * weight

    return scores


@app.post("/post", response_model=List[DomainInfo])
def recommend_occupations(request: LookupRequest):
    encoder = init_encoder()
    occupation_scores = score_all(
        request_entities=request.entities,
        encoder=encoder,
        top_k=request.top_k_per_input,
    )

    if not occupation_scores:
        return []

    index_data = load_data()
    occ_meta = index_data["occupation"]["metadata"]
    occ_by_id = metadata_get_by_id(occ_meta)

    best_by_domain = pick_best_per_domain(occupation_scores)

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
            limit=request.extra_skill_limit or config.extra_skills_n,
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
