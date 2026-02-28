from _models import Entity
from config import config

import os, json, pickle
from typing import Any, Dict, List, Tuple
from functools import lru_cache

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


def get_meta_by_id(metadata: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {item.get("id"): item for item in metadata}


def query(
        index,
        metadata,
        query_vector: np.float32,
        top_n: int = config.result_n
) -> List[List[Tuple[Dict[str, Any], float]]]:
    scores, indices = index.search(query_vector, top_n)
    results = []
    for score_row, idx_row in zip(scores, indices):
        row_results = []
        for score, idx in zip(score_row, idx_row):
            if idx == -1: continue # faiss uses for not found/wrong value
            row_results.append((metadata[int(idx)], float(score)))
        results.append(row_results)
    return results


def match_any(entities: List[Entity], encoder: SentenceTransformer, top_n: int = 5) -> Dict[str, float]:
    # returns found entities with cosine scores
    indexes = load_data()
    if not entities: return {}

    texts = [e.text for e in entities]
    texts_encoded = encoder.encode(
        texts,
        normalize_embeddings=config.normalize_embeddings,
        convert_to_numpy=True,
        show_progress_bar=False,
    )

    scores = {}
    for entity, vector in zip(entities, texts_encoded):
        target_index = entity.entity_type
        matches = query(
            index=indexes[target_index]["index"],
            metadata=indexes[target_index]["metadata"],
            query_vector=np.expand_dims(vector, axis=0),
            top_n=top_n
        )
        for data, score in matches[0]:
            id = data.get("id")
            # when same or even different ngrams match the same ESCO record - keep the more accurate one
            scores[id] = max(scores.get(id, 0.0), score)

    return scores


def match_occupations(texts: List[str], encoder: SentenceTransformer, top_n: int) -> List[Tuple[str, float]]:
    matched: List[Tuple[str, float]] = []
    for text in texts:
        entities = [Entity(text=text, entity_type='occupation')]
        scores = match_any(entities, encoder, top_n=top_n)
        if not scores:
            continue
        occ_id, score = sorted(scores.items(), key=lambda i: i[1], reverse=True)[0]
        matched.append((occ_id, score))
    return matched