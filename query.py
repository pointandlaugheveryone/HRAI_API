from functools import lru_cache

import numpy as np
import numpy.typing as npt
from typing import List, Tuple
from config import conf
from pos_extraction import text_to_ngrams
from models.EntityResult import EntityResult
from setup import get_database, get_encoder, get_metadata, idx2id


def _query_resume(text: str) -> List[Tuple[float, int]]:
    model = get_encoder()
    ngrams: List[str] = text_to_ngrams(text)
    best_for_match: dict[int, Tuple[float, int]] = {} # when more entities score as the same esco record

    db = get_database('all')
    for ent in ngrams:
        query_vector = model.encode(
            [ent],
            normalize_embeddings=True,
            convert_to_numpy=True
        )

        scores: npt.NDArray[np.float32]
        indices: npt.NDArray[np.int64]
        results: List[Tuple[float, int]] = []
        scores, indices = db['index'].search(query_vector, k=1)

        # this is not actually a loop (iterates once)
        # but numpy reports idx cant be converted to int otherwise
        for score, faiss_id in zip(scores[0], indices[0]):
            i = int(faiss_id)
            if conf.match_cutoff > score or i == -1: continue
            results.append((score,i))

            if ((i not in best_for_match and len(best_for_match.values()) < conf.max_ents) or   # add new esco ents by default
                    (i in best_for_match and score > best_for_match[i][0])):                    # or add if is better match than existing match for this entity
                best_for_match[i] = (float(score), i)

    return sorted(best_for_match.values(), key=lambda x: x[0], reverse=True)


# noinspection DuplicatedCode
@lru_cache(maxsize=5) # makes testing with same files quicker
def extract_from_resume(text: str) -> List[EntityResult]:
    results: List[EntityResult] = []
    values = _query_resume(text)

    for score, faiss_id in values:
        keyid = idx2id(faiss_id)
        meta = get_metadata(keyid)

        results.append(EntityResult(
            id=keyid,
            cosine_score=score,
            entity_type=meta.get('entity_type',''),
            esco_uri=meta.get('esco_uri',''),
            label=meta.get('preferred_label',''),
            code=meta.get('code',''),
            isco_code=meta.get('isco_code',''),
            description=meta.get('description',''),
        ))
    return results


# noinspection DuplicatedCode
def query_type(
        ents: List[str],
        label: str = 'all',
        search_k=conf.result_n
) -> List[EntityResult]:

    db = get_database(label)
    model = get_encoder()
    results: List[EntityResult] = []
    best_for_match: dict[int, Tuple[float, int]] = {}

    for ent in ents:
        query_vector = model.encode(
            [ent],
            normalize_embeddings=True,
            convert_to_numpy=True
        )
        scores: npt.NDArray[np.float32]
        indices: npt.NDArray[np.int64]
        scores, indices = db['index'].search(query_vector, k=search_k)

        for score, idx in zip(scores[0], indices[0]):
            i = int(idx)
            if conf.match_cutoff > score or i == -1:
                continue
            if i not in best_for_match or score > best_for_match[i][0]:
                best_for_match[i] = (float(score), i)

    for score, id in sorted(best_for_match.values(), key=lambda x: x[0], reverse=True):
        meta = idx2id(id)
        results.append(EntityResult(
            id=id,
            cosine_score=score,
            entity_type=meta.get('entity_type', ''),
            esco_uri=meta.get('esco_uri', ''),
            label=meta.get('preferred_label', ''),
            code=meta.get('code', ''),
            isco_code=meta.get('isco_code', ''),
            description=meta.get('description', '')
        ))

    return results