import numpy as np
import numpy.typing as npt
from typing import List, Tuple
from config import config
from pos_extraction import text_to_ngrams
from models.EntityResult import EntityResult
from setup import get_database, get_encoder


def _query_resume(text: str) -> List[Tuple[float, int]]:
    """
    Embed extracted ngrams, search the global FAISS index, and return best entity matches.
    """
    model = get_encoder()
    ngrams: List[str] = text_to_ngrams(text)
    best_for_match: dict[int, Tuple[float, int]] = {} # when more entities score as the same esco record

    db = get_database('all')
    for ent in ngrams:
        query_vector = model.encode(
            [ent],
            normalize_embeddings=True, # accurate cosine similarity
            convert_to_numpy=True
        )

        scores: npt.NDArray[np.float32]
        indices: npt.NDArray[np.int64]
        results: List[Tuple[float, int]] = []
        scores, indices = db['index'].search(query_vector, k=1)
        for score, idx in zip(scores[0], indices[0]): # this is not actually a loop (1 iteration) but numpy reports idx cant be converted to int otherwise
            i = int(idx)
            if config.match_cutoff > score or i == -1: continue
            results.append((score,i))

            # Keep only the strongest hit per metadata record while respecting the global cap
            if ((i not in best_for_match and len(best_for_match.values()) < config.max_ents) or
                    (i in best_for_match and score > best_for_match[i][0])):
                best_for_match[i] = (float(score), i)

    return sorted(best_for_match.values(), key=lambda x: x[0], reverse=True)


# noinspection DuplicatedCode
def extract_from_resume(text: str) -> List[EntityResult]:
    """
    extract all entities from input parsed from a pdf file
    """
    results: List[EntityResult] = []
    values = _query_resume(text)
    for score, id in values:
        meta = get_database('all')['metadata'][id]
        results.append(EntityResult(
            id=id,
            cosine_score=score,
            entity_type=meta.get('entity_type',''),
            esco_uri=meta.get('esco_uri',''),
            label=meta.get('preferred_label',''),
            code=meta.get('code',''),
            isco_code=meta.get('isco_code',''),
            description=meta.get('description','')
        ))
    return results


# noinspection DuplicatedCode
def query_type(
        ents: List[str],
        label: str,
        search_k=config.result_n
) -> List[EntityResult]:
    """
     # search label-specific index file for those labels
    """

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
            if config.match_cutoff > score or i == -1:
                continue
            # Same deduplication rule as resume queries so downstream scoring stays consistent
            if ((i not in best_for_match and len(best_for_match) < config.max_ents) or
                    (i in best_for_match and score > best_for_match[i][0])):
                best_for_match[i] = (float(score), i)

    for score, id in sorted(best_for_match.values(), key=lambda x: x[0], reverse=True):
        meta = db['metadata'][id]
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