import numpy as np
import numpy.typing as npt
from typing import List
from config import config
from extraction import text_to_ngrams
from models.Result import Result
from setup import get_database, get_encoder


def _query_resume(text: str):
    model = get_encoder()
    ngrams = text_to_ngrams(text)
    best_for_match = {} # when more entities score as the same esco record

    db = get_database('all')
    for ent in ngrams:
        query_vector = model.encode(
            [ent],
            normalize_embeddings=True, # accurate cosine similarity
            convert_to_numpy=True
        )

        scores: npt.NDArray[np.float32]
        indices: npt.NDArray[np.int64]
        results = []
        scores, indices = db['index'].search(query_vector, k=1)
        for score, idx in zip(scores[0], indices[0]): # this is not actually a loop (1 iteration) but numpy reports idx cant be converted to int otherwise
            i = int(idx)
            if config.match_cutoff > score or i == -1: continue
            results.append((score,i))

            if ((i not in best_for_match and len(best_for_match.values()) < config.max_ents) or
                    (i in best_for_match and score > best_for_match[i][0])):
                best_for_match[i] = (float(score), i)

    return sorted(best_for_match.values(), key=lambda x: x[0], reverse=True)


def extract_from_resume(text):
    results = []
    values = _query_resume(text)
    for score, id in values:
        meta = get_database('all')['metadata'][id]
        results.append(Result(
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


def query_type(ents: List[str], label: str):
    # TODO
    return



