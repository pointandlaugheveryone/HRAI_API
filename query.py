from typing import List, Dict, Literal, Optional

from config import conf
from pos_extraction import text_to_ngrams
from models.EntityResult import EntityResult

from sentence_transformers import SentenceTransformer
import numpy as np
import numpy.typing as npt

from load import get_encoder
import faiss, os, json


# noinspection DuplicatedCode
def extract_from_resume(
        db,
        metadata: Dict[str:Dict],
        model: SentenceTransformer,
        text: str,
        min_score=conf.match_cutoff
) -> List[EntityResult]:


    results: List[EntityResult] = []
    ent_scores = []

    ngrams: List[str] = text_to_ngrams(text)

    for ent in ngrams:
        query_vector = model.encode(
            [ent],
            normalize_embeddings=True,
            convert_to_numpy=True
        )

        scores: npt.NDArray[np.float32]
        indices: npt.NDArray[np.int64]
        scores, indices = db.search(query_vector, k=1)

        # this is not actually a loop (iterates once)
        # but numpy reports idx cant be converted to int otherwise
        for score,idx in zip(scores[0], indices[0]):
            if min_score> score or idx == -1: continue
            id = int(idx)
            ent_scores.append((str(id), float(score)))

    for id, score in ent_scores:
        id_str = str(id)
        meta = metadata.get(id_str, {})
        results.append(EntityResult(
            id=id_str,
            cosine_score=score,
            entity_type=meta.get('entity_type', ''),
            esco_uri=meta.get('esco_uri', ''),
            label=meta.get('preferred_label', ''),
            code=meta.get('code', ''),
            description=meta.get('description', '')
        ))
    return results


# noinspection DuplicatedCode
def query_type(
        db,
        metadata: Dict[str:Dict],
        model: SentenceTransformer,
        ents: Optional[List[str]],
        ent_type: Literal['occupation', 'skill','isco_group','skill_group'],
        min_score=conf.match_cutoff,
) -> List[EntityResult]:

    if not ents: return []

    results: List[EntityResult] = []
    ent_scores = []
    other_results:List[EntityResult] = []
    for ent in ents:
        query_vector = model.encode(
            [ent],
            normalize_embeddings=True,
            convert_to_numpy=True
        )

        scores: npt.NDArray[np.float32]
        indices: npt.NDArray[np.int64]
        scores, indices = db.search(query_vector,k=10)

        for score, idx in zip(scores[0], indices[0]):
            if min_score > score or idx == -1: continue
            id = int(idx)
            ent_scores.append((str(id), float(score)))

    for id, score in ent_scores:
        id_str = str(id)
        meta = metadata.get(id_str, {})

        res = EntityResult(
            id=id_str,
            cosine_score=score,
            entity_type=meta.get('entity_type', ''),
            esco_uri=meta.get('esco_uri', ''),
            label=meta.get('preferred_label', ''),
            code=meta.get('code', ''),
            description=meta.get('description', '')
        )
        if ent_type != meta.get('entity_type', ''):
            other_results.append(res)
            continue
        results.append(res)

    return results if len(results) > 0 else other_results