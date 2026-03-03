from typing import List, Dict

from config import conf
from pos_extraction import text_to_ngrams

from models.EntityResult import EntityResult
from load import get_database, get_encoder, get_metadata

from sentence_transformers import SentenceTransformer
import numpy as np
import numpy.typing as npt


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
            ent_scores.append((str(id), int(score)))

    for id, score in ent_scores:
        meta = metadata.get(str(id),'')
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


# noinspection DuplicatedCode
def query_type(
        db,
        metadata: Dict[str:Dict],
        model: SentenceTransformer,
        ents: List[str],
        label: str,
        min_score=conf.match_cutoff,
) -> List[EntityResult]:

    results: List[EntityResult] = []
    ent_scores = []

    for ent in ents:
        query_vector = model.encode(
            [ent],
            normalize_embeddings=True,
            convert_to_numpy=True
        )

        scores: npt.NDArray[np.float32]
        indices: npt.NDArray[np.int64]
        scores, indices = db.search(query_vector, k=1)

        for score, idx in zip(scores[0], indices[0]):
            if min_score > score or idx == -1: continue
            id = int(idx)
            ent_scores.append((str(id), int(score)))

    for id, score in ent_scores:
        meta = metadata.get(str(id), '')
        results.append(EntityResult(
            id=id,
            cosine_score=score,
            entity_type=meta.get('entity_type', ''),
            esco_uri=meta.get('esco_uri', ''),
            label=label,
            code=meta.get('code', ''),
            isco_code=meta.get('isco_code', ''),
            description=meta.get('description', '')
        ))

    return results