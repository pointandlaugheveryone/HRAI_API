from datamodels import Entity
from settings import config
import os, json, pickle
from typing import Dict, List, Tuple
from functools import lru_cache
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


@lru_cache(maxsize=1)
def load_data():
    db_dir = os.path.join(config.data_dir, "db")
    pickle_dir = os.path.join(config.data_dir, "pickle")
    db = {}
    names = ["skill", "occupation", "skill_group", "isco_group", "all"]
    for name in names:
        idx = faiss.read_index(os.path.join(db_dir, f"{name}.index"))
        with open(os.path.join(db_dir, f"{name}_metadata.json"), "r", encoding="utf-8") as f:
            metadata = json.load(f)
        with open(os.path.join(pickle_dir, f"{name}_id-idx.pkl"), "rb") as f:
            id_idx = pickle.load(f)

        db[name] = {
            "index": idx,
            "metadata": metadata,
            "id_to_idx": id_idx,
        }
    return db


def load_relations(
        occ_to_s_path: str = os.path.join(config.data_dir, 'occ_to_skill.json'),
        s_to_occ_path: str = os.path.join(config.data_dir, 'skill_to_occ.json'),
):
    with open(occ_to_s_path,'r', encoding="utf-8") as o:
        occupation_to_skills = json.loads(o.read())
    with open(s_to_occ_path, 'r', encoding="utf-8") as s:
        skills_to_occupations = json.loads(s.read())

    return occupation_to_skills, skills_to_occupations


def query(index, metadata, query_vector: np.float32, top_n):
    scores, indices = index.search(query_vector, top_n)
    results = []
    for score_row, idx_row in zip(scores, indices):
        row_results = []
        for score, idx in zip(score_row, idx_row):
            if idx == -1:
                continue
            row_results.append((metadata[int(idx)], float(score)))
        results.append(row_results)
    return results


def get_meta_by_id(meta_dict):
    return {item.get("id"): item for item in meta_dict}

def match_entities(
         entities: List[Entity],
         encoder: SentenceTransformer,
         top_n: int = 5,
 ) -> Dict[str, float]:
    indexes = load_data()
    if not entities:
        return {}

    texts = [e.text for e in entities]
    texts_encoded = encoder.encode(
        texts,
        normalize_embeddings=config.normalize_embeddings,
        convert_to_numpy=True,
    )

    scores: Dict[str, float] = {}
    for entity, vector in zip(entities, texts_encoded):
        target_index = entity.entity_type if entity.entity_type in indexes else "all"
        matches = query(
            index=indexes[target_index]["index"],
            metadata=indexes[target_index]["metadata"],
            query_vector=np.expand_dims(vector, axis=0),
            top_n=top_n,
        )
        for data, score in matches[0]:
            id = data.get("id")
            scores[id] = max(scores.get(id, 0.0), score)
    return scores


def match_occupations(texts: List[str], encoder, top_n: int) -> List[Tuple[str, float]]:
    matched: List[Tuple[str, float]] = []
    for text in texts:
        scores = match_entities([Entity(text=text, entity_type='occupation')], encoder, top_n=top_n)
        if not scores:
            continue
        occ_id, score = sorted(scores.items(), key=lambda i: i[1], reverse=True)[0]
        matched.append((occ_id, score))
    return matched