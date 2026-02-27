from datamodels import Entity
from main import config
import os, json, pickle
from typing import Dict, List
from functools import lru_cache # store {arg: values} for function -> quicker index lookups and loading
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


@lru_cache(maxsize=1)
def load_data():
    skill_index = faiss.read_index(os.path.join(config.data_dir, "db/skill.index"))
    occ_index = faiss.read_index(os.path.join(config.data_dir, "db/occupation.index"))

    with open(os.path.join(config.data_dir, "db/skill_metadata.json"), "r", encoding="utf-8") as f:
        skill_meta = json.load(f)
    with open(os.path.join(config.data_dir, "db/occupation_metadata.json"), "r", encoding="utf-8") as f:
        occ_meta = json.load(f)

    with open(os.path.join(config.data_dir, "pickle/skill_id-idx.pkl"), "rb") as f:
        skill_id_idx = pickle.load(f)
    with open(os.path.join(config.data_dir, "pickle/occupation_id-idx.pkl"), "rb") as f:
        occ_id_idx = pickle.load(f)

    idx_map = {
        "skill": {
            "index": skill_index,
            "metadata": skill_meta,
            "id_to_idx": skill_id_idx,
        },
        "occupation": {
            "index": occ_index,
            "metadata": occ_meta,
            "id_to_idx": occ_id_idx,
        },
    }
    return idx_map


def load_relations(
        occ_to_s_path = os.path.join(config.data_dir, 'skill_to_occ.json'),
        s_to_occ_path = os.path.join(config.data_dir, 'skill_to_occ.json')
):
    with open(occ_to_s_path,'r') as o:
        occupation_to_skills = json.loads(o.read())
    with open(s_to_occ_path, 'r') as s:
        skills_to_occupations = json.loads(s.read())

    return occupation_to_skills, skills_to_occupations


def query(index, metadata, query_vector: np.float32, top_n):
    scores, indices = index.search(query_vector, top_n)
    results = []
    for row, idx_row in zip(scores, indices):
        row = []
        for score, idx in zip(row, idx_row):
            if idx == -1: continue
            row.append((metadata[int(idx)], float(score)))
        results.append(row)
    return results


def metadata_get_by_id(meta_dict):
    return {item.get("id"): item for item in meta_dict}

def match_entities(
        entities: List[Entity],
        encoder: SentenceTransformer,
        top_n: int = 5,
) -> Dict[str, float]:
    indexes = load_data()
    texts = [e.text for e in entities]
    if not texts: return {}

    texts_encoded = encoder.encode(
        texts,
        normalize_embeddings=config.normalize_embeddings,
        convert_to_numpy=True,
    )
    results = query(index=indexes["all"]["index"],
                    metadata=indexes["all"]["metadata"],
                    query_vector=texts_encoded, top_n=top_n)

    scores: Dict[str, float] = {}
    for row in results:
        for data, score in row:
            id = data.get("id")
            scores[id] = max(scores.get(id, 0.0),
                             score)
    return scores