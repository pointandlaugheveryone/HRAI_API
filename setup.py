import os, pickle, json

from config import config
from functools import lru_cache

import faiss
from sentence_transformers import models, SentenceTransformer


@lru_cache(maxsize=1)
def get_database(entity_type):
    db = {}
    idx = faiss.read_index(os.path.join(config.db_dir, f"{entity_type}.index"))
    with open(os.path.join(config.meta_dir, f"{entity_type}.json"), "r", encoding="utf-8") as f:
        metadata = json.load(f)
    with open(os.path.join(config.pickle_dir, f"{entity_type}.pkl"), "rb") as f:
        id_idx = pickle.load(f)

    return {
        "index": idx,
        "metadata": metadata,
        "id_to_index": id_idx,
    }


@lru_cache(maxsize=1)
def get_relations():
    occ2s_path = os.path.join(config.data_dir, 'occ_to_skill.json')
    s2occ_path = os.path.join(config.data_dir, 'skill_to_occ.json')
    with open(occ2s_path, 'r', encoding="utf-8") as o: occ2s = json.loads(o.read())
    with open(s2occ_path, 'r', encoding="utf-8") as s: s2occ = json.loads(s.read())
    return occ2s, s2occ

@lru_cache(maxsize=1)
def get_encoder():
    word_embedding = models.Transformer(
        model_name_or_path=config.model_name,
        max_seq_length=128,
    )

    pooling = models.Pooling(
        word_embedding.get_word_embedding_dimension(),
        pooling_mode_cls_token=True
    )

    return SentenceTransformer(modules=[word_embedding, pooling])