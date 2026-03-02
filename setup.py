import os, json
from functools import lru_cache
from pathlib import Path

from config import conf

import faiss
from sentence_transformers import models, SentenceTransformer


@lru_cache(maxsize=5)
def get_database(entity_type):
    idx = faiss.read_index(os.path.join(conf.db_dir, f"{entity_type}.index"))
    return idx

@lru_cache(maxsize=1)
def get_metadata():
    with open(os.path.join(conf.data_dir, f"key_to_ent.json"), "r", encoding="utf-8") as f:
        return json.load(f)


@lru_cache(maxsize=1)
def _get_id_ix_map():
    return json.loads(
        Path(
            os.path.join(conf.data_dir, 'id_idx.json')
        ).read_text(encoding='utf-8')
    )


@lru_cache(maxsize=1)
def _get_ix_id_map():
    return json.loads(
        Path(
            os.path.join(conf.data_dir, 'idx_id.json')
        ).read_text(encoding='utf-8')
    )


def id2idx(id: str):
    return _get_id_ix_map()[id]


def idx2id(idx: int):
    return _get_ix_id_map()[idx]


@lru_cache(maxsize=1)
def get_relations():
    occ2s_path = os.path.join(conf.data_dir, 'occ_to_skill.json')
    s2occ_path = os.path.join(conf.data_dir, 'skill_to_occ.json')
    with open(occ2s_path, 'r', encoding="utf-8") as o: occ2s = json.loads(o.read())
    with open(s2occ_path, 'r', encoding="utf-8") as s: s2occ = json.loads(s.read())
    return occ2s, s2occ


@lru_cache(maxsize=1)
def get_encoder():
    word_embedding = models.Transformer(
        model_name_or_path=conf.model_name,
        max_seq_length=128,
    )

    pooling = models.Pooling(
        word_embedding.get_word_embedding_dimension(),
        pooling_mode_cls_token=True
    )

    return SentenceTransformer(modules=[word_embedding, pooling])