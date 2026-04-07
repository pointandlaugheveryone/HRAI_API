import os, json
from config import conf
from sentence_transformers import models, SentenceTransformer


def get_relations():
    occ2s_path = os.path.join(conf.data_dir, 'occ_to_skill.json')
    s2occ_path = os.path.join(conf.data_dir, 'skill_to_occ.json')
    with open(occ2s_path, 'r', encoding="utf-8") as o: occ2s = json.loads(o.read())
    with open(s2occ_path, 'r', encoding="utf-8") as s: s2occ = json.loads(s.read())
    return occ2s, s2occ


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