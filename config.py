import os
from dataclasses import dataclass

@dataclass
class Config:
    base_dir = os.getcwd()
    data_dir = os.path.join(base_dir, 'data')

    db_dir = os.path.join(data_dir, 'db')
    pickle_dir = os.path.join(data_dir, 'id_to_index')
    meta_dir = os.path.join(data_dir, 'metadata')

    model_name = 'Seznam/simcse-dist-mpnet-paracrawl-cs-en'

    tagger_name = 'czech-pdt-2.5.udpipe'
    max_ngram = 3
    entity_cutoff = 0.65 # vector that at least contextually falls under found ESCO value
    max_ents = 40 # maximum deduplicated found esco skills

    # embedding configuration
    batch_size = 256
    normalize_embeddings = True
    match_cutoff = 0.7
    result_n = 10
    hf_token: str = os.getenv('HF_TOKEN','')

config = Config()
