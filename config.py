import os
from dataclasses import dataclass

@dataclass
class Config:
    base_dir: str = os.getcwd()
    data_dir: str = os.path.join(base_dir, 'data')

    db_dir: str = os.path.join(data_dir, 'db')
    pickle_dir: str = os.path.join(data_dir, 'id_to_index')
    meta_dir: str = os.path.join(data_dir, 'metadata')

    model_name: str = 'Seznam/simcse-dist-mpnet-paracrawl-cs-en'

    tagger_name: str = 'czech-pdt-2.5.udpipe'
    max_ngram: int = 3 # nspan range
    max_ents: int = 40 # maximum result entities found in input by score, can be set higher with better hardware

    ### embedding configuration
    batch_size: int = 256 # how many text spans to embed per encoder call
    normalize_embeddings: bool = True # enables cosine similarity accuracy
    match_cutoff: float = 0.7   # filters vector search results by similarity
                                # 0.7 basically at least returns some skill entity
    hf_token: str = os.getenv('HF_TOKEN', '')

config = Config()