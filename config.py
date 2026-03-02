import os
from dataclasses import dataclass

@dataclass
class Config:
    base_dir: str = os.getcwd()
    data_dir: str = os.path.join(base_dir, 'data')
    db_dir: str = os.path.join(data_dir, 'db')

    model_name: str = 'Seznam/simcse-dist-mpnet-paracrawl-cs-en'
    tagger_name: str = '_czech_pdt_2.5.udpipe'

    max_ngram: int = 3                  # nspan range
    max_ents: int = 40                  # maximum result entities found in input by score, can be set higher with better hardware
    min_skills: int = 5                 # minimum skills for an occupation to be detected from user input

    ### embedding configuration
    batch_size: int = 256               # how many text spans to embed per encoder call
    normalize_embeddings: bool = True   # enables cosine similarity accuracy
    match_cutoff: float = 0.65          # filters vector search results by similarity
    result_n = 5                        # how many entities to search for when querying a manually input string

    hf_token: str = os.getenv('HF_TOKEN', '')

conf = Config()