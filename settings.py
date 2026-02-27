from dotenv import load_dotenv
import os
from typing import Optional
from pydantic import BaseModel


class Config(BaseModel):
    base_dir: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir: str = os.path.join(base_dir, 'db')
    db_dir: str = os.path.join(data_dir, 'db')
    pickle_dir: str = os.path.join(data_dir, 'pickle')

    model_name: str = 'Seznam/simcse-dist-mpnet-paracrawl-cs-en'
    normalize_embeddings: bool = True
    match_cutoff: float = 0.7  # vector that at least contextually falls under found ESCO value
    top_result_n: int = 10
    extra_skills_n: int = 25
    hf_token: Optional[str] = os.getenv('HF_TOKEN')

load_dotenv()
config = Config()
