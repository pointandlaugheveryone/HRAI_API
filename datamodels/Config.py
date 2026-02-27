import os
from pydantic import BaseModel, model_validator
from typing import Optional


class Config(BaseModel):
    base_dir: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir: str = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
    db_dir: str = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "db")
    model_name: str = 'Seznam/simcse-dist-mpnet-paracrawl-cs-en'
    normalize_embeddings: bool = True
    match_cutoff: float = 0.7  # vector at least contextually falls under found ESCO value
    top_result_n: int = 10
    extra_skills_n: int = 25
    hf_token: Optional[str] = os.getenv("HF_TOKEN")

    @model_validator(mode='after')
    def prompt_for_hf_token(self):
        if self.hf_token: return self
        try:
            prompt = input(
                'input your huggingface token for faster download ("n" to continue anyway): '
            ).strip()
        except EOFError:
            prompt = ""
        if prompt and prompt.lower() != "n":
            self.hf_token = prompt
        return self