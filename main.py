from dotenv import load_dotenv

from config import conf
from load import get_encoder
from query import query_type, extract_from_resume
from suggestions import get_expanded_skills, get_domain_reports
from parse_doc import extract_text, extract_text_from_pdf_skills_section

import json, os
from typing import List, Optional

from fastapi import FastAPI, UploadFile, File, HTTPException
import faiss
from pydantic import BaseModel


class TextInputRequest(BaseModel):
    occupations: Optional[List[str]]
    skills: List[str]
    min_set_score: Optional[float] = conf.match_cutoff

class QueryRequest(BaseModel):
    query: str
    query_type: str
    min_set_score: Optional[float] = conf.match_cutoff


load_dotenv()
db = faiss.read_index(os.path.join(conf.db_dir, f"all.index"))
with open(os.path.join(conf.data_dir, f"key_to_ent.json"),'r') as f:
    metadata = json.loads(f.read())
model = get_encoder()


app = FastAPI(title="HRAI API")


def _clean_text(text: str) -> str:
    # Normalize whitespace to improve matching consistency.
    return " ".join(text.split())


def _normalize_items(items: Optional[List[str]]) -> List[str]:
    if not items:
        return []
    return [" ".join(item.split()) for item in items if item and item.strip()]


def _extract_resume_text(file_bytes: bytes, filename: str, use_skills_section: bool = True) -> str:
    if use_skills_section and filename.lower().endswith(".pdf"):
        text = extract_text_from_pdf_skills_section(file_bytes)
        if text:
            return _clean_text(text)
    return _clean_text(extract_text(file_bytes, filename=filename))


def resume_skills_from_bytes(
    file_bytes: bytes,
    filename: str,
    min_score: Optional[float] = None,
    use_skills_section: bool = True,
) -> dict:
    text = _extract_resume_text(file_bytes, filename, use_skills_section=use_skills_section)
    entities = extract_from_resume(db, metadata, model, text, min_score or conf.match_cutoff)
    suggestions = get_expanded_skills(metadata, entities)
    return {"suggestions": suggestions}


def resume_domains_from_bytes(
    file_bytes: bytes,
    filename: str,
    min_score: Optional[float] = None,
    use_skills_section: bool = True,
) -> dict:
    text = _extract_resume_text(file_bytes, filename, use_skills_section=use_skills_section)
    entities = extract_from_resume(db, metadata, model, text, min_score or conf.match_cutoff)
    suggestions = get_expanded_skills(metadata, entities)
    domains = get_domain_reports(suggestions)
    return {"domains": domains}


def text_skills(
    skills: List[str],
    occupations: Optional[List[str]] = None,
    min_score: Optional[float] = None,
) -> dict:
    min_score = min_score or conf.match_cutoff
    occs = query_type(db, metadata, model, _normalize_items(occupations), "occupation", min_score=min_score)
    skill_results = query_type(db, metadata, model, _normalize_items(skills), "skill", min_score=min_score)
    suggestions = get_expanded_skills(metadata, occs + skill_results)
    return {"suggestions": suggestions}


def text_domains(
    skills: List[str],
    occupations: Optional[List[str]] = None,
    min_score: Optional[float] = None,
) -> dict:
    min_score = min_score or conf.match_cutoff
    occs = query_type(db, metadata, model, _normalize_items(occupations), "occupation", min_score=min_score)
    skill_results = query_type(db, metadata, model, _normalize_items(skills), "skill", min_score=min_score)
    suggestions = get_expanded_skills(metadata, skill_results + occs)
    domains = get_domain_reports(suggestions)
    return {"domains": domains}


def query_entities(
    query: str,
    query_type_name: str,
    min_score: Optional[float] = None,
) -> dict:
    results = query_type(
        db,
        metadata,
        model,
        ents=query,
        ent_type=query_type_name,
        min_score=min_score or conf.match_cutoff,
    )
    return {"results": results}


@app.post("/resume/skills")
async def post_resume_get_skills(file: UploadFile = File(...)):
    try:
        file_bytes = await file.read()
        text = _extract_resume_text(file_bytes, filename=file.filename or "")

    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))

    entities = extract_from_resume(db, metadata, model, text)
    suggestions = get_expanded_skills(metadata, entities)
    return {"suggestions": suggestions}


@app.post("/resume/domains")
async def post_resume_get_domains(file: UploadFile = File(...)):
    try:
        file_bytes = await file.read()
        text = _extract_resume_text(file_bytes, filename=file.filename or "")
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))

    entities = extract_from_resume(db, metadata, model, text)
    suggestions = get_expanded_skills(metadata, entities)
    domains = get_domain_reports(suggestions)
    return {"domains": domains}


@app.post("/text/skills")
async def post_text_get_skills(req:TextInputRequest): # assume frontend will handle occupations not being null (i will hate myself)
    min_score = req.min_set_score or conf.match_cutoff
    occs = query_type(db, metadata, model, _normalize_items(req.occupations), 'occupation', min_score=min_score)
    skills = query_type(db, metadata, model, _normalize_items(req.skills), 'skill', min_score=min_score)
    suggestions = get_expanded_skills(metadata, occs + skills)
    return {"suggestions": suggestions}


@app.post("/text/domains")
def post_text_get_domains(req: TextInputRequest): # input a list of skills, occupations manually
    min_score = req.min_set_score or conf.match_cutoff
    occs = query_type(db, metadata, model, _normalize_items(req.occupations), 'occupation', min_score=min_score)
    skills = query_type(db, metadata, model, _normalize_items(req.skills), 'skill', min_score=min_score)

    suggestions = get_expanded_skills(metadata, skills + occs)
    domains = get_domain_reports(suggestions)
    return {"domains": domains}


@app.post("/query")
def query(req: QueryRequest): # eg. find target occupation // for some reason the min score parameter messes up the lookup when set to None, even with a default value
    results = query_type(
        db,
        metadata,
        model,
        ents=req.query,
        ent_type=req.query_type,
        min_score=req.min_set_score or conf.match_cutoff,
    )
    return {"results": results}