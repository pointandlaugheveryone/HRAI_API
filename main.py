from dotenv import load_dotenv

from config import conf
from load import get_encoder
from query import query_type, extract_from_resume
from suggestions import get_expanded_skills, get_domain_reports
from parse_doc import extract_text, UnsupportedFileTypeError

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

@app.post("/resume/skills")
async def post_resume_get_skills(file: UploadFile = File(...)):
    try:
        file_bytes = await file.read()
        text = extract_text(file_bytes, filename=file.filename)

    except UnsupportedFileTypeError as e:
        raise HTTPException(status_code=400, detail=str(e))

    entities = extract_from_resume(db, metadata, model, text)
    suggestions = get_expanded_skills(metadata, entities)
    return suggestions


@app.post("/resume/domains")
async def post_resume_get_domains(file: UploadFile = File(...)):
    try:
        file_bytes = await file.read()
        text = extract_text(file_bytes, filename=file.filename or '')
    except UnsupportedFileTypeError as e:
        raise HTTPException(status_code=400, detail=str(e))

    entities = extract_from_resume(db, metadata, model, text)
    suggestions = get_expanded_skills(metadata, entities)
    domains = get_domain_reports(suggestions)
    return domains


@app.post("/text/skills")
async def post_text_get_skills(req:TextInputRequest): # assume frontend will handle occupations not being null (i will hate myself)
    min_score = req.min_set_score or conf.match_cutoff
    occs = query_type(db, metadata, model, req.occupations, 'occupation', min_score=min_score)
    skills = query_type(db, metadata, model, req.skills, 'skill', min_score=min_score)
    suggestions = get_expanded_skills(metadata, occs+skills)
    return suggestions


@app.post("/text/domains")
def post_text_get_domains(req: TextInputRequest): # input a list of skills, occupations manually
    min_score = req.min_set_score or conf.match_cutoff
    occs = []
    if req.occupations:
        occs = query_type(db,metadata,model,req.occupations,'occupation', min_score=min_score)
    skills = query_type(db,metadata,model,req.skills,'skill', min_score=min_score)

    suggestions = get_expanded_skills(metadata,skills+occs)
    domains = get_domain_reports(suggestions)
    return domains


@app.post("/query")
def query(req: QueryRequest): # eg. find target occupation
    results = query_type(db,metadata,model, [req.query], req.query_type, req.min_set_score)
    return results