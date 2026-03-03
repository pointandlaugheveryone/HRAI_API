import json
import os
from typing import List, Optional
from pathlib import Path

import faiss

from config import conf
from load import get_encoder
from models.EntityResult import EntityResult
from models.RequestModel import ManualContentRequest, QueryRequest
from models.ResponseModel import SkillsResponse, SuggestionResponse, DomainResponse, TargetJobResponse
from query import query_type, extract_from_resume
from suggestions import match_occupations, get_domain_skills
from parse_doc import extract_text, UnsupportedFileTypeError

from fastapi import FastAPI, UploadFile, File, Form, HTTPException



db = faiss.read_index(os.path.join(conf.db_dir, f"all.index"))

with open(os.path.join(conf.data_dir, f"key_to_ent.json"), "r", encoding="utf-8") as f:
    metadata = json.load(f)

model = get_encoder()

app = FastAPI(title="HRAI API")


@app.post("/resume", response_model=SuggestionResponse)
async def post_resume(
    file: UploadFile = File(...),
    target_job: Optional[str] = Form(None), # fastapi needs this to be form when multi part data is sent
):
    # noinspection DuplicatedCode
    try:
        file_bytes = await file.read()
        text = extract_text(file_bytes, filename=file.filename or '')
    except UnsupportedFileTypeError as e:
        raise HTTPException(status_code=400, detail=str(e))

    entities = extract_from_resume(text)
    skills = [e for e in entities if e.entity_type == 'skill']
    occupations = [e for e in entities if e.entity_type == 'occupation']

    suggestions = match_occupations(
        skills=skills,
        occupations=occupations,
        target_job=None)

    target_suggestion = None
    if target_job:
        target_suggestions = match_occupations(skills, None, target_job)
        target_suggestion = target_suggestions[0] if target_suggestions else None

    return SuggestionResponse(
        top_suggestion=suggestions[0] if suggestions else None,
        target_suggestion=target_suggestion)


@app.post("/resume/domains", response_model=DomainResponse)
async def post_resume_domains(
    file: UploadFile = File(...),
):
    # noinspection DuplicatedCode
    try:
        file_bytes = await file.read()
        text = extract_text(file_bytes, filename=file.filename or '')
    except UnsupportedFileTypeError as e:
        raise HTTPException(status_code=400, detail=str(e))

    entities = extract_from_resume(text)
    print(f'ent cnt {len(entities)}') # TODO remove
    skills = [e for e in entities if e.entity_type == 'skill']
    suggestions = match_occupations(skills, None, None)
    domains = get_domain_skills(suggestions)

    return DomainResponse(domains=domains)


@app.post("/text", response_model=SkillsResponse)
def post_text(req: ManualContentRequest):
    skill_labels = [s.strip() for s in req.skills.split(",")]
    skill_entities = query_type(skill_labels, 'skill')
    occupation_entities = query_type(skill_labels, 'occupation')

    suggestions = match_occupations(skill_entities, occupation_entities, req.target_job)
    return SkillsResponse(suggestions=suggestions)


@app.post("/text/goal", response_model=TargetJobResponse)
def post_text_goal(req: ManualContentRequest):
    skill_labels = [s.strip() for s in req.skills.split(",")]
    skill_entities = query_type(skill_labels, 'skill')
    occupation_entities = query_type(skill_labels, 'occupation')

    top_suggestions = match_occupations(skill_entities, occupation_entities, None)
    top_suggestion = top_suggestions[0] if top_suggestions else None

    target_suggestion = None
    if req.target_job:
        target_suggestions = match_occupations(skill_entities, None, req.target_job)
        target_suggestion = target_suggestions[0] if target_suggestions else None

    return TargetJobResponse(top_suggestion=top_suggestion, target_suggestion=target_suggestion)


@app.post("/query", response_model=List[EntityResult])
def query(req: QueryRequest):
    label = req.entity_type
    results = query_type([req.text], label, min_score=req.n if req.n else 5)
    return results