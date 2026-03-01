from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from typing import List, Optional

from models.EntityResult import EntityResult
from models.RequestModel import ManualContentRequest, QueryRequest
from models.ResponseModel import SkillsResponse, SuggestionResponse, DomainResponse, TargetJobResponse
from query import query_type, extract_from_resume
from suggestions import match_occupations, get_domain_skills
from parse_doc import extract_text, UnsupportedFileTypeError


app = FastAPI(title="HRAI API")

@app.post("/resume", response_model=SuggestionResponse)
async def get_resume_info(
    file: UploadFile = File(...),
    target_job: Optional[str] = Form(None)
):
    """
    pdf/docx/odt + target job
    -> occupation, match scored suggestions
    """
    # noinspection DuplicatedCode
    try:
        file_bytes = await file.read()
        text = extract_text(file.filename or '', file_bytes)
    except UnsupportedFileTypeError as e:
        raise HTTPException(status_code=400, detail=str(e))

    entities = extract_from_resume(text)
    skills = [e for e in entities if e.entity_type == 'skill']
    occupations = [e for e in entities if e.entity_type == 'occupation']

    suggestions = match_occupations(skills, occupations, target_job)
    target_suggestion = None
    if target_job:
        target_suggestions = match_occupations(skills, None, target_job)
        target_suggestion = target_suggestions[0] if target_suggestions else None

    return SuggestionResponse(top_suggestion=suggestions[0], target_suggestion=target_suggestion)


@app.post("/resume/domains", response_model=DomainResponse)
async def get_resume_info_isco(file: UploadFile = File(...)):
    """
    pdf/docx/odt 
    → suggestions groupedby domains
    """
    # noinspection DuplicatedCode
    try:
        file_bytes = await file.read()
        text = extract_text(file.filename or '', file_bytes)
    except UnsupportedFileTypeError as e:
        raise HTTPException(status_code=400, detail=str(e))

    entities = extract_from_resume(text)
    skills = [e for e in entities if e.entity_type == 'skill']
    occupations = [e for e in entities if e.entity_type == 'occupation']

    suggestions = match_occupations(skills, occupations, None)
    domains = get_domain_skills(suggestions)

    return DomainResponse(domains=domains)


@app.post("/text", response_model=SkillsResponse)
def get_skill_info(req: ManualContentRequest):
    """
    skills string
    → suggestions ordered by match score
    OR
    skills string, target job
    → top match + target job skill suggestion
    """
    skill_labels = [s.strip() for s in req.skills.split(",")]
    skill_entities = query_type(skill_labels, 'skill')
    occupation_entities = query_type(skill_labels, 'occupation')

    suggestions = match_occupations(skill_entities, occupation_entities, req.target_job)
    return SkillsResponse(suggestions=suggestions)


@app.post("/text/goal", response_model=TargetJobResponse)
def get_skill_job_info(req: ManualContentRequest):
    """
    skills string, target job
    → target match, global match
    """
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
def query_entities(req: QueryRequest):
    """
    string, entity type (optional), n (optional)
    → n EntityResults
    """
    label = req.entity_type
    results = query_type([req.text], label, search_k=req.n if req.n else None)
    return results