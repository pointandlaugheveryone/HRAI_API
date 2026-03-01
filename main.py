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
async def analyze_resume(
    file: UploadFile = File(...),
    target_job: Optional[str] = Form(None)
):
    """
    Pipeline 1: pdf/docx/odt + target job → occupation info, skill suggestions, match score
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

    top_suggestion = suggestions[0] if suggestions else None
    target_suggestion = None
    if target_job:
        target_suggestions = match_occupations(skills, None, target_job)
        target_suggestion = target_suggestions[0] if target_suggestions else None

    return SuggestionResponse(top_suggestion=top_suggestion, target_suggestion=target_suggestion)


@app.post("/resume/domains", response_model=DomainResponse)
async def analyze_resume_domains(file: UploadFile = File(...)):
    """
    Pipeline 2: pdf/docx/odt → suggestions grouped into ISCO domains
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
def suggest_from_skills(req: ManualContentRequest):
    """
    Pipeline 3: skills string → suggestions ordered by match score
    Pipeline 4: skills string + target_job → top match + target job suggestion
    """
    skill_labels = [s.strip() for s in req.skills.split(",")]
    skill_entities = query_type(skill_labels, 'skill')
    occupation_entities = query_type(skill_labels, 'occupation')

    suggestions = match_occupations(skill_entities, occupation_entities, req.target_job)
    return SkillsResponse(suggestions=suggestions)


@app.post("/text/goal", response_model=TargetJobResponse)
def suggest_with_target(req: ManualContentRequest):
    """
    Pipeline 4: skills string + target_job → top general match + target job match
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
    Pipeline 5: string, entity type (optional), n → n EntityResults
    """
    label = req.entity_type
    results = query_type([req.text], label, search_k=req.n)
    return results