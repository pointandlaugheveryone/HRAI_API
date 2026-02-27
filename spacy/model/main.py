from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import spacy
from spacy.tokens import Doc
from app.models import RequestModel, ResponseModel, Document, Entity

'''
nlp = spacy.load('cvnlp')
if 'entity_ruler' not in nlp.pipe_names:
    ruler = nlp.add_pipe("entity_ruler").from_disk("patterns.jsonl") # type: ignore 
'''

def process_data(doc: Doc):
    ents = [
        {
            "text": ent.text,
            "label": ent.label_,
            "start": ent.start_char,
            "end": ent.end_char,
        }
        for ent in doc.ents
    ]
    return {"text": doc.text, "ents": ents}


app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"])


@app.post("/post/", summary="Process batches of text", response_model=ResponseModel)
def process_articles(query: RequestModel):
    """Process a batch of articles and return the entities predicted by the
    given model. Each record in the data should have a key "text".
    """
    nlp = spacy.load(query.request_nlp_model)
    response_body = []
    texts = (document.text for document in query.docs)
    for doc in nlp.pipe(texts):
        response_body.append(process_data(doc))
    return {"result": response_body}