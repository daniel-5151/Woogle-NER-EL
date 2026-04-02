import os
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from app.src.process import return_entities
from app.src.data import load_kb, load_ner_model, load_el_model
from app.src.kb import index_kb

app = FastAPI()
file_path = os.path.join(os.path.dirname(__file__), "index.html")

# Delay loading until startup
ner_model, cross_encoder, kb, db, cur = None, None, None, None, None

@app.on_event("startup")
def startup_event():
    global ner_model, cross_encoder, kb, cur
    print("Loading models...")
    ner_model = load_ner_model()
    cross_encoder = load_el_model()
    print("Models loaded!")
    print("Loading Knowledge Base")
    kb = load_kb()
    db, cur = index_kb(kb)
    print("Knowledge Base Loaded")

class TextInput(BaseModel):
    text: str

@app.post("/analyze")
def analyze(input: TextInput):
    print("Process running")
    results = return_entities(input.text, ner_model, cross_encoder, cur)
    print("Process finished")
    return {"entities": results}

@app.get("/", response_class=HTMLResponse)
def home():
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


