from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from src.process import return_entities
from src.data import load_kb, load_ner_model, load_el_model
from src.kb import index_kb

app = FastAPI()

# Delay loading until startup to avoid crashes
ner_model, cross_encoder = None, None

@app.on_event("startup")
def startup_event():
    global ner_model, cross_encoder
    print("Loading models...")
    ner_model = load_ner_model()
    cross_encoder = load_el_model()
    print("Models loaded!")

class TextInput(BaseModel):
    text: str

# Simple test endpoint
# @app.get("/analyze")
# def analyze_test(input: str = "Wilders heeft met de PVV keihard verloren tijdens de Tweede Kamer verkiezingen."):
#     kb = load_kb()
#     db, cur = index_kb(kb)
#     results = return_entities(input, ner_model, cross_encoder, cur)
#     db.close()
#     return {"entities": results}

@app.post("/analyze")
def analyze(input: TextInput):
    print("Loading Knowledge Base")
    kb = load_kb()
    db, cur = index_kb(kb)
    print("Knowledge Base Loaded")
    results = return_entities(input.text, ner_model, cross_encoder, cur)
    db.close()
    return {"entities": results}

@app.get("/", response_class=HTMLResponse)
def home():
    with open("index.html", "r", encoding="utf-8") as f:
        return f.read()


