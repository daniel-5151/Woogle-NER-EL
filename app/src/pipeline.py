from src.data import load_kb, load_ner_model, load_el_model
from src.kb import index_kb

def load_pipeline():
    print("Loading Knowledge Base")
    kb = load_kb()

    print("Preparing Knowledge Base")
    _, cur = index_kb(kb)

    print("Loading NLP models")
    ner_model = load_ner_model()
    cross_encoder = load_el_model()

    return ner_model, cross_encoder, cur