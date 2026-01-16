import pandas as pd
import spacy
from sentence_transformers import CrossEncoder

def load_kb():
    return pd.read_pickle("../knowledge_base/alias_table.pkl")

def load_ner_model():
    nlp = spacy.load("../output/model-best")
    nlp.add_pipe('sentencizer')
    return nlp

def load_el_model():
    return CrossEncoder("../bert_output/ms-marco-MiniLM/checkpoint-279")

def load_woogle_data(path: str):
    return pd.read_csv(path)