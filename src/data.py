import pandas as pd
import spacy
from sentence_transformers import CrossEncoder
from huggingface_hub import snapshot_download
from huggingface_hub import upload_folder
from huggingface_hub import HfApi
import os

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"

HF_REPO_ID = "daniel-5151/woo-ner"

local_dir = snapshot_download(repo_id=HF_REPO_ID)

# api = HfApi()

# api.upload_folder(
#     folder_path="../bert_output/ms-marco-MiniLM/checkpoint-279",
#     repo_id=HF_REPO_ID,
#     path_in_repo="ms-marco-MiniLM"
# )

def load_kb():
    path = os.path.join(local_dir, "alias_table.pkl")
    return pd.read_pickle(path)
    # return pd.read_pickle("../knowledge_base/alias_table.pkl")

def load_ner_model():
    model_path = os.path.join(local_dir, "model-best")
    nlp = spacy.load(model_path)

    # nlp = spacy.load("../output/model-best")
    nlp.add_pipe('sentencizer')
    return nlp

def load_el_model():
    model_path = os.path.join(local_dir, "ms-marco-MiniLM")
    return CrossEncoder(model_path)

def load_woogle_data(path: str):
    return pd.read_csv(path)