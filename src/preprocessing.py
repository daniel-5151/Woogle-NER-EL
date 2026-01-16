import unicodedata
import re

def normalize_unicode(text: str):
    text = unicodedata.normalize('NFKD', text)
    text = ''.join([c for c in text if not unicodedata.combining(c)])
    return text

def clean_text(text: str):
    text = normalize_unicode(str(text))
    text = re.sub(r'-\s+', '', text)
    text = re.sub(r'[^a-zA-Z0-9]', ' ', text)
    tokens = text.split()
    return ' '.join(token.strip() for token in tokens).lower()

def sample_documents(df, n_samples, seed=42):
        sampled_doc_ids = df['foi_documentId'].drop_duplicates().sample(n=n_samples, random_state=seed)
        sampled_pages = df[df['foi_documentId'].isin(sampled_doc_ids)].copy()
        return sampled_pages

def preprocess_df(df, sample: bool = False, n: int = 5):
    if sample:
        df = sample_documents(df, n)

    df = df.groupby("foi_documentId")['foi_bodyTextOCR']\
        .apply(lambda pages: " ".join(pages.dropna().astype(str)))\
        .reset_index()
     
    return df