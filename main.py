from src.data import load_kb, load_ner_model, load_el_model, load_woogle_data
from src.kb import index_kb
from src.preprocessing import normalize_unicode, clean_text
from src.utils import prepare_candidates, normalize_scores, compute_final_scores
# from named_entity_recognition import named_entity_recognition
from src.process import test

def main():
    print("Loading Knowledge Base")
    kb = load_kb()

    print("Loading NLP models")
    ner_model = load_ner_model()
    cross_encoder = load_el_model()

    print("indexing_kb")
    db, cur = index_kb(kb)

    print("geslaagd")
    df1 = load_woogle_data("../woo_data/2b_clean.csv")
    # df2 = load_woogle_data("../woo_data/2c_clean.csv")
    # df3 = load_woogle_data("../woo_data/2e-b_clean.csv")
    # df4 = load_woogle_data("../woo_data/2i_clean.csv")
    
    def sample_documents(df, n_samples, seed=42):
        sampled_doc_ids = df['foi_documentId'].drop_duplicates().sample(n=n_samples, random_state=seed)
        sampled_pages = df[df['foi_documentId'].isin(sampled_doc_ids)].copy()
        return sampled_pages

    n = 5
    df = sample_documents(df1, n)

    df = df.groupby("foi_documentId")['foi_bodyTextOCR']\
        .apply(lambda pages: " ".join(pages.dropna().astype(str)))\
        .reset_index()
    
    test(df, ner_model, cross_encoder, cur)

if __name__ == "__main__":
    main()


