from src.data import load_kb, load_ner_model, load_el_model, load_woogle_data
from src.kb import index_kb
from src.preprocessing import preprocess_df
from src.process import process_batches

def main():
    print("Loading Knowledge Base")
    kb = load_kb()

    print("Indexing Knowledge Base")
    _, cur = index_kb(kb)

    print("Loading NLP models")
    ner_model = load_ner_model()
    cross_encoder = load_el_model()

    print("Loading Woogle data")
    df1 = load_woogle_data("../woo_data/2b_clean.csv")
    # df2 = load_woogle_data("../woo_data/2c_clean.csv")
    # df3 = load_woogle_data("../woo_data/2e-b_clean.csv")
    # df4 = load_woogle_data("../woo_data/2i_clean.csv")
    
    df = preprocess_df(df1, True, 30)
    
    results = process_batches(df, ner_model, cross_encoder, cur)
    print(results)

if __name__ == "__main__":
    main()


