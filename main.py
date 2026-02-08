from src.process import return_entities
from src.pipeline import load_pipeline

if __name__ == "__main__":
    ner_model, cross_encoder, cur = load_pipeline()

    # print(results)
    text = "Wilders heeft met de PVV keihard verloren tijdens de Tweede Kamer verkiezingen."

    results = return_entities(text, ner_model, cross_encoder, cur)
    print(results)


