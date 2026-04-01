from src.process import return_entities
from src.pipeline import load_pipeline

if __name__ == "__main__":
    ner_model, cross_encoder, cur = load_pipeline()

    # print(results)
    text = "Geert Wilders van de PVV wilt dat de NPO geen geld meer krijgt."

    results = return_entities(text, ner_model, cross_encoder, cur)
    print(results)


