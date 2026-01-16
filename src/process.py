from tqdm import tqdm
from src.ner import named_entity_recognition
from src.el import entity_linking

def return_entities(text: str):
    if len(text) > 200000: # Documents with high number of tokens can overload the RAM
        return []


    entities = named_entity_recognition(text)
    entities_norm = []

    for entity in entities:
        mention = entity['mention']
        context = entity['context']
        start = entity['start']
        linked = entity_linking(mention, context)
        if linked != 'NILL':
            entities_norm.append({
                'entity': linked,
                'start': start
            })


    return entities_norm

def test(df, ner_model, el_model, cur, batch_size=5):
    results = []

    total_batches = len(df) // batch_size + 1
    for batch_idx in tqdm(range(total_batches), desc="Processing Batches"):
        start = batch_idx * batch_size
        end = min(start + batch_size, len(df))
        
        batch = df.iloc[start:end].copy()
        batch_results = []
        for text in batch['foi_bodyTextOCR']:
            entities = named_entity_recognition(text, ner_model)
            
            entities_norm = []

            for entity in entities:
                mention = entity['mention']
                context = entity['context']
                start = entity['start']
                linked = entity_linking(mention, context, el_model, cur)
                if linked != 'NILL':
                    entities_norm.append({
                        'entity': linked,
                        'start': start
                    })


        print(entities_norm)
    return 0