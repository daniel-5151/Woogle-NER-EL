import pandas as pd
from tqdm import tqdm
import gc

from src.ner import named_entity_recognition
from src.el import entity_linking

def return_entities(text: str, ner_model, el_model, cur):
    if len(text) > 200000: # Documents with high number of tokens can overload the RAM
        return []


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


    return entities_norm

def process_batches(df, ner_model, el_model, cur, batch_size=5):
    results = []

    total_batches = len(df) // batch_size + 1
    for batch_idx in tqdm(range(total_batches), desc="Processing Batches"):
        start = batch_idx * batch_size
        end = min(start + batch_size, len(df))
        
        batch = df.iloc[start:end].copy()
        batch_results = []
    
        for text in batch['foi_bodyTextOCR']:
            try:
                ents = return_entities(text, ner_model, el_model, cur)
            except Exception as e:
                # print(f"Error processing text: {e}")
                ents = []
            batch_results.append(ents)
    
        batch['entities'] = batch_results
        results.append(batch)
        
        del batch, batch_results
        gc.collect()

    df_results = pd.concat(results, ignore_index=True)
    return df_results