from src.utils import prepare_candidates, compute_final_scores
from src.preprocessing import clean_text

import numpy as np

def candidate_generation(mention, cur, limit=15, window=10):
    terms = mention.split()
    if len(terms) >= 2:
        quoted_terms = ' '.join(f'"{t}"' for t in terms)
        query = f"NEAR({quoted_terms}, {window})"
    else:
        query = f"surface_form:{mention}"

    qids = []
    res = cur.execute(f"""
        SELECT qid, MIN(rank) as best_rank, label, normal_form, description
        FROM kb
        WHERE surface_form MATCH ?
        GROUP BY qid
        ORDER BY best_rank
        LIMIT ?
        """, (query, limit)).fetchall()

    for candidate in res:
        qids.append(candidate[0])

    return qids, res

def ranking(qids, candidates, context, model, alpha):
    if len(qids) == 0:
        return None

    fts_scores, labels, descriptions = prepare_candidates(candidates)

    pairs = [(context, cand) for cand in descriptions]
    encoder_scores = np.array(model.predict(pairs)) # Select re-ranking model
    fts_scores = abs(np.array(fts_scores))

    final_scores = compute_final_scores(encoder_scores, fts_scores, alpha)

    best_idx = np.argmax(final_scores)
    return qids[best_idx], labels[best_idx], final_scores[best_idx]

def entity_linking(mention, context, model, cur, limit=10, alpha=0.55, threshold=0.55): # Select hyperparameters
    entity_clean = clean_text(mention)
    qids, candidates = candidate_generation(entity_clean, cur, limit)
    result = ranking(qids, candidates, context, model, alpha)

    if result is None:
        return 'NILL'

    qid, label, score = result
    if score > threshold:
        return qid # Change to 'label' to return page titles 
    else:
        return 'NILL'