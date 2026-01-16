import numpy as np

def prepare_candidates(candidates):
    fts_scores = []
    labels = []
    descriptions = []

    for i, (qid, rank, label, surface_from, desc) in enumerate(candidates):
        fts_scores.append(rank)
        labels.append(label)
        descriptions.append(f"{label} - {desc}")

    return fts_scores, labels, descriptions

def normalize_scores(scores, min_val, max_val):
    return (scores - min_val) / (max_val - min_val)

def compute_final_scores(encoder_scores, fts_scores, alpha):
    fts_min, fts_max = 4.2948, 25.2106
    encoder_min, encoder_max = -11.0917, 6.2494

    fts_scores_norm = normalize_scores(np.abs(fts_scores), fts_min, fts_max)
    encoder_scores_norm = normalize_scores(encoder_scores, encoder_min, encoder_max)

    return alpha * encoder_scores_norm + (1 - alpha) * fts_scores_norm