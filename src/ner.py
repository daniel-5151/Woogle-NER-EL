def named_entity_recognition(text, nlp):
    doc = nlp(text)
    entities = []
    for ent in doc.ents:
        entities.append({
            'mention': ent.text,
            'context': ent.sent.text,
            'start': ent.start
        })
    return entities