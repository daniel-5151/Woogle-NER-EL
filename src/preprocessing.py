import unicodedata
import re

def normalize_unicode(text):
    text = unicodedata.normalize('NFKD', text)
    text = ''.join([c for c in text if not unicodedata.combining(c)])
    return text

def clean_text(text):
    text = normalize_unicode(str(text))
    text = re.sub(r'-\s+', '', text)
    text = re.sub(r'[^a-zA-Z0-9]', ' ', text)
    tokens = text.split()
    return ' '.join(token.strip() for token in tokens).lower()