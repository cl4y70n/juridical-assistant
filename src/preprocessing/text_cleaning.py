import re, unicodedata

def clean_text(text: str) -> str:
    text = unicodedata.normalize('NFKC', text)
    text = re.sub(r'\s+', ' ', text)
    text = ''.join(ch for ch in text if ch.isprintable())
    return text.strip()

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100):
    tokens = text.split()
    chunks = []
    i = 0
    while i < len(tokens):
        chunk = ' '.join(tokens[i:i+chunk_size])
        chunks.append(chunk)
        i += chunk_size - overlap
    return chunks
