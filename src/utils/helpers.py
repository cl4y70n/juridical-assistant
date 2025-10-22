import os
from typing import List

def ensure_dirs():
    for d in ['data/raw', 'data/processed', 'data/processed/chunks', 'data/index']:
        os.makedirs(d, exist_ok=True)

def save_chunks(doc_id: str, chunks: List[str]):
    os.makedirs('data/processed/chunks', exist_ok=True)
    ids = []
    for i, c in enumerate(chunks):
        chunk_id = f"{doc_id}_chunk_{i}"
        path = f"data/processed/chunks/{chunk_id}.txt"
        with open(path, 'w', encoding='utf-8') as f:
            f.write(c)
        ids.append(chunk_id)
    return ids
