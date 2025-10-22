# Script to build embeddings for all chunks in data/processed/chunks
import os
from src.embeddings.embedder import Embedder
import numpy as np
import pickle

def main():
    e = Embedder()
    chunk_files = [f for f in os.listdir('data/processed/chunks') if f.endswith('.txt')]
    texts = []
    ids = []
    for fn in chunk_files:
        with open(os.path.join('data/processed/chunks', fn), 'r', encoding='utf-8') as f:
            texts.append(f.read())
            ids.append(fn.replace('.txt',''))
    embs = e.encode(texts, batch_size=32)
    os.makedirs('data/index', exist_ok=True)
    np.save('data/index/embeddings.npy', embs)
    with open('data/index/ids.pkl', 'wb') as f:
        pickle.dump(ids, f)
    print('Saved embeddings and ids.')

if __name__ == '__main__':
    main()
