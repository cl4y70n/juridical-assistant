# Script to load embeddings and build FAISS index
import numpy as np
from src.indexer.faiss_indexer import FaissIndexer
import os, pickle

def main():
    embs = np.load('data/index/embeddings.npy')
    with open('data/index/ids.pkl','rb') as f:
        ids = pickle.load(f)
    dim = embs.shape[1]
    idx = FaissIndexer(dim=dim)
    idx.add(embs, ids)
    idx.save()
    print('FAISS index built and saved.')

if __name__ == '__main__':
    main()
