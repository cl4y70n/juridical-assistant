import faiss, numpy as np, os, pickle
from typing import List

class FaissIndexer:
    def __init__(self, dim: int, index_path: str = 'data/index/faiss.index', ids_path: str = 'data/index/ids.pkl'):
        self.dim = dim
        self.index_path = index_path
        self.ids_path = ids_path
        self.index = faiss.IndexFlatIP(dim)
        self.ids = []

    def add(self, vectors: np.ndarray, ids: List[str]):
        assert vectors.shape[1] == self.dim
        faiss.normalize_L2(vectors)
        self.index.add(vectors)
        self.ids.extend(ids)

    def search(self, vector: np.ndarray, k: int = 8):
        faiss.normalize_L2(vector)
        D, I = self.index.search(vector, k)
        results = []
        for i, idxs in enumerate(I):
            res = []
            for j, idx in enumerate(idxs):
                if idx == -1:
                    continue
                res.append({'id': self.ids[idx], 'score': float(D[i][j])})
            results.append(res)
        return results

    def save(self):
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        faiss.write_index(self.index, self.index_path)
        with open(self.ids_path, 'wb') as f:
            pickle.dump(self.ids, f)

    def load(self):
        if os.path.exists(self.index_path):
            self.index = faiss.read_index(self.index_path)
        if os.path.exists(self.ids_path):
            with open(self.ids_path, 'rb') as f:
                self.ids = pickle.load(f)
