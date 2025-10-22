from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List

class Embedder:
    def __init__(self, model_name: str = 'sentence-transformers/all-mpnet-base-v2'):
        self.model = SentenceTransformer(model_name)
        self.dim = self.model.get_sentence_embedding_dimension()

    def encode(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        embs = self.model.encode(texts, batch_size=batch_size, show_progress_bar=True, convert_to_numpy=True)
        return embs

if __name__ == '__main__':
    e = Embedder()
    print('Dim:', e.dim)
    print(e.encode(['teste de embedding', 'outra sentença']))
