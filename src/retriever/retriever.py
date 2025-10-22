from typing import List
from src.embeddings.embedder import Embedder
from src.indexer.faiss_indexer import FaissIndexer

class Retriever:
    def __init__(self, embedder: Embedder, faiss_index: FaissIndexer):
        self.embedder = embedder
        self.index = faiss_index

    def retrieve(self, query: str, k: int = 8):
        q_emb = self.embedder.encode([query])
        results = self.index.search(q_emb, k=k)
        return results[0]

if __name__ == '__main__':
    emb = Embedder()
    idx = FaissIndexer(dim=emb.dim)
    idx.load()
    r = Retriever(emb, idx)
    print(r.retrieve('Qual o prazo de prescrição?'))
