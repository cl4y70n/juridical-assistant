from src.embeddings.embedder import Embedder

def test_embedder_dim():
    e = Embedder()
    em = e.encode(['teste'])
    assert em.shape[1] == e.dim
