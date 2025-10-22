from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import os

from src.embeddings.embedder import Embedder
from src.indexer.faiss_indexer import FaissIndexer
from src.retriever.retriever import Retriever
from src.llm.serve_llm import LLMService

app = FastAPI(title='ChatLegal API')

EMBED_MODEL = os.environ.get('EMBED_MODEL', 'sentence-transformers/all-mpnet-base-v2')
embedder = Embedder(model_name=EMBED_MODEL)
faiss_idx = FaissIndexer(dim=embedder.dim)
faiss_idx.load()
retriever = Retriever(embedder, faiss_idx)
llm = LLMService()

class Query(BaseModel):
    question: str
    top_k: int = 8

@app.post('/chat')
async def chat(q: Query):
    if not q.question:
        raise HTTPException(status_code=400, detail='Pergunta vazia')
    hits = retriever.retrieve(q.question, k=q.top_k)
    context_texts = []
    for h in hits:
        try:
            with open(f"data/processed/chunks/{h['id']}.txt", 'r', encoding='utf-8') as f:
                context_texts.append(f.read())
        except FileNotFoundError:
            continue
    prompt = build_prompt(q.question, context_texts)
    resp = llm.generate(prompt)
    return {'answer': resp, 'sources': [h['id'] for h in hits]}

def build_prompt(question: str, contexts: List[str]):
    system = ("Você é ChatLegal, um assistente jurídico que responde com precisão e cita as fontes."
              " Responda de forma sucinta e sempre mostre as evidências usadas.")
    ctx = '\n\n---\n\n'.join(contexts[:5])
    prompt = f"{system}\n\nContextos:\n{ctx}\n\nPergunta: {question}\n\nResposta:"
    return prompt
