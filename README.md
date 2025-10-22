# ChatLegal — Chatbot Jurídico Inteligente

## Descrição do Projeto

**ChatLegal** é um **chatbot jurídico** que responde perguntas baseadas em documentos legais, utilizando técnicas de **NLP, embeddings, LLM fine-tuning** e **RAG (Retrieval Augmented Generation)**.
O sistema permite consultas jurídicas rápidas e precisas, recuperando contexto de documentos e oferecendo respostas explicáveis.

---

## Arquitetura do Projeto

```
ChatLegal_Project/
│
├─ data/
│  ├─ raw/                # Documentos brutos (PDF, DOCX, TXT)
│  ├─ processed/          # Texto processado
│  │  └─ chunks/          # Chunks de texto prontos para embeddings
│  └─ index/              # Index FAISS e embeddings
│
├─ src/
│  ├─ ingestion/          # Extração de texto de arquivos
│  ├─ preprocessing/      # Limpeza e chunking de texto
│  ├─ embeddings/         # Criação de embeddings
│  ├─ indexer/            # FAISS index
│  ├─ retriever/          # Recuperação de chunks relevantes
│  ├─ llm/                # Fine-tuning LoRA e serviço LLM
│  ├─ api/                # API FastAPI
│  ├─ webui/              # Interface Streamlit
│  └─ utils/              # Helpers gerais
│
├─ notebooks/             # Notebooks de EDA ou testes
├─ infra/                 # Scripts de deploy/K8s
├─ requirements.txt       # Dependências
├─ Dockerfile
├─ docker-compose.yml
├─ params.yaml            # Configurações de modelo, index e treino
└─ README.md
```

---

## Tecnologias e Bibliotecas

* **Python 3.10+**
* **NLP e LLM:** Transformers, Hugging Face, PEFT (LoRA), sentence-transformers
* **Indexação e Recuperação:** FAISS, ChromaDB (opcional)
* **API e UI:** FastAPI, Uvicorn, Streamlit
* **Manipulação de Dados:** Pandas, Numpy
* **Deployment:** Docker, Docker Compose, Kubernetes (manifests inclusos)
* **Versionamento ML e MLOps:** DVC e MLflow (opcional)

---

## Funcionalidades

1. Ingestão automática de documentos (`.pdf`, `.docx`, `.txt`).
2. Pré-processamento com limpeza, normalização e chunking.
3. Geração de embeddings de chunks com **SentenceTransformers**.
4. Construção de índice FAISS para recuperação de contexto.
5. Chatbot com LLM fine-tunado via LoRA, capaz de gerar respostas precisas.
6. Interface web para consultas jurídicas em tempo real.
7. API REST para integração com outros sistemas.
8. Explicabilidade com retorno de fontes e evidências utilizadas.

---

## Como Rodar Localmente

### 1. Clonar o repositório

```bash
git clone https://github.com/cl4y70n/juridical-assistant.git
cd juridical-assistant
```

### 2. Criar e ativar ambiente virtual

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

### 3. Instalar dependências

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Colocar documentos legais

* Adicione PDFs, DOCX ou TXT em `data/raw/`.

### 5. Ingestão de documentos

```bash
python src/ingestion/ingest.py --input data/raw --output data/processed
```

### 6. Pré-processamento e chunking

```python
from src.preprocessing.text_cleaning import clean_text, chunk_text
from src.utils.helpers import save_chunks

# Exemplo
text = open('data/processed/sample.txt', 'r').read()
chunks = chunk_text(clean_text(text))
save_chunks('sample_doc', chunks)
```

### 7. Gerar embeddings

```bash
python src/make_embeddings.py
```

### 8. Construir FAISS index

```bash
python src/build_index.py
```

### 9. Rodar API FastAPI

```bash
uvicorn src.api.main:app --reload
```

### 10. Rodar interface Streamlit

```bash
streamlit run src/webui/app.py
```

---

## Deploy com Docker

1. Build da imagem:

```bash
docker build -t chatlegal:latest .
```

2. Rodar via Docker Compose:

```bash
docker-compose up --build
```

3. Acesse:

* API: `http://localhost:8000`
* WebUI: `http://localhost:8501`

---

## Scripts Úteis

* `src/ingestion/ingest.py` → extrai texto de arquivos.
* `src/preprocessing/text_cleaning.py` → limpeza e chunking de textos.
* `src/make_embeddings.py` → cria embeddings de chunks.
* `src/build_index.py` → cria índice FAISS.
* `src/llm/fine_tune_lora.py` → treina LLM LoRA.
* `src/api/main.py` → API FastAPI.
* `src/webui/app.py` → interface web Streamlit.

---

## Contribuição

1. Fork do repositório.
2. Criar branch: `git checkout -b feature/minha-feature`.
3. Commit: `git commit -m "Minha feature"`.
4. Push: `git push origin feature/minha-feature`.
5. Abrir Pull Request.

---

## Autor

* **Clayton** – [GitHub](https://github.com/cl4y70n)

---

Quer que eu faça isso?
