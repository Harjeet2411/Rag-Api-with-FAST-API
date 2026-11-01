# RAG API with FastAPI

A Retrieval-Augmented Generation API that answers questions about my profile, fully local: **Ollama** serves the models (`qwen2.5:0.5b` chat, `nomic-embed-text` embeddings), **ChromaDB** stores the vectors and **FastAPI** exposes the endpoints.

Needs Ollama running locally (see [Rag-Api-Ollama](https://github.com/Harjeet2411/Rag-Api-Ollama)) and Python 3.10+.


```bash
python -m venv venv
venv\Scripts\activate                # Windows  (source venv/bin/activate on Linux/macOS)
pip install -r requirements.txt
ollama pull nomic-embed-text
python build_knowledge_base.py
```
