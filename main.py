"""RAG API: answer questions about a profile using Chroma retrieval + a local Ollama model."""
import uuid

import chromadb
import ollama
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

DB_PATH = "./chroma_db"
COLLECTION = "personal_profile"
CHAT_MODEL = "qwen2.5:0.5b"

app = FastAPI(title="Personal RAG API")

client = chromadb.PersistentClient(path=DB_PATH)
embed = OllamaEmbeddingFunction(
    model_name="nomic-embed-text",
    url="http://localhost:11434",
)
collection = client.get_or_create_collection(name=COLLECTION, embedding_function=embed)


class Document(BaseModel):
    user: str
    text: str


@app.post("/documents")
def add_document(doc: Document):
    """Secret mission: store another user's profile so /ask can be filtered by user."""
    chunks = [c.strip() for c in doc.text.split("\n\n") if c.strip()]
    if not chunks:
        raise HTTPException(status_code=400, detail="text is empty")
    batch = uuid.uuid4().hex[:8]
    collection.add(
        ids=[f"{doc.user}-{batch}-{i}" for i in range(len(chunks))],
        documents=chunks,
        metadatas=[
            {"source": "api", "chunk_index": i, "user": doc.user}
            for i in range(len(chunks))
        ],
    )
    return {"user": doc.user, "chunks_added": len(chunks)}


@app.get("/ask")
def ask(question: str, user: str | None = None):
    where = {"user": user} if user else None
    results = collection.query(query_texts=[question], n_results=2, where=where)
    context_chunks = results["documents"][0] if results["documents"] else []
    if not context_chunks:
        raise HTTPException(status_code=404, detail="no documents found for this user")
    context = "\n\n".join(context_chunks)

    prompt = (
        "Answer the question using only the context below. "
        "If the context does not contain the answer, say you don't know.\n\n"
        f"Context:\n{context}\n\nQuestion: {question}"
    )
    response = ollama.chat(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    return {
        "question": question,
        "answer": response["message"]["content"],
        "context_used": context_chunks,
    }
