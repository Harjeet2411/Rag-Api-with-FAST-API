"""RAG API: answer questions about a profile using Chroma retrieval + a local Ollama model."""
import chromadb
import ollama
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction
from fastapi import FastAPI

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


@app.get("/ask")
def ask(question: str):
    results = collection.query(query_texts=[question], n_results=2)
    context_chunks = results["documents"][0]
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
