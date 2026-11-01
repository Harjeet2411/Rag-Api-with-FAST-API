"""Split profile.txt into chunks, embed them with Ollama and store them in Chroma."""
import chromadb
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction

PROFILE_PATH = "profile.txt"
DB_PATH = "./chroma_db"
COLLECTION = "personal_profile"


def load_chunks(path: str) -> list[str]:
    with open(path, encoding="utf-8") as f:
        text = f.read()
    # Paragraphs are separated by a blank line.
    return [chunk.strip() for chunk in text.split("\n\n") if chunk.strip()]


def main() -> None:
    chunks = load_chunks(PROFILE_PATH)
    print(f"Loaded {len(chunks)} chunks from {PROFILE_PATH}")

    client = chromadb.PersistentClient(path=DB_PATH)
    embed = OllamaEmbeddingFunction(
        model_name="nomic-embed-text",
        url="http://localhost:11434",
    )
    collection = client.get_or_create_collection(
        name=COLLECTION, embedding_function=embed
    )

    collection.upsert(
        ids=[f"chunk{i}" for i in range(len(chunks))],
        documents=chunks,
        metadatas=[
            {"source": PROFILE_PATH, "chunk_index": i, "user": "harjeet"}
            for i in range(len(chunks))
        ],
    )
    print(f"Added {len(chunks)} chunks to the '{COLLECTION}' collection.")
    print("Knowledge base built successfully!")


if __name__ == "__main__":
    main()
