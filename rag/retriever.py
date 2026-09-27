from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from document_loader import load_and_chunk_documents


# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
VECTOR_STORE_PATH = BASE_DIR / "rag" / "vector_store.index"


# Embedding model
MODEL_NAME = "all-MiniLM-L6-v2"
model = SentenceTransformer(MODEL_NAME)


def load_vector_store():
    """Load the saved FAISS vector index."""
    if not VECTOR_STORE_PATH.exists():
        raise FileNotFoundError(
            f"Vector store not found: {VECTOR_STORE_PATH}"
        )

    return faiss.read_index(str(VECTOR_STORE_PATH))


def search_documents(query: str, top_k: int = 3):
    """
    Search the vector store for documents
    semantically similar to the user's query.
    """

    # Load FAISS index
    index = load_vector_store()

    # Load document chunks
    chunks = load_and_chunk_documents()

    # Convert query into an embedding
    query_embedding = model.encode([query])

    # Convert to float32 for FAISS
    query_embedding = np.array(query_embedding).astype("float32")

    # Search FAISS
    distances, indices = index.search(query_embedding, top_k)

    results = []

    for distance, index_id in zip(distances[0], indices[0]):

        if index_id == -1:
            continue

        chunk = chunks[index_id]

        results.append(
            {
                "source": chunk["source"],
                "chunk_id": chunk["chunk_id"],
                "text": chunk["text"],
                "distance": float(distance),
            }
        )

    return results


if __name__ == "__main__":

    query = "How long do I have to request a refund?"

    results = search_documents(query, top_k=3)

    print("\nSEMANTIC SEARCH RESULTS")
    print("=" * 60)

    print(f"\nQuery: {query}")

    for i, result in enumerate(results, start=1):

        print(f"\nResult {i}")
        print("-" * 60)

        print(f"Source: {result['source']}")
        print(f"Chunk ID: {result['chunk_id']}")
        print(f"Distance: {result['distance']:.4f}")

        print("\nText:")
        print(result["text"])