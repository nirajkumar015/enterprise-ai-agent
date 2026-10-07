from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from .document_loader import load_and_chunk_documents


BASE_DIR = Path(__file__).resolve().parent.parent
VECTOR_STORE_PATH = BASE_DIR / "rag" / "vector_store.index"

MODEL_NAME = "all-MiniLM-L6-v2"

# Maximum acceptable FAISS distance
MAX_DISTANCE = 1.2

model = SentenceTransformer(MODEL_NAME)


def load_vector_store():
    if not VECTOR_STORE_PATH.exists():
        raise FileNotFoundError(
            f"Vector store not found: {VECTOR_STORE_PATH}"
        )

    return faiss.read_index(str(VECTOR_STORE_PATH))


def search_documents(query: str, top_k: int = 3):
    index = load_vector_store()

    chunks = load_and_chunk_documents()

    query_embedding = model.encode([query])

    query_embedding = np.array(query_embedding).astype("float32")

    distances, indices = index.search(
        query_embedding,
        top_k
    )

    results = []

    for distance, index_id in zip(
        distances[0],
        indices[0]
    ):
        if index_id == -1:
            continue

        # Reject results that are not sufficiently relevant
        if distance > MAX_DISTANCE:
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

    query = "What is the company's employee vacation policy?"

    results = search_documents(
        query,
        top_k=3
    )

    print("\nSEMANTIC SEARCH RESULTS")
    print("=" * 60)

    print(f"\nQuery: {query}")

    if not results:
        print("\nNo sufficiently relevant documents found.")

    for i, result in enumerate(results, start=1):

        print(f"\nResult {i}")
        print("-" * 60)

        print(f"Source: {result['source']}")
        print(f"Chunk ID: {result['chunk_id']}")
        print(f"Distance: {result['distance']:.4f}")

        print("\nText:")
        print(result["text"])