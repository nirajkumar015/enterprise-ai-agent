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


def create_vector_store():
    """
    Load document chunks, convert them into embeddings,
    and store the embeddings in a FAISS index.
    """

    chunks = load_and_chunk_documents()

    if not chunks:
        raise ValueError("No document chunks found.")

    # Extract text from chunks
    texts = [chunk["text"] for chunk in chunks]

    # Generate embeddings
    embeddings = model.encode(
        texts,
        show_progress_bar=True
    )

    # Convert to NumPy float32
    embeddings = np.array(embeddings).astype("float32")

    # Create FAISS index
    embedding_dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(embedding_dimension)

    # Add embeddings to FAISS
    index.add(embeddings)

    # Save FAISS index
    faiss.write_index(index, str(VECTOR_STORE_PATH))

    print("\nVECTOR STORE CREATED")
    print("=" * 60)
    print(f"Documents chunks: {len(chunks)}")
    print(f"Embedding dimension: {embedding_dimension}")
    print(f"FAISS vectors: {index.ntotal}")
    print(f"Saved to: {VECTOR_STORE_PATH}")


if __name__ == "__main__":
    create_vector_store()