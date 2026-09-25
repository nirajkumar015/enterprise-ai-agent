# What this code does

# Our pipeline becomes:

# Company Documents
#        ↓
# document_loader.py
#        ↓
# 15 chunks
#        ↓
# embeddings.py
#        ↓
# Sentence Transformer
#        ↓
# 15 numerical vectors

# The important line is:

# embeddings = model.encode(texts)

# It converts all our document chunks into vectors.





from sentence_transformers import SentenceTransformer

from document_loader import load_and_chunk_documents


# =========================================================
# EMBEDDING MODEL
# =========================================================

MODEL_NAME = "all-MiniLM-L6-v2"

model = SentenceTransformer(MODEL_NAME)


# =========================================================
# CREATE EMBEDDINGS
# =========================================================

def create_embeddings(chunks):
    """
    Convert document chunks into numerical vectors.
    """

    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    embeddings = model.encode(
        texts,
        show_progress_bar=True
    )

    return embeddings


# =========================================================
# TEST
# =========================================================

if __name__ == "__main__":

    chunks = load_and_chunk_documents()

    embeddings = create_embeddings(chunks)

    print("\nEMBEDDING TEST")
    print("=" * 60)

    print(
        f"Number of chunks: {len(chunks)}"
    )

    print(
        f"Embedding shape: {embeddings.shape}"
    )

    print("\nFirst embedding:")
    print(embeddings[0])



