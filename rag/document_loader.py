#   What this does

# Our five files:

# company_docs/
#     ↓
# load_documents()
#     ↓
# Full document text
#     ↓
# chunk_text()
#     ↓
# Smaller piecesWhat this does

# Our five files:

# company_docs/
#     ↓
# load_documents()
#     ↓
# Full document text
#     ↓
# chunk_text()
#     ↓
# Smaller pieces     

# For example:

# refund_policy.txt

# might become:

# Chunk 0 → Refund policy + eligibility...
# Chunk 1 → Damaged products + processing...
# Chunk 2 → Non-refundable cases...



from pathlib import Path


# =========================================================
# CONFIGURATION
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DOCUMENTS_PATH = BASE_DIR / "docs" / "company_docs"


# =========================================================
# LOAD DOCUMENTS
# =========================================================

def load_documents():
    """
    Load all .txt files from the company_docs directory.
    """

    documents = []

    for file_path in DOCUMENTS_PATH.glob("*.txt"):

        text = file_path.read_text(
            encoding="utf-8"
        )

        documents.append(
            {
                "text": text,
                "source": file_path.name,
            }
        )

    return documents


# =========================================================
# CHUNK DOCUMENT
# =========================================================

def chunk_text(
    text,
    chunk_size=500,
    overlap=100
):
    """
    Split document text into overlapping chunks.
    """

    chunks = []

    start = 0

    while start < len(text):

        end = start + chunk_size

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


# =========================================================
# LOAD + CHUNK ALL DOCUMENTS
# =========================================================

def load_and_chunk_documents():

    documents = load_documents()

    chunks = []

    for document in documents:

        document_chunks = chunk_text(
            document["text"]
        )

        for index, chunk in enumerate(
            document_chunks
        ):

            chunks.append(
                {
                    "text": chunk,
                    "source": document["source"],
                    "chunk_id": index,
                }
            )

    return chunks


# =========================================================
# TEST
# =========================================================

if __name__ == "__main__":

    print("\nDOCUMENT PATH")
    print("=" * 60)
    print(DOCUMENTS_PATH)

    chunks = load_and_chunk_documents()

    print("\nDOCUMENT INGESTION")
    print("=" * 60)

    print(
        f"Total chunks created: {len(chunks)}"
    )

    print("\nSample chunks:")
    print("-" * 60)

    for chunk in chunks[:5]:

        print(
            f"\nSource: {chunk['source']}"
        )

        print(
            f"Chunk ID: {chunk['chunk_id']}"
        )

        print(
            f"Text: {chunk['text']}"
        )










