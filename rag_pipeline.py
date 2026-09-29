from rag.retriever import search_documents
from rag.llm import generate_response


def build_rag_prompt(question: str, retrieved_documents: list) -> str:
    """
    Build a prompt that provides the LLM with
    evidence retrieved from the company knowledge base.
    """

    evidence_parts = []

    for i, document in enumerate(retrieved_documents, start=1):
        evidence_parts.append(
            f"""
Source {i}: {document['source']}
Chunk ID: {document['chunk_id']}

{document['text']}
"""
        )

    evidence = "\n".join(evidence_parts)

    prompt = f"""
You are an enterprise customer support assistant.

Answer the user's question using ONLY the information
provided in the retrieved company documents below.

Rules:
1. Do not invent or assume information.
2. If the retrieved documents do not contain enough
   information to answer the question, clearly say:
   "The available company documents do not contain
   enough information to answer this question."
3. Give a concise and helpful answer.
4. Mention the relevant source document when possible.
5. Do not use outside knowledge.

Retrieved company documents:
----------------------------

{evidence}

----------------------------

User question:
{question}

Answer:
"""

    return prompt


def answer_question(question: str, top_k: int = 3):
    """
    Complete RAG pipeline:

    1. Retrieve relevant documents.
    2. Build a prompt using retrieved evidence.
    3. Send the evidence and question to the LLM.
    4. Return the answer and source information.
    """

    # Step 1: Retrieve relevant documents
    retrieved_documents = search_documents(
        question,
        top_k=top_k
    )

    if not retrieved_documents:
        return {
            "answer": (
                "I could not find relevant information "
                "in the company knowledge base."
            ),
            "sources": []
        }

    # Step 2: Build RAG prompt
    prompt = build_rag_prompt(
        question,
        retrieved_documents
    )

    # Step 3: Generate answer using the LLM
    answer = generate_response(prompt)

    # Step 4: Prepare source information
    sources = []

    for document in retrieved_documents:
        sources.append(
            {
                "source": document["source"],
                "chunk_id": document["chunk_id"],
                "distance": document["distance"],
            }
        )

    return {
        "answer": answer,
        "sources": sources,
    }


if __name__ == "__main__":

    questions = [
        "How long do I have to request a refund?",
        "How long does standard shipping take?",
        "What should I do if my laptop is not turning on?",
        "What is the company's employee vacation policy?"
    ]

    for question in questions:

        print("\n" + "=" * 70)
        print("QUESTION")
        print("=" * 70)
        print(question)

        result = answer_question(question)

        print("\nANSWER")
        print("-" * 70)
        print(result["answer"])

        print("\nSOURCES")
        print("-" * 70)

        for source in result["sources"]:
            print(
                f"- {source['source']} "
                f"(chunk {source['chunk_id']}, "
                f"distance {source['distance']:.4f})"
            )