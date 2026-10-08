from sqlalchemy import text

from app.database import SessionLocal


def get_query(question: str):
    question = question.lower()

    if "customers" in question:
        return "SELECT COUNT(*) AS total FROM customers;"

    elif "orders" in question:
        return "SELECT COUNT(*) AS total FROM orders;"

    elif "products" in question:
        return "SELECT COUNT(*) AS total FROM products;"

    elif "open tickets" in question:
        return """
        SELECT COUNT(*) AS total
        FROM support_tickets
        WHERE status = 'Open';
        """

    elif "tickets" in question:
        return "SELECT COUNT(*) AS total FROM support_tickets;"

    else:
        return None


def execute_analytics(question: str):

    query = get_query(question)

    if query is None:
        return {
            "success": False,
            "answer": "I don't have a supported analytics query for that question."
        }

    db = SessionLocal()

    try:
        result = db.execute(text(query))

        row = result.fetchone()

        total = row[0]

        return {
            "success": True,
            "answer": f"The result is {total}.",
            "query": query.strip()
        }

    finally:
        db.close()


if __name__ == "__main__":

    test_questions = [
        "How many customers do we have?",
        "How many orders do we have?",
        "How many products do we have?",
        "How many tickets do we have?",
        "How many open tickets do we have?",
    ]

    for question in test_questions:

        print("\nQuestion:")
        print(question)

        result = execute_analytics(question)

        print("\nResult:")
        print(result)