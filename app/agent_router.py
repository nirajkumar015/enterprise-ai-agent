def route_question(question: str) -> str:
    question = question.lower()

    knowledge_keywords = [
        "refund",
        "shipping",
        "delivery",
        "warranty",
        "policy",
        "troubleshooting",
        "return",
    ]

    analytics_keywords = [
        "customers",
        "orders",
        "sales",
        "products",
        "how many",
        "count",
        "average",
        "total",
        "open tickets",
    ]

    ticket_keywords = [
        "ticket",
        "complaint",
        "issue",
        "priority",
        "sentiment",
        "analyze",
    ]

    # Analytics comes before general ticket matching
    if any(
        keyword in question
        for keyword in analytics_keywords
    ):
        return "ANALYTICS"

    elif any(
        keyword in question
        for keyword in ticket_keywords
    ):
        return "TICKET"

    elif any(
        keyword in question
        for keyword in knowledge_keywords
    ):
        return "KNOWLEDGE"

    else:
        return "UNSUPPORTED"


if __name__ == "__main__":

    test_questions = [
        "How long is the refund policy?",
        "How many customers do we have?",
        "Analyze ticket priority",
        "How many open tickets do we have?",
        "What is the weather today?",
    ]

    for question in test_questions:

        route = route_question(question)

        print(
            f"Question: {question}"
        )

        print(
            f"Route: {route}"
        )

        print("-" * 50)