def analyze_ticket(text: str):
    text_lower = text.lower()

    negative_words = [
        "bad",
        "broken",
        "wrong",
        "late",
        "delayed",
        "not working",
        "problem",
        "issue",
        "failed",
        "never received",
    ]

    if any(word in text_lower for word in negative_words):
        sentiment = "negative"
    else:
        sentiment = "neutral"

    if any(word in text_lower for word in ["delivery", "delivered", "shipping", "arrive", "late", "delayed"]):
        category = "shipping"
    elif any(word in text_lower for word in ["payment", "refund", "billing", "charged"]):
        category = "billing"
    elif any(word in text_lower for word in ["laptop", "mouse", "keyboard", "headphones", "product"]):
        category = "product"
    elif any(word in text_lower for word in ["login", "password", "account"]):
        category = "account"
    else:
        category = "other"

    return {
        "sentiment": sentiment,
        "category": category,
    }


if __name__ == "__main__":
    ticket = "My order was delayed and I have still not received it."

    result = analyze_ticket(ticket)

    print(result)