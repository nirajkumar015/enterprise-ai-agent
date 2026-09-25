import os
import joblib


# =========================================================
# MODEL PATHS
# =========================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "ml",
    "priority_model.joblib"
)

VECTORIZER_PATH = os.path.join(
    BASE_DIR,
    "ml",
    "tfidf_vectorizer.joblib"
)


# =========================================================
# LOAD MODEL AND VECTORIZER
# =========================================================

model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)


# =========================================================
# PREDICT PRIORITY
# =========================================================

def predict_priority(
    ticket_type: str,
    subject: str,
    description: str
):
    """
    Predict the priority of a support ticket.

    Returns:
        priority
        confidence
        probabilities
    """

    text = (
        str(ticket_type)
        + " "
        + str(subject)
        + " "
        + str(description)
    )

    # Convert text into TF-IDF features
    text_tfidf = vectorizer.transform([text])

    # Predict priority
    prediction = model.predict(text_tfidf)[0]

    # Get probability for each class
    probabilities = model.predict_proba(text_tfidf)[0]

    probability_map = dict(
        zip(
            model.classes_,
            probabilities
        )
    )

    # Highest probability
    confidence = float(max(probabilities))

    return {
        "priority": prediction,
        "confidence": confidence,
        "probabilities": {
            key: float(value)
            for key, value in probability_map.items()
        }
    }


# =========================================================
# TEST
# =========================================================

if __name__ == "__main__":

    result = predict_priority(
        ticket_type="Technical issue",
        subject="Laptop stopped working",
        description=(
            "My laptop suddenly stopped working "
            "and I need immediate assistance."
        )
    )

    print("\nPRIORITY PREDICTION")
    print("=" * 50)

    print(f"Priority: {result['priority']}")
    print(f"Confidence: {result['confidence']:.2%}")

    print("\nProbabilities:")

    for priority, probability in result["probabilities"].items():
        print(
            f"{priority}: {probability:.2%}"
        )