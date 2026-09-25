import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

from xgboost import XGBClassifier


# =========================================================
# CONFIGURATION
# =========================================================

DATA_PATH = "data/enterprise_priority_dataset.csv"

MODEL_PATH = "ml/priority_model.joblib"
VECTORIZER_PATH = "ml/tfidf_vectorizer.joblib"


# =========================================================
# 1. LOAD DATASET
# =========================================================

df = pd.read_csv(DATA_PATH)

print("\nDATASET")
print("=" * 60)
print(f"Rows: {len(df)}")
print(f"Columns: {len(df.columns)}")

print("\nPriority distribution:")
print(df["priority"].value_counts())


# =========================================================
# 2. CREATE TEXT FEATURE
# =========================================================

df["full_text"] = (
    df["ticket_type"].astype(str)
    + " "
    + df["subject"].astype(str)
    + " "
    + df["description"].astype(str)
)

X = df["full_text"]
y = df["priority"]


# =========================================================
# 3. TRAIN / TEST SPLIT
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

print("\nDATA SPLIT")
print("=" * 60)
print(f"Training samples: {len(X_train)}")
print(f"Testing samples:  {len(X_test)}")


# =========================================================
# 4. TF-IDF
# =========================================================

vectorizer = TfidfVectorizer(
    max_features=10000,
    ngram_range=(1, 2),
    stop_words="english",
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print("\nTF-IDF")
print("=" * 60)
print(f"Training matrix: {X_train_tfidf.shape}")
print(f"Testing matrix:  {X_test_tfidf.shape}")


# =========================================================
# 5. LABEL ENCODING
# =========================================================

label_encoder = LabelEncoder()

y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

print("\nCLASS MAPPING")
print("=" * 60)

for number, label in enumerate(label_encoder.classes_):
    print(f"{number} -> {label}")


# =========================================================
# 6. LOGISTIC REGRESSION
# =========================================================

print("\nTRAINING LOGISTIC REGRESSION...")
print("=" * 60)

logistic_model = LogisticRegression(
    max_iter=2000,
    random_state=42,
)

logistic_model.fit(
    X_train_tfidf,
    y_train
)

logistic_pred = logistic_model.predict(
    X_test_tfidf
)

logistic_accuracy = accuracy_score(
    y_test,
    logistic_pred
)


# =========================================================
# 7. RANDOM FOREST
# =========================================================

print("\nTRAINING RANDOM FOREST...")
print("=" * 60)

random_forest_model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1,
)

random_forest_model.fit(
    X_train_tfidf,
    y_train
)

random_forest_pred = random_forest_model.predict(
    X_test_tfidf
)

random_forest_accuracy = accuracy_score(
    y_test,
    random_forest_pred
)


# =========================================================
# 8. XGBOOST
# =========================================================

print("\nTRAINING XGBOOST...")
print("=" * 60)

xgb_model = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="multi:softprob",
    eval_metric="mlogloss",
    random_state=42,
    n_jobs=-1,
)

xgb_model.fit(
    X_train_tfidf,
    y_train_encoded
)

xgb_pred_encoded = xgb_model.predict(
    X_test_tfidf
)

xgb_pred = label_encoder.inverse_transform(
    xgb_pred_encoded.astype(int)
)

xgb_accuracy = accuracy_score(
    y_test,
    xgb_pred
)


# =========================================================
# 9. MODEL COMPARISON
# =========================================================

print("\n\nMODEL COMPARISON")
print("=" * 60)

comparison = pd.DataFrame(
    {
        "Model": [
            "Logistic Regression",
            "Random Forest",
            "XGBoost",
        ],
        "Accuracy": [
            logistic_accuracy,
            random_forest_accuracy,
            xgb_accuracy,
        ],
    }
)

print(
    comparison.to_string(index=False)
)


# =========================================================
# 10. LOGISTIC REGRESSION REPORT
# =========================================================

print("\n\nLOGISTIC REGRESSION REPORT")
print("=" * 60)

print(
    classification_report(
        y_test,
        logistic_pred
    )
)


# =========================================================
# 11. RANDOM FOREST REPORT
# =========================================================

print("\nRANDOM FOREST REPORT")
print("=" * 60)

print(
    classification_report(
        y_test,
        random_forest_pred
    )
)


# =========================================================
# 12. XGBOOST REPORT
# =========================================================

print("\nXGBOOST REPORT")
print("=" * 60)

print(
    classification_report(
        y_test,
        xgb_pred
    )
)


# =========================================================
# 13. CONFUSION MATRICES
# =========================================================

print("\n\nLOGISTIC REGRESSION CONFUSION MATRIX")
print("=" * 60)

print(
    confusion_matrix(
        y_test,
        logistic_pred
    )
)


print("\nRANDOM FOREST CONFUSION MATRIX")
print("=" * 60)

print(
    confusion_matrix(
        y_test,
        random_forest_pred
    )
)


print("\nXGBOOST CONFUSION MATRIX")
print("=" * 60)

print(
    confusion_matrix(
        y_test,
        xgb_pred
    )
)


# =========================================================
# 14. SAVE LOGISTIC REGRESSION MODEL
# =========================================================

joblib.dump(
    logistic_model,
    MODEL_PATH
)

joblib.dump(
    vectorizer,
    VECTORIZER_PATH
)

print("\n\nMODEL SAVING")
print("=" * 60)

print(f"Model saved to:      {MODEL_PATH}")
print(f"Vectorizer saved to: {VECTORIZER_PATH}")


# =========================================================
# 15. SAMPLE PREDICTIONS
# =========================================================

sample_tickets = [
    (
        "Product inquiry",
        "Product information",
        "I would like some information about this product."
    ),
    (
        "Product issue",
        "Product problem",
        "I am having a problem with my product and need assistance."
    ),
    (
        "Technical issue",
        "Laptop stopped working",
        "My laptop has stopped working and I need it fixed immediately."
    ),
    (
        "Critical technical issue",
        "Critical system failure",
        "Emergency! The system has completely failed and critical business operations are blocked."
    ),
]


sample_text = [
    ticket_type + " " + subject + " " + description
    for ticket_type, subject, description in sample_tickets
]


sample_tfidf = vectorizer.transform(
    sample_text
)


logistic_samples = logistic_model.predict(
    sample_tfidf
)


print("\n\nSAMPLE PREDICTIONS")
print("=" * 60)

for i, ticket in enumerate(sample_tickets):

    print("\nTicket:")
    print(ticket[2])

    print(
        f"Predicted Priority: {logistic_samples[i]}"
    )


# =========================================================
# 16. PREDICTION FUNCTION
# =========================================================

def predict_priority(
    ticket_type,
    subject,
    description
):
    """
    Predict ticket priority using the trained
    Logistic Regression model.
    """

    text = (
        str(ticket_type)
        + " "
        + str(subject)
        + " "
        + str(description)
    )

    text_tfidf = vectorizer.transform(
        [text]
    )

    prediction = logistic_model.predict(
        text_tfidf
    )[0]

    probabilities = logistic_model.predict_proba(
        text_tfidf
    )[0]

    probability_map = dict(
        zip(
            logistic_model.classes_,
            probabilities
        )
    )

    confidence = float(
        max(probabilities)
    )

    return {
        "priority": prediction,
        "confidence": confidence,
        "probabilities": {
            key: float(value)
            for key, value in probability_map.items()
        },
    }


# =========================================================
# 17. FUNCTION TEST
# =========================================================

if __name__ == "__main__":

    result = predict_priority(
        "Technical issue",
        "Laptop stopped working",
        "My laptop suddenly stopped working and I need immediate assistance."
    )

    print("\n\nFUNCTION TEST")
    print("=" * 60)

    print(result)


# =========================================================
# IMPORTANT NOTE
# =========================================================

print("\n\nIMPORTANT")
print("=" * 60)

print(
    "These results are from a controlled synthetic "
    "development dataset."
)

print(
    "They should NOT be interpreted as real-world "
    "production performance."
)