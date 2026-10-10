from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Example training tickets
tickets = [
    "urgent laptop failure needs immediate repair",
    "critical system failure needs urgent support",
    "immediate help required for broken laptop",
    "please share product documentation",
    "normal question about product features",
    "request for general product information",
]

# Labels corresponding to the training tickets
labels = [
    "URGENT",
    "URGENT",
    "URGENT",
    "NORMAL",
    "NORMAL",
    "NORMAL",
]

# Convert text into numerical features
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(tickets)

# Create and train the classifier
model = LogisticRegression(random_state=42)
model.fit(X, labels)

# Predict for a new ticket using the SAME vectorizer
new_ticket = ["urgent failure needs immediate support"]
new_ticket_vector = vectorizer.transform(new_ticket)

prediction = model.predict(new_ticket_vector)[0]
probabilities = model.predict_proba(new_ticket_vector)[0]

print("New ticket:", new_ticket[0])
print("Predicted class:", prediction)
print("\nClass probabilities:")

for label, probability in zip(model.classes_, probabilities):
    print(f"{label}: {probability:.2%}")