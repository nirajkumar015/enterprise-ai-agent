from sklearn.feature_extraction.text import TfidfVectorizer

# Example support tickets
tickets = [
    "My laptop is broken",
    "My laptop is overheating",
    "The delivery is delayed",
    "My order delivery is late",
]

# Convert text into numerical features
vectorizer = TfidfVectorizer()

X = vectorizer.fit_transform(tickets)

# Display the learned vocabulary
print("Vocabulary:")
print(vectorizer.get_feature_names_out())

# Display the numerical representation
print("\nTF-IDF matrix shape:")
print(X.shape)

print("\nTF-IDF values:")
print(X.toarray().round(3))