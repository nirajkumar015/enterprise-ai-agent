from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# 1. Features: study hours
X = [[1], [2], [3], [5], [6], [8]]

# 2. Labels: 0 = Fail, 1 = Pass
y = [0, 0, 0, 1, 1, 1]

# 3. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.33,
    random_state=42,
    stratify=y
)

# 4. Create the model
model = DecisionTreeClassifier(random_state=42)

# 5. Train the model
model.fit(X_train, y_train)

# 6. Predict results for test data
y_pred = model.predict(X_test)

# 7. Evaluate the model
accuracy = accuracy_score(y_test, y_pred)

print("Actual results:", y_test)
print("Predicted results:", y_pred)
print("Accuracy:", accuracy)

# 8. Predict for a new example
new_student = [[4]]
prediction = model.predict(new_student)[0]

if prediction == 1:
    print("Prediction for 4 study hours: Pass")
else:
    print("Prediction for 4 study hours: Fail")