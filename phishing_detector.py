import sys
print(sys.executable)
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

# Load dataset
data = pd.read_csv("phishing_emails.csv")

# Remove missing values
data = data.dropna()

# Features and labels
X = data["email"]
y = data["label"]

# Dataset information
print("Total Emails:", len(data))
print("\nClass Distribution:")
print(y.value_counts())

# Convert text to numerical features
vectorizer = TfidfVectorizer(
    stop_words="english",
    lowercase=True,
    ngram_range=(1, 2)
)

X_tfidf = vectorizer.fit_transform(X)

# Split data with class balance
X_train, X_test, y_train, y_test = train_test_split(
    X_tfidf,
    y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

# Train model
model = MultinomialNB()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
print("\nAccuracy:")
print(f"{accuracy_score(y_test, y_pred):.2f}")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

# Custom Prediction
print("\nEmail Prediction")
new_email = input("Enter Email Content: ")

email_vector = vectorizer.transform([new_email])

prediction = model.predict(email_vector)[0]
probability = model.predict_proba(email_vector).max()

print("\nPrediction:", prediction.upper())
print("Confidence:", f"{probability * 100:.2f}%")