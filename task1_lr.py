import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

# Load data
train = pd.read_json("train.jsonl", lines=True)
val = pd.read_json("val.jsonl", lines=True)

# Labels
train["label"] = train["tags"].apply(lambda x: x[0])
val["label"] = val["tags"].apply(lambda x: x[0])

# Text features
train["text"] = (
    train["postText"].astype(str)
    + " "
    + train["targetTitle"].astype(str)
)

val["text"] = (
    val["postText"].astype(str)
    + " "
    + val["targetTitle"].astype(str)
)

# TF-IDF
vectorizer = TfidfVectorizer(
    max_features=10000,
    stop_words="english"
)
X_train = vectorizer.fit_transform(train["text"])
X_val = vectorizer.transform(val["text"])

# Model
model = LogisticRegression(
    max_iter=2000,
    random_state=42,
    class_weight="balanced"
)
model.fit(X_train, train["label"])

# Validation
predictions = model.predict(X_val)

f1 = f1_score(
    val["label"],
    predictions,
    average="macro"
)

print(f"Validation Macro F1: {f1:.4f}")

# Load test data
test = pd.read_json("test.jsonl", lines=True)

# Build test text
test["text"] = (
    test["postText"].astype(str)
    + " "
    + test["targetTitle"].astype(str)
)

# Transform test data
X_test = vectorizer.transform(test["text"])

# Predict
test_predictions = model.predict(X_test)

# Create submission
submission = pd.DataFrame({
    "id": test["id"],
    "spoilerType": test_predictions
})

submission.to_csv("task1_lr_submission.csv", index=False)

print("Submission file saved.")