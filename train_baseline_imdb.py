import re
import json
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

DATA_PATH = "C:\\Users\\aniq\\OneDrive\\Desktop\\Movie Review Sentiment Analysis\\IMDB Dataset.csv"
def clean_text(text: str) -> str:
    # remove HTML
    text = re.sub(r"<.*?>", " ", text)
    # keep letters/numbers/basic punctuation, remove weird symbols
    text = re.sub(r"[^a-zA-Z0-9\s']", " ", text)
    # normalize spaces + lowercase
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text

def main():
    df = pd.read_csv(DATA_PATH)

    # Expect columns: review, sentiment
    df = df.dropna(subset=["review", "sentiment"]).copy()
    df["review"] = df["review"].astype(str).apply(clean_text)
    df["label"] = df["sentiment"].map({"negative": 0, "positive": 1})

    X = df["review"].values
    y = df["label"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=50000,
            ngram_range=(1, 2),
            stop_words="english"
        )),
        ("clf", LogisticRegression(max_iter=200))
    ])

    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds)

    print("✅ Baseline Results (TF-IDF + Logistic Regression)")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1-score : {f1:.4f}\n")

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, preds), "\n")

    print("Classification Report:")
    print(classification_report(y_test, preds, target_names=["negative", "positive"]))

    # Save results as JSON
    results = {
        "model": "baseline",
        "accuracy": float(acc),
        "f1": float(f1)
    }
    with open("results_baseline.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\n✅ Results saved to results_baseline.json")

if __name__ == "__main__":
    main()