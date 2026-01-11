import re
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

DATA_PATH = "IMDB Dataset.csv"

def clean_text(text):
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"[^a-zA-Z0-9\s']", " ", text)
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text

df = pd.read_csv(DATA_PATH)
df["review"] = df["review"].astype(str).apply(clean_text)
df["label"] = df["sentiment"].map({"negative": 0, "positive": 1})

X_train, X_test, y_train, y_test = train_test_split(
    df["review"], df["label"],
    test_size=0.2, random_state=42, stratify=df["label"]
)

model = Pipeline([
    ("tfidf", TfidfVectorizer(stop_words="english", ngram_range=(1,2))),
    ("clf", LogisticRegression(max_iter=200))
])

model.fit(X_train, y_train)

# save model
joblib.dump(model, "sentiment_model.pkl")

print("✅ Model trained and saved as sentiment_model.pkl")