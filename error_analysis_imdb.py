import re
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

DATA_PATH = "C:\\Users\\aniq\\OneDrive\\Desktop\\Movie Review Sentiment Analysis\\IMDB Dataset.csv"

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
preds = model.predict(X_test)

# Find wrong predictions
errors = X_test[preds != y_test]
true_labels = y_test[preds != y_test]
pred_labels = preds[preds != y_test]

error_df = pd.DataFrame({
    "review": errors.values,
    "true_label": true_labels.values,
    "predicted_label": pred_labels
})

# Get confidence scores (probability of predicted class)
probs = model.predict_proba(X_test[preds != y_test])
error_df["confidence"] = [max(p) for p in probs]

# Categorize errors
def categorize_error(review, true_label, pred_label):
    review_lower = review.lower()
    
    # Sarcasm indicators
    sarcasm_words = ["yeah right", "sure", "great", "wonderful", "fantastic", "love", "perfect"]
    if any(word in review_lower for word in sarcasm_words) and true_label == 0 and pred_label == 1:
        return "Sarcasm (False Positive)"
    if any(word in review_lower for word in sarcasm_words) and true_label == 1 and pred_label == 0:
        return "Sarcasm (False Negative)"
    
    # Negation handling
    negation_words = ["not good", "not great", "not worth", "dont recommend", "didnt like", "not funny"]
    if any(word in review_lower for word in negation_words):
        return "Negation Handling"
    
    # Mixed sentiment
    if "but" in review_lower or "however" in review_lower:
        return "Mixed Sentiment"
    
    return "Other"

error_df["error_type"] = error_df.apply(
    lambda row: categorize_error(row["review"], row["true_label"], row["predicted_label"]),
    axis=1
)

print("Sample Misclassified Reviews:\n")
print(error_df.head(10))

print("\n\nError Type Distribution:")
print(error_df["error_type"].value_counts())

print("\n\nMisclassification Summary:")
print(f"Total Errors: {len(error_df)}")
print(f"False Positives (Negative -> Positive): {sum((error_df['true_label']==0) & (error_df['predicted_label']==1))}")
print(f"False Negatives (Positive -> Negative): {sum((error_df['true_label']==1) & (error_df['predicted_label']==0))}")

print("\n\nLowest Confidence Predictions (Most uncertain):")
print(error_df.nsmallest(5, "confidence")[["review", "true_label", "predicted_label", "confidence", "error_type"]])