import re
import pandas as pd
import torch
from transformers import pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

DATA_PATH = "C:\\Users\\aniq\\OneDrive\\Desktop\\Movie Review Sentiment Analysis\\IMDB Dataset.csv"

def clean_text(text: str) -> str:
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# Load data
df = pd.read_csv(DATA_PATH).dropna(subset=["review", "sentiment"]).copy()
df["review"] = df["review"].astype(str).apply(clean_text)
df["label"] = df["sentiment"].map({"negative": 0, "positive": 1})

# Train/test split
train_df, test_df = train_test_split(
    df[["review", "label"]],
    test_size=0.2,
    random_state=42,
    stratify=df["label"]
)

print("Loading pre-trained DistilBERT classifier...")
# Use pre-trained sentiment classifier (no fine-tuning, just inference)
classifier = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english", device=0 if torch.cuda.is_available() else -1)

print(f"Classifying {len(test_df)} test samples...")
preds = []
for i, review in enumerate(test_df["review"].values):
    if i % 500 == 0:
        print(f"  Progress: {i}/{len(test_df)}")
    result = classifier(review[:512])  # Limit to 512 tokens
    # Map: NEGATIVE=0, POSITIVE=1
    pred = 1 if result[0]["label"] == "POSITIVE" else 0
    score = result[0]["score"]
    preds.append((pred, score))

y_true = test_df["label"].values
y_pred = [p[0] for p in preds]
confidences = [p[1] for p in preds]

acc = accuracy_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

print("\n✅ Pre-trained DistilBERT Results (No Fine-tuning)")
print(f"Accuracy: {acc:.4f}")
print(f"F1-score : {f1:.4f}\n")

print("Confusion Matrix:")
print(confusion_matrix(y_true, y_pred), "\n")

print("Classification Report:")
print(classification_report(y_true, y_pred, target_names=["negative", "positive"]))

# Compare with baseline
print("\n" + "="*50)
print("COMPARISON: Baseline vs Pre-trained BERT")
print("="*50)
print(f"{'Metric':<20} {'Baseline':<15} {'DistilBERT':<15}")
print("-"*50)
print(f"{'Accuracy':<20} {'0.9025':<15} {acc:.4f}")
print(f"{'F1-score':<20} {'0.9034':<15} {f1:.4f}")

# Show some examples
print("\n\nSample Predictions:")
for i in range(5):
    idx = test_df.index[i]
    review = test_df.loc[idx, "review"][:100] + "..."
    true_label = "Negative" if y_true[i] == 0 else "Positive"
    pred_label = "Negative" if y_pred[i] == 0 else "Positive"
    confidence = confidences[i]
    print(f"\nReview: {review}")
    print(f"True: {true_label}, Predicted: {pred_label}, Confidence: {confidence:.4f}")
