import re
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AdamW,
    get_linear_schedule_with_warmup
)

DATA_PATH = "C:\\Users\\aniq\\OneDrive\\Desktop\\Movie Review Sentiment Analysis\\IMDB Dataset.csv"

def clean_text(text: str) -> str:
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def main():
    # Load and prepare data
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

    # Load tokenizer and model
    model_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    # Tokenize
    def tokenize_batch(texts, labels):
        encodings = tokenizer(
            list(texts),
            truncation=True,
            padding=True,
            max_length=256,
            return_tensors="pt"
        )
        return encodings, torch.tensor(list(labels), dtype=torch.long)

    train_encodings, train_labels = tokenize_batch(train_df["review"].values, train_df["label"].values)
    test_encodings, test_labels = tokenize_batch(test_df["review"].values, test_df["label"].values)

    # Create DataLoaders
    train_dataset = TensorDataset(
        train_encodings["input_ids"],
        train_encodings["attention_mask"],
        train_labels
    )
    test_dataset = TensorDataset(
        test_encodings["input_ids"],
        test_encodings["attention_mask"],
        test_labels
    )

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    # Setup device and training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=2e-5)
    total_steps = len(train_loader) * 2  # 2 epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    # Training loop
    model.train()
    for epoch in range(2):
        total_loss = 0
        for batch_idx, (input_ids, attention_mask, labels) in enumerate(train_loader):
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()

            loss.backward()
            optimizer.step()
            scheduler.step()

            if (batch_idx + 1) % 100 == 0:
                print(f"Epoch {epoch+1}/2, Batch {batch_idx+1}/{len(train_loader)}, Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(train_loader)
        print(f"\nEpoch {epoch+1} completed. Avg Loss: {avg_loss:.4f}\n")

    # Evaluation
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for input_ids, attention_mask, labels in test_loader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1).cpu().numpy()

            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    print("\n✅ BERT Results (DistilBERT Fine-tuned)")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1-score : {f1:.4f}\n")

    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred), "\n")

    print("Classification Report:")
    print(classification_report(y_true, y_pred, target_names=["negative", "positive"]))

    # Save model
    model.save_pretrained("./bert_imdb_model")
    tokenizer.save_pretrained("./bert_imdb_model")
    print("\n✅ Model saved to ./bert_imdb_model")

if __name__ == "__main__":
    main()
