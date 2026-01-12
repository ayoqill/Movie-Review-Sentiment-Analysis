import re
import json
import numpy as np
import pandas as pd

from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)

DATA_PATH = "C:\\Users\\aniq\\OneDrive\\Desktop\\Movie Review Sentiment Analysis\\IMDB Dataset.csv"

def clean_text(text: str) -> str:
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds)
    }

def main():
    df = pd.read_csv(DATA_PATH).dropna(subset=["review", "sentiment"]).copy()
    df["review"] = df["review"].astype(str).apply(clean_text)
    df["label"] = df["sentiment"].map({"negative": 0, "positive": 1})

    # Train/test split (same style as baseline)
    train_df, test_df = train_test_split(
        df[["review", "label"]],
        test_size=0.2,
        random_state=42,
        stratify=df["label"]
    )

    train_ds = Dataset.from_pandas(train_df.reset_index(drop=True))
    test_ds  = Dataset.from_pandas(test_df.reset_index(drop=True))

    model_name = "distilbert-base-uncased"  # lighter & faster than bert-base
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize(batch):
        return tokenizer(batch["review"], truncation=True, padding="max_length", max_length=128, return_tensors=None)

    train_ds = train_ds.map(tokenize, batched=True)
    test_ds  = test_ds.map(tokenize, batched=True)

    train_ds = train_ds.remove_columns(["review"])
    test_ds  = test_ds.remove_columns(["review"])
    train_ds.set_format("torch")
    test_ds.set_format("torch")

    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    args = TrainingArguments(
        output_dir="bert_imdb_out",
        eval_strategy="epoch",
        save_strategy="epoch",
        num_train_epochs=1,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        learning_rate=2e-5,
        weight_decay=0.01,
        logging_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    trainer.train()

    # Final evaluation + detailed report
    preds = trainer.predict(test_ds)
    y_true = preds.label_ids
    y_pred = np.argmax(preds.predictions, axis=1)

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    print("\n✅ BERT Results (DistilBERT Fine-tuned)")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1-score : {f1:.4f}\n")

    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred), "\n")

    print("Classification Report:")
    print(classification_report(y_true, y_pred, target_names=["negative", "positive"]))

    # Save results as JSON
    results = {
        "model": "bert_finetuned",
        "accuracy": float(acc),
        "f1": float(f1)
    }
    with open("results_bert_finetuned.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\n✅ Results saved to results_bert_finetuned.json")

    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred), "\n")

    print("Classification Report:")
    print(classification_report(y_true, y_pred, target_names=["negative", "positive"]))

if __name__ == "__main__":
    main()