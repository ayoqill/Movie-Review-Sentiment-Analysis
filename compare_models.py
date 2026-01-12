import json
import pandas as pd

files = [
    "results_baseline.json",
    "results_bert_pretrained.json",
    "results_bert_finetuned.json"
]

rows = []

for file in files:
    with open(file, "r") as f:
        rows.append(json.load(f))

df = pd.DataFrame(rows)
print("\nModel Comparison\n")
print(df)
