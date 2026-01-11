import re
import joblib

def clean_text(text):
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"[^a-zA-Z0-9\s']", " ", text)
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text

model = joblib.load("sentiment_model.pkl")

print("🎬 Movie Review Sentiment Demo")
print("Type a movie review (or 'exit' to quit)\n")

while True:
    user_input = input("Review: ")
    if user_input.lower() == "exit":
        break

    cleaned = clean_text(user_input)
    pred = model.predict([cleaned])[0]
    prob = model.predict_proba([cleaned])[0].max()

    sentiment = "Positive 😊" if pred == 1 else "Negative 😞"
    print(f"Prediction: {sentiment} (confidence: {prob:.2f})\n")