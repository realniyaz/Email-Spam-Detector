# prediction.py
import pickle
import os
from src.preprocessing import preprocess_text

# Define paths
MODEL_PATH = os.path.join("models", "spam_classifier.pkl")
TFIDF_PATH = os.path.join("models", "tfidf.pkl")

try:
    with open(MODEL_PATH, 'rb') as file:
        model = pickle.load(file)
    with open(TFIDF_PATH, 'rb') as file:
        tfidf = pickle.load(file)
except FileNotFoundError:
    print("Error: Model or TF-IDF vectorizer files not found. Train the model first.")
    exit() # Exit the script if the model isn't found

def predict_spam(text):
    preprocessed_text = preprocess_text(text)
    text_vectorized = tfidf.transform([preprocessed_text])
    prediction = model.predict(text_vectorized)
    return prediction[0]  # Return the prediction string ("spam" or "ham")

if __name__ == "__main__":
    while True:
        user_input = input("Enter an email to check for spam (or type 'exit'): ")
        if user_input.lower() == 'exit':
            break

        prediction = predict_spam(user_input)
        print(f"Prediction: {prediction}")
