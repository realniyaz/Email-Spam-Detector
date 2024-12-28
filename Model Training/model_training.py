import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from src.preprocessing import load_and_preprocess_data

# Load and preprocess the data
df = load_and_preprocess_data(r"C:\Users\Dell\Desktop\Email Spam Detector\email.csv")
if df is None:
    print("Failed to load data, exiting.")
else:
    print("Data loaded successfully.")

# Feature Extraction and Model Training (only if data loading was successful)
if df is not None:
    tfidf = TfidfVectorizer()
    X = tfidf.fit_transform(df['Message'])
    y = df['Category']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = MultinomialNB()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    # Evaluate the model
    print(classification_report(y_test, y_pred))

    # Additional Evaluation Metrics and Visualization
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy}")

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=model.classes_, yticklabels=model.classes_)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.show()

    #Example of prediction
    new_emails = ["Congratulations you have won a lottery of 100000$", "Hello how are you doing today"]
    new_emails_transformed = tfidf.transform(new_emails)
    predictions = model.predict(new_emails_transformed)
    print(predictions)
