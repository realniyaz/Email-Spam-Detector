# Email-Spam-Detector

This project implements a machine learning model to classify emails as spam or ham (non-spam). It uses a Multinomial Naive Bayes classifier trained on TF-IDF vectorized email text.

## Features

*   Trains a Multinomial Naive Bayes model.
*   Uses TF-IDF (Term Frequency-Inverse Document Frequency) for feature extraction.
*   Includes a script for interactive spam/ham prediction.
*   Saves the trained model and TF-IDF vectorizer for later use.
*   Handles potential errors during data loading, model training, and file saving/loading.

## Requirements

*   Python 3.x
*   `pandas`
*   `scikit-learn`
*   `matplotlib` (optional - for confusion matrix visualization)
*   `seaborn` (optional - for confusion matrix visualization)
*   `nltk`
*   `pickle`

You can install these using:

```bash
pip install -r requirements.txt

```

## Project Structure
Email-Spam-Detection/
├── data/
│   └── email.csv       # Your email data (CSV format)

├── models/
│   ├── spam_classifier.pkl # Saved trained model

│   └── tfidf.pkl          # Saved TF-IDF vectorizer

├── notebooks/
│   └── model_training.ipynb # Notebook for initial exploration (optional)

├── src/
│   └── preprocessing.py # Data preprocessing functions

├── model_training.py   # Script to train and save the model

├── prediction.py       # Script for interactive prediction

├── requirements.txt    # Project dependencies

└── README.md           # This file

## Code Explanation

1. src/preprocessing.py: Contains the load_and_preprocess_data() and preprocess_text() functions for loading data from the CSV file and cleaning and preprocessing the email text (removing non-alphanumeric characters, converting to lowercase, removing stop words).

2. model_training.py: Trains the model, evaluates it, and saves the trained model and TF-IDF vectorizer.

3. prediction.py: Loads the saved model and vectorizer and provides an interactive interface for making predictions on new emails.

## Contributing
Contributions are welcome! Please feel free to open issues or submit pull requests.
