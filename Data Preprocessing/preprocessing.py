import pandas as pd
import re
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords', quiet=True)  # Download stopwords quietly

def preprocess_text(text):
    """Preprocesses text data."""
    if pd.isna(text):
        return ""
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower()
    words = text.split()
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    return " ".join(words)

def load_and_preprocess_data(filepath):
    """Loads data from CSV and preprocesses the text."""
    try:
        df = pd.read_csv(filepath, encoding='latin1')
        df = df.iloc[:,:2]
        df.columns = ['Category','Message']
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return None
    except pd.errors.ParserError:
        print(f"Error: Could not parse CSV file at {filepath}. Check the file format.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None

    df['Message'] = df['Message'].apply(preprocess_text)
    return df

if __name__ == "__main__":
    # Example usage when running the script directly
    filepath = r"C:\Users\Dell\Desktop\Email Spam Detector\email.csv"  # Path to your CSV file
    df = load_and_preprocess_data(filepath)
    if df is not None:
        print("Data loaded and preprocessed successfully.")
        print(df.head())
    else:
        print("Data loading failed.")
