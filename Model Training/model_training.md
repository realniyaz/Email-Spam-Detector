# Model Training and Evaluation

This section describes the process of training a machine learning model for email spam detection and evaluating its performance.

## 1. Data Loading and Preprocessing

The process begins by loading the email data from a CSV file (e.g., `email.csv`) and preprocessing the text messages. The preprocessing steps include:

*   **Removing non-alphanumeric characters:** This step removes any characters that are not letters or numbers, such as punctuation and special symbols.
*   **Converting to lowercase:** Converting all text to lowercase ensures that words like "Hello" and "hello" are treated as the same word.
*   **Removing stop words:** Stop words are common words (e.g., "the," "a," "is") that do not carry much meaning and are removed to reduce noise in the data.

This preprocessing is performed using the `load_and_preprocess_data` function from the `preprocessing.py` module.

## 2. Feature Extraction

After preprocessing, the text data needs to be converted into numerical features that the machine learning model can understand. This is done using the Term Frequency-Inverse Document Frequency (TF-IDF) vectorizer.

*   **TF-IDF:** TF-IDF measures the importance of a word in a document relative to a collection of documents (the corpus). It gives higher weights to words that are frequent in a specific email but rare in the overall dataset.

The `TfidfVectorizer` from `scikit-learn` is used to perform this transformation.

## 3. Train-Test Split

The dataset is then split into two parts:

*   **Training set (80%):** Used to train the machine learning model.
*   **Testing set (20%):** Used to evaluate the performance of the trained model on unseen data.

This split is done using the `train_test_split` function from `scikit-learn`. A `random_state` is set for reproducibility.

## 4. Model Training

A Multinomial Naive Bayes classifier is used for this spam detection task.

*   **Multinomial Naive Bayes:** This is a probabilistic classifier that is well-suited for text classification problems. It assumes that the features (word frequencies) are independent of each other, given the class (spam or ham).

The `MultinomialNB` classifier from `scikit-learn` is instantiated and trained on the training data using the `fit` method.

## 5. Model Evaluation

After training, the model's performance is evaluated on the testing set. The following metrics are used:

*   **Classification Report:** This report provides precision, recall, F1-score, and support for each class (spam and ham).
    *   **Precision:** Out of all the emails predicted as spam, what proportion were actually spam?
    *   **Recall:** Out of all the emails that were actually spam, what proportion did the model correctly identify?
    *   **F1-score:** The harmonic mean of precision and recall, balancing both metrics.
    *   **Support:** The number of actual occurrences of each class in the test set.
*   **Accuracy:** The overall proportion of correctly classified emails.
*   **Confusion Matrix:** A table that visualizes the model's performance by showing the number of true positives, true negatives, false positives, and false negatives.

The `classification_report`, `accuracy_score`, and `confusion_matrix` functions from `scikit-learn` are used for evaluation. A heatmap is generated using `seaborn` to visualize the confusion matrix.

## 6. Example Prediction

After training and evaluation, the model can be used to predict whether new emails are spam or ham. The new emails must be preprocessed and transformed using the same `TfidfVectorizer` used during training before being passed to the model's `predict` method.

## Code Example (Python)

```python
# ... (imports and data loading/preprocessing) ...

tfidf = TfidfVectorizer()
X = tfidf.fit_transform(df['Message'])
y = df['Category']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = MultinomialNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(classification_report(y_test, y_pred))
# ... (other evaluation metrics and visualization) ...

new_emails = ["Congratulations you have won a lottery of 100000$", "Hello how are you doing today"]
new_emails_transformed = tfidf.transform(new_emails)
predictions = model.predict(new_emails_transformed)
print(predictions)
