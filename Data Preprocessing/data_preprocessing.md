# Data Preprocessing

#### 1. import necessary libraries:

1. pandas: For data manipulation (DataFrames).
2. re: For regular expressions (used for character removal).
3. nltk: Natural Language Toolkit for stop word removal.
4. stopwords from nltk.corpus: List of common English stop words.
5. text_preprocessing(text) function:

#### 2. Handling NaN Values: 
The function now starts by checking if the input text is a NaN (Not a Number) value using pd.isna(text). If it is, the function immediately returns an empty string (""). This prevents errors that would occur if you tried to perform string operations on NaN values.

#### 3. Remove non-alphabetic characters: 
re.sub('[^a-zA-Z]', ' ', text) uses a regular expression to replace any character that is not an uppercase or lowercase letter (a-z, A-Z) with a space. This effectively removes numbers, punctuation, and other special characters.
Convert to lowercase: text.lower() converts the entire text to lowercase. This is important because it ensures that words like "Hello" and "hello" are treated as the same word.

#### 4. Split into words: 
text.split() splits the text string into a list of individual words, using spaces as delimiters.

#### 5. Remove stop words:
stopwords.words('english') retrieves the list of English stop words (common words like "the", "a", "is", "are", etc.).
stop_words = set(stopwords.words('english')) converting the stop words list to set for faster lookup.

A list comprehension [word for word in words if word not in stop_words] creates a new list containing only the words that are not in the stop words list.

#### 6. Join words back into a string: 
" ".join(words) joins the filtered words back together into a single string, with spaces separating each word.

#### 7. Return the preprocessed text: The function returns the cleaned and preprocessed text string.

## Example Usage:

A sample DataFrame df is created for demonstration purposes. Replace this with your actual data loading code.
```df['Email'] = df['Email'].apply(text_preprocessing)``` applies the text_preprocessing function to each value in the 'Email' column of the DataFrame. The .apply() method is a powerful way to apply a function to every element in a pandas Series (a column in a DataFrame).
The original and preprocessed dataframes are printed to show the effect of the function.
