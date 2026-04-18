# =============================================================================
# MODULE 1: EMAIL PREPROCESSING
# File: src/preprocess.py
# Role: Loads the dataset, cleans email text, and extracts TF-IDF features.
# =============================================================================

import pandas as pd
import numpy as np
import re
import string
import os
import pickle

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

# ------------------------------------------------------------------
# Download required NLTK data (run once)
# ------------------------------------------------------------------
def download_nltk_data():
    for pkg in ['stopwords', 'punkt', 'punkt_tab']:
        try:
            nltk.download(pkg, quiet=True)
        except Exception:
            pass

# ------------------------------------------------------------------
# Text Cleaning
# ------------------------------------------------------------------
def clean_text(text: str) -> str:
    """
    Cleans raw email text:
    1. Lowercase
    2. Remove HTML tags
    3. Remove URLs
    4. Remove punctuation & numbers
    5. Remove extra whitespace
    """
    if not isinstance(text, str):
        return ""

    text = text.lower()                                          # lowercase
    text = re.sub(r'<[^>]+>', ' ', text)                        # strip HTML
    text = re.sub(r'http\S+|www\S+', ' ', text)                 # remove URLs
    text = re.sub(r'[^a-z\s]', ' ', text)                       # keep only letters
    text = re.sub(r'\s+', ' ', text).strip()                    # collapse spaces
    return text


def remove_stopwords_and_stem(text: str, stop_words: set, stemmer) -> str:
    """Tokenise, remove stop words, apply Porter stemming."""
    tokens = text.split()
    tokens = [stemmer.stem(w) for w in tokens if w not in stop_words and len(w) > 2]
    return ' '.join(tokens)


# ------------------------------------------------------------------
# Dataset Loading  (supports several column-name conventions)
# ------------------------------------------------------------------
def load_dataset(filepath: str) -> pd.DataFrame:
    """
    Loads the CSV dataset and returns a DataFrame with two columns:
        'text'  – email body
        'label' – 0 = legitimate, 1 = phishing/spam
    """
    print(f"[INFO] Loading dataset from: {filepath}")
    df = pd.read_csv(filepath, encoding='latin-1')

    print(f"[INFO] Raw columns: {df.columns.tolist()}")
    print(f"[INFO] Raw shape  : {df.shape}")

    # ---- normalise column names ----
    df.columns = df.columns.str.strip().str.lower()

    # ---- detect text column ----
    text_col = None
    for candidate in ['text', 'email', 'message', 'body', 'email_text', 'mail']:
        if candidate in df.columns:
            text_col = candidate
            break
    if text_col is None:
        # fallback: pick the column with the longest average string length
        str_cols = df.select_dtypes(include='object').columns.tolist()
        text_col = max(str_cols, key=lambda c: df[c].astype(str).str.len().mean())
        print(f"[WARN] Text column not found by name; using '{text_col}' (longest strings).")

    # ---- detect label column ----
    label_col = None
    for candidate in ['label', 'class', 'spam', 'phishing', 'type', 'category', 'target']:
        if candidate in df.columns:
            label_col = candidate
            break
    if label_col is None:
        raise ValueError(f"Cannot find a label column. Columns present: {df.columns.tolist()}")

    df = df[[text_col, label_col]].copy()
    df.columns = ['text', 'label']

    # ---- normalise labels to 0 / 1 ----
    unique_labels = df['label'].unique()
    print(f"[INFO] Unique labels before mapping: {unique_labels}")

    # map common string labels → integer
    str_map = {
        'spam': 1, 'phishing': 1, 'ham': 0, 'safe': 0, 'legitimate': 0,
        'not spam': 0, 'not phishing': 0, '1': 1, '0': 0
    }
    df['label'] = df['label'].astype(str).str.strip().str.lower().map(
        lambda v: str_map.get(v, v)
    )

    # if still non-numeric try coercion
    df['label'] = pd.to_numeric(df['label'], errors='coerce')
    df.dropna(subset=['label'], inplace=True)
    df['label'] = df['label'].astype(int)

    print(f"[INFO] Label distribution:\n{df['label'].value_counts()}\n")
    return df


# ------------------------------------------------------------------
# Full Preprocessing Pipeline
# ------------------------------------------------------------------
def preprocess_pipeline(filepath: str,
                        max_features: int = 10000,
                        test_size: float = 0.2,
                        random_state: int = 42):
    """
    End-to-end preprocessing:
        1. Load CSV
        2. Clean text
        3. Remove stopwords + stem
        4. TF-IDF vectorisation
        5. Train/test split
    Returns X_train, X_test, y_train, y_test, vectorizer
    """
    download_nltk_data()

    df = load_dataset(filepath)

    stop_words = set(stopwords.words('english'))
    stemmer     = PorterStemmer()

    print("[INFO] Cleaning and preprocessing text …")
    df['clean_text'] = df['text'].apply(clean_text)
    df['clean_text'] = df['clean_text'].apply(
        lambda t: remove_stopwords_and_stem(t, stop_words, stemmer)
    )

    # drop empties after cleaning
    df = df[df['clean_text'].str.len() > 0].reset_index(drop=True)
    print(f"[INFO] Samples after cleaning: {len(df)}")

    # ---- TF-IDF ----
    print(f"[INFO] Fitting TF-IDF vectorizer (max_features={max_features}) …")
    vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2))
    X = vectorizer.fit_transform(df['clean_text'])
    y = df['label'].values

    # ---- Split ----
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    print(f"[INFO] Train size: {X_train.shape[0]}  |  Test size: {X_test.shape[0]}")

    # ---- Save vectorizer ----
    os.makedirs('../models', exist_ok=True)
    with open('../models/tfidf_vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)
    print("[INFO] TF-IDF vectorizer saved to ../models/tfidf_vectorizer.pkl")

    return X_train, X_test, y_train, y_test, vectorizer


# ------------------------------------------------------------------
# Convenience: preprocess a single raw email string (for inference)
# ------------------------------------------------------------------
def preprocess_single_email(text: str, vectorizer) -> object:
    """Clean + vectorise a single raw email string for prediction."""
    download_nltk_data()
    stop_words = set(stopwords.words('english'))
    stemmer    = PorterStemmer()
    cleaned    = clean_text(text)
    cleaned    = remove_stopwords_and_stem(cleaned, stop_words, stemmer)
    return vectorizer.transform([cleaned])


# ------------------------------------------------------------------
# Quick smoke test
# ------------------------------------------------------------------
if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print("Usage: python preprocess.py <path_to_dataset.csv>")
        sys.exit(1)

    X_train, X_test, y_train, y_test, vec = preprocess_pipeline(sys.argv[1])
    print("[DONE] Preprocessing complete.")
    print(f"       X_train shape: {X_train.shape}")
    print(f"       X_test  shape: {X_test.shape}")
