# =============================================================================
# MODULE 3: PHISHING DETECTION PREDICTION SYSTEM
# File: src/predict.py
# Role: Loads saved model + vectorizer and classifies new email text.
# =============================================================================

import os
import sys
import pickle

# Make sure src/ is on the path when called from CLI
sys.path.insert(0, os.path.dirname(__file__))
from preprocess import preprocess_single_email


# ------------------------------------------------------------------
# Model Loading
# ------------------------------------------------------------------
def load_model_and_vectorizer(model_path: str   = '../models/phishing_model.pkl',
                              vec_path: str     = '../models/tfidf_vectorizer.pkl'):
    """Load saved model and TF-IDF vectorizer from disk."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found at '{model_path}'. "
            "Run 'python src/train_model.py <dataset>' first."
        )
    if not os.path.exists(vec_path):
        raise FileNotFoundError(
            f"Vectorizer not found at '{vec_path}'. "
            "Run 'python src/train_model.py <dataset>' first."
        )

    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    with open(vec_path, 'rb') as f:
        vectorizer = pickle.load(f)

    return model, vectorizer


# ------------------------------------------------------------------
# Single Email Prediction
# ------------------------------------------------------------------
def predict_email(email_text: str, model, vectorizer) -> dict:
    """
    Classify a raw email string.

    Returns a dict:
        {
          'label'      : 'Phishing' | 'Legitimate',
          'label_code' : 1 | 0,
          'confidence' : float (0–100 %)
        }
    """
    features = preprocess_single_email(email_text, vectorizer)
    prediction = model.predict(features)[0]

    confidence = None
    if hasattr(model, 'predict_proba'):
        proba      = model.predict_proba(features)[0]
        confidence = round(float(proba[prediction]) * 100, 2)
    elif hasattr(model, 'decision_function'):
        score = model.decision_function(features)[0]
        # crude sigmoid to get a rough probability
        import math
        confidence = round(100 / (1 + math.exp(-score)), 2)

    label = 'Phishing' if prediction == 1 else 'Legitimate'
    return {
        'label'      : label,
        'label_code' : int(prediction),
        'confidence' : confidence,
    }


# ------------------------------------------------------------------
# Batch Prediction from a text file (one email per line)
# ------------------------------------------------------------------
def batch_predict(filepath: str, model, vectorizer):
    """Read emails from a plain-text file (one email per line) and print results."""
    with open(filepath, 'r', encoding='utf-8') as f:
        emails = [line.strip() for line in f if line.strip()]

    print(f"\n[INFO] Classifying {len(emails)} emails …\n")
    print(f"{'#':<5} {'Label':<12} {'Confidence':>12}   Preview")
    print('-' * 70)
    for i, email in enumerate(emails, 1):
        result = predict_email(email, model, vectorizer)
        conf   = f"{result['confidence']}%" if result['confidence'] is not None else 'N/A'
        preview = email[:55] + ('…' if len(email) > 55 else '')
        print(f"{i:<5} {result['label']:<12} {conf:>12}   {preview}")


# ------------------------------------------------------------------
# Interactive CLI
# ------------------------------------------------------------------
def interactive_mode(model, vectorizer):
    print("\n" + "="*60)
    print("   AI Phishing Email Detection System — Interactive Mode")
    print("   Type 'quit' or 'exit' to stop.")
    print("="*60 + "\n")

    while True:
        print("Paste your email text below (press Enter twice when done):")
        lines = []
        while True:
            line = input()
            if line == '':
                if lines:
                    break
            elif line.lower() in ('quit', 'exit'):
                print("Goodbye!")
                return
            else:
                lines.append(line)

        email_text = '\n'.join(lines)
        result     = predict_email(email_text, model, vectorizer)

        print(f"\n{'─'*40}")
        print(f"  Result     : {'🚨 PHISHING' if result['label_code'] == 1 else '✅ Legitimate'}")
        print(f"  Label      : {result['label']}")
        if result['confidence'] is not None:
            print(f"  Confidence : {result['confidence']}%")
        print(f"{'─'*40}\n")


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------
if __name__ == '__main__':
    model, vectorizer = load_model_and_vectorizer()

    if len(sys.argv) == 2:
        # batch mode: argument is a text file
        batch_predict(sys.argv[1], model, vectorizer)
    else:
        interactive_mode(model, vectorizer)
