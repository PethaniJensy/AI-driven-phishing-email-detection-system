# =============================================================================
# MODULE 2: MACHINE LEARNING MODEL TRAINING & EVALUATION
# File: src/train_model.py
# Role: Trains Naive Bayes, Logistic Regression, and Random Forest classifiers,
#       evaluates them, and saves the best model.
# =============================================================================

import os
import sys
import pickle
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import matplotlib
matplotlib.use('Agg')          # headless backend (no GUI needed)
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.naive_bayes      import MultinomialNB
from sklearn.linear_model     import LogisticRegression
from sklearn.ensemble         import RandomForestClassifier
from sklearn.metrics          import (accuracy_score, classification_report,
                                      confusion_matrix, ConfusionMatrixDisplay)

# Make sure the src/ directory is importable
sys.path.insert(0, os.path.dirname(__file__))
from preprocess import preprocess_pipeline


# ------------------------------------------------------------------
# Train & Evaluate a single model
# ------------------------------------------------------------------
def train_evaluate(model, model_name: str,
                   X_train, X_test, y_train, y_test) -> dict:
    print(f"\n{'='*60}")
    print(f"  Training: {model_name}")
    print(f"{'='*60}")

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc    = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred,
                                   target_names=['Legitimate', 'Phishing'],
                                   output_dict=True)
    cm     = confusion_matrix(y_test, y_pred)

    print(f"  Accuracy : {acc*100:.2f}%")
    print(f"\n  Classification Report:")
    print(classification_report(y_test, y_pred,
                                target_names=['Legitimate', 'Phishing']))

    # ---- Confusion Matrix Plot ----
    fig, ax = plt.subplots(figsize=(5, 4))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=['Legitimate', 'Phishing'])
    disp.plot(ax=ax, colorbar=False, cmap='Blues')
    ax.set_title(f'Confusion Matrix – {model_name}')
    plt.tight_layout()
    plot_path = f'../models/confusion_matrix_{model_name.replace(" ", "_")}.png'
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"  [INFO] Confusion matrix saved → {plot_path}")

    return {
        'model'     : model,
        'name'      : model_name,
        'accuracy'  : acc,
        'report'    : report,
        'confusion' : cm,
    }


# ------------------------------------------------------------------
# Compare all models and save the best one
# ------------------------------------------------------------------
def run_training(dataset_path: str,
                 max_features: int = 10000,
                 test_size: float  = 0.2):

    # ---- Preprocessing ----
    X_train, X_test, y_train, y_test, vectorizer = preprocess_pipeline(
        dataset_path, max_features=max_features, test_size=test_size
    )

    # ---- Define Models ----
    models = [
        (MultinomialNB(), "Naive Bayes"),
        (LogisticRegression(max_iter=1000, C=1.0, solver='lbfgs',
                            random_state=42), "Logistic Regression"),
        (RandomForestClassifier(n_estimators=100, random_state=42,
                                n_jobs=-1), "Random Forest"),
    ]

    results = []
    for clf, name in models:
        res = train_evaluate(clf, name, X_train, X_test, y_train, y_test)
        results.append(res)

    # ---- Pick Best Model by Accuracy ----
    best = max(results, key=lambda r: r['accuracy'])
    print(f"\n{'='*60}")
    print(f"  ★  Best Model : {best['name']}  "
          f"(Accuracy = {best['accuracy']*100:.2f}%)")
    print(f"{'='*60}\n")

    # ---- Save Best Model ----
    os.makedirs('../models', exist_ok=True)
    model_path = '../models/phishing_model.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(best['model'], f)
    print(f"[INFO] Best model saved → {model_path}")

    # ---- Accuracy Comparison Bar Chart ----
    names  = [r['name'] for r in results]
    scores = [r['accuracy'] * 100 for r in results]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(names, scores, color=['#4C72B0', '#55A868', '#C44E52'],
                  edgecolor='black')
    ax.set_ylim(80, 100)
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Model Accuracy Comparison')
    for bar, score in zip(bars, scores):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.2,
                f'{score:.2f}%', ha='center', va='bottom', fontsize=10)
    plt.tight_layout()
    chart_path = '../models/accuracy_comparison.png'
    plt.savefig(chart_path, dpi=150)
    plt.close()
    print(f"[INFO] Accuracy comparison chart saved → {chart_path}")

    return best['model'], vectorizer, results


# ------------------------------------------------------------------
# Evaluation helper (reusable from other scripts)
# ------------------------------------------------------------------
def detailed_evaluation(model, X_test, y_test):
    """Print a full evaluation report for a fitted model."""
    y_pred = model.predict(X_test)
    acc    = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred,
                                   target_names=['Legitimate', 'Phishing'])
    cm     = confusion_matrix(y_test, y_pred)

    print(f"Accuracy  : {acc*100:.2f}%")
    print(f"\nClassification Report:\n{report}")
    print(f"Confusion Matrix:\n{cm}")
    return acc, report, cm


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------
if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python train_model.py <path_to_dataset.csv>")
        sys.exit(1)

    dataset_path = sys.argv[1]
    run_training(dataset_path)
    print("\n[DONE] Training complete.")
