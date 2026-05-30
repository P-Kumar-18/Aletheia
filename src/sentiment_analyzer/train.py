"""Training pipeline for the IMDB sentiment classifier."""

from __future__ import annotations

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / "models"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"


def load_imdb_data() -> pd.DataFrame:
    """Load the IMDB dataset from Hugging Face and return a DataFrame."""
    dataset = load_dataset("imdb")
    train_df = dataset["train"].to_pandas()
    test_df = dataset["test"].to_pandas()
    data = pd.concat([train_df, test_df], ignore_index=True)
    return data.rename(columns={"text": "review", "label": "sentiment"})


def build_pipeline() -> Pipeline:
    """Create the TF-IDF plus Logistic Regression pipeline."""
    return Pipeline(
        [
            ("tfidf", TfidfVectorizer(stop_words="english", max_features=10000)),
            ("model", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)),
        ]
    )


def save_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    """Save a confusion matrix plot for reporting."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.imshow(cm, cmap="Blues")
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_xticks([0, 1], labels=["Negative", "Positive"])
    ax.set_yticks([0, 1], labels=["Negative", "Positive"])

    for row in range(cm.shape[0]):
        for col in range(cm.shape[1]):
            ax.text(col, row, cm[row, col], ha="center", va="center", color="black")

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "confusion_matrix.png", dpi=200)
    plt.close(fig)


def main() -> None:
    """Train, evaluate, and persist the sentiment model."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    data = load_imdb_data()
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        data["review"],
        data["sentiment"],
        test_size=0.2,
        random_state=42,
        stratify=data["sentiment"],
    )

    pipeline = build_pipeline()
    pipeline.fit(train_texts, train_labels)
    predictions = pipeline.predict(test_texts)

    accuracy = accuracy_score(test_labels, predictions)
    f1 = f1_score(test_labels, predictions)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 score: {f1:.4f}")
    print(classification_report(test_labels, predictions, target_names=["Negative", "Positive"]))

    save_confusion_matrix(np.asarray(test_labels), np.asarray(predictions))
    joblib.dump(pipeline, MODELS_DIR / "sentiment_model.joblib")


if __name__ == "__main__":
    main()