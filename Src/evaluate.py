"""
evaluate.py
Role: Model Evaluation
Project: Contextual Predictive Maintenance (IoT Edge AI)
"""

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)


def load_model(model_path):
    """Load trained model."""
    return joblib.load(model_path)


def evaluate_model(model, data_path):
    """Evaluate trained model."""

    # Load dataset
    df = pd.read_csv(data_path)

    # Target
    y = df["Machine failure"]

    # Features
    X = df.drop(columns=["Machine failure"])

    # Remove unwanted columns
    drop_cols = ["UDI", "Product ID", "Type", "timestamp"]

    for col in drop_cols:
        if col in X.columns:
            X.drop(columns=col, inplace=True)

    # Keep only numeric columns
    X = X.select_dtypes(include=["number"])

    print("\nFeatures used for evaluation:")
    print(X.columns.tolist())

    # Predictions
    predictions = model.predict(X)

    # Metrics
    accuracy = accuracy_score(y, predictions)
    precision = precision_score(y, predictions)
    recall = recall_score(y, predictions)
    f1 = f1_score(y, predictions)

    print("\n========== MODEL EVALUATION ==========")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")

    print("\nConfusion Matrix")
    print(confusion_matrix(y, predictions))

    print("\nClassification Report")
    print(classification_report(y, predictions))


def main():
    model_path = "models/model.pkl"
    test_data = "data/processed/fused_data.csv"

    model = load_model(model_path)

    evaluate_model(model, test_data)


if __name__ == "__main__":
    main()


    
