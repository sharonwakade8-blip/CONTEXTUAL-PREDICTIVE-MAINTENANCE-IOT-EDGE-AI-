"""
predict.py
Role: Prediction Module
Project: Contextual Predictive Maintenance (IoT Edge AI)
"""

import joblib
import pandas as pd
from pathlib import Path


def load_model(model_path):
    """Load trained model."""
    return joblib.load(model_path)


def predict(model, input_path, output_path):

    # Load dataset
    df = pd.read_csv(input_path)

    # Copy original data for output
    results = df.copy()

    # Remove target column if present
    if "Machine failure" in df.columns:
        X = df.drop(columns=["Machine failure"])
    else:
        X = df.copy()

    # Remove columns that were excluded during training
    drop_cols = ["UDI", "Product ID", "Type", "timestamp"]

    for col in drop_cols:
        if col in X.columns:
            X.drop(columns=col, inplace=True)

    # Keep only numeric columns
    X = X.select_dtypes(include=["number"])

    print("\nPrediction Features:")
    print(X.columns.tolist())

    # Make predictions
    results["Prediction"] = model.predict(X)

    # Prediction probabilities
    if hasattr(model, "predict_proba"):
        results["Failure_Probability"] = model.predict_proba(X)[:, 1]

    # Save output
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_path, index=False)

    print(f"\nPredictions saved to {output_path}")
    print(results.head())


def main():
    model_path = "models/model.pkl"
    input_file = "data/processed/fused_data.csv"
    output_file = "output/predictions.csv"

    model = load_model(model_path)

    predict(model, input_file, output_file)


if __name__ == "__main__":
    main()

    
