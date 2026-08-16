"""
train.py
Role: Model Training
Project: Contextual Predictive Maintenance (IoT Edge AI)
"""

import joblib
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


def load_data(file_path):
    """Load processed dataset."""
    return pd.read_csv(file_path)


def train_model(df):
    """Train Random Forest model."""

    # Target
    y = df["Machine failure"]

    # Features
    X = df.drop(columns=["Machine failure"])

    # Remove unwanted columns if they exist
    drop_cols = ["UDI", "Product ID", "Type", "timestamp"]

    for col in drop_cols:
        if col in X.columns:
            X.drop(columns=col, inplace=True)

    # Keep only numeric columns
    X = X.select_dtypes(include=["number"])

    print("\nTraining Features:")
    print(X.columns.tolist())

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    # Create model
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    # Train model
    model.fit(X_train, y_train)

    # Test accuracy
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"\nTraining Accuracy: {accuracy:.4f}")

    return model


def save_model(model, output_path):
    """Save trained model."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_path)
    print(f"Model saved to {output_path}")


def main():
    input_file = "data/processed/fused_data.csv"
    model_file = "models/model.pkl"

    df = load_data(input_file)

    model = train_model(df)

    save_model(model, model_file)


if __name__ == "__main__":
    main()
    
