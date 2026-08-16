"""
utils.py
Role: Utility Functions
Project: Contextual Predictive Maintenance (IoT Edge AI)
"""

import os
import joblib
import pandas as pd
from pathlib import Path


def ensure_directory(path):
    """
    Create a directory if it does not exist.
    """
    Path(path).mkdir(parents=True, exist_ok=True)


def load_csv(file_path):
    """
    Load a CSV file.
    """
    return pd.read_csv(file_path)


def save_csv(df, file_path):
    """
    Save a DataFrame to a CSV file.
    """
    ensure_directory(Path(file_path).parent)
    df.to_csv(file_path, index=False)
    print(f"Saved file: {file_path}")


def save_model(model, model_path):
    """
    Save a trained model.
    """
    ensure_directory(Path(model_path).parent)
    joblib.dump(model, model_path)
    print(f"Model saved: {model_path}")


def load_model(model_path):
    """
    Load a trained model.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    return joblib.load(model_path)


def print_shape(df, name="Dataset"):
    """
    Display dataset shape.
    """
    print(f"{name}: {df.shape[0]} rows, {df.shape[1]} columns")


def print_missing_values(df):
    """
    Display missing values.
    """
    print("\nMissing Values")
    print(df.isnull().sum())


def dataset_summary(df):
    """
    Display dataset summary.
    """
    print("\nDataset Summary")
    print(df.describe(include="all"))

    
