"""
feature_engineering.py
Role: Feature Engineering Engineer (Samruddhi)
Project: Contextual Predictive Maintenance (IoT Edge AI)

This script:
1. Loads the raw AI4I dataset
2. Creates rolling mean and rolling standard deviation features
3. Creates lag features
4. Drops rows with missing values
5. Saves the engineered dataset
"""

import pandas as pd
from pathlib import Path


def load_data(filepath: str) -> pd.DataFrame:
    """Load dataset."""
    df = pd.read_csv(filepath)
    print(f"Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def add_rolling_features(df: pd.DataFrame, sensor_cols: list, window: int = 3):
    """Create rolling mean and rolling standard deviation."""
    for col in sensor_cols:
        df[f"{col}_roll_mean"] = df[col].rolling(window=window).mean()
        df[f"{col}_roll_std"] = df[col].rolling(window=window).std()
    return df


def add_lag_features(df: pd.DataFrame, sensor_cols: list, lags=[1, 2, 3]):
    """Create lag features."""
    for col in sensor_cols:
        for lag in lags:
            df[f"{col}_lag_{lag}"] = df[col].shift(lag)
    return df


def drop_incomplete_rows(df: pd.DataFrame):
    """Remove rows with missing values created by rolling/lag."""
    before = len(df)
    df = df.dropna().reset_index(drop=True)
    print(f"Dropped {before - len(df)} incomplete rows")
    return df


def main():

    input_path = "Data/raw/ai4i2020.csv"
    output_path = "Data/processed/feature_engineered.csv"

    sensor_cols = [
        "Air temperature [K]",
        "Process temperature [K]",
        "Rotational speed [rpm]",
        "Torque [Nm]",
        "Tool wear [min]"
    ]

    df = load_data(input_path)

    # Keep original order
    df = df.reset_index(drop=True)

    # Feature Engineering
    df = add_rolling_features(df, sensor_cols, window=3)
    df = add_lag_features(df, sensor_cols, lags=[1, 2, 3])
    df = drop_incomplete_rows(df)

    # Create output folder if it doesn't exist
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Save dataset
    df.to_csv(output_path, index=False)

    print("\nFeature engineering completed successfully.")
    print(f"Saved to: {output_path}")
    print(f"Final dataset shape: {df.shape}")


if __name__ == "__main__":
    main()