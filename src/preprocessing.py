"""
preprocessing.py
Role: Data Collection & Preprocessing Engineer (Sakshi)
Project: Contextual Predictive Maintenance (IoT Edge AI)

What this does:
1. Uses data_ingestion.py to load + explore the raw dataset
2. Cleans it (removes duplicates, fills missing values, caps outliers)
3. Adds synthetic timestamps (needed later for rolling time-series features)
4. Normalizes numeric sensor columns
5. Saves the final clean dataset as processed_data.csv
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler
from data_ingestion import load_data, explore_data
from pathlib import Path



def clean_data(df: pd.DataFrame, numeric_cols: list) -> pd.DataFrame:
    """Remove duplicates, fill missing values, and cap outliers using IQR."""
    df = df.drop_duplicates()

    # Fill missing numeric values with the column median
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

    # Cap outliers using the IQR method (keeps rows, just limits extreme values)
    for col in numeric_cols:
        q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        df[col] = df[col].clip(lower, upper)

    print(f"\nAfter cleaning: {df.shape[0]} rows remain")
    return df


def add_timestamps(df: pd.DataFrame) -> pd.DataFrame:
    """Add a synthetic sequential timestamp column (AI4I has no real time data)."""
    df["timestamp"] = pd.date_range(start="2026-01-01", periods=len(df), freq="10min")
    return df


def normalize_data(df: pd.DataFrame, numeric_cols: list) -> pd.DataFrame:
    """Scale numeric sensor columns to have mean 0, std 1."""
    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    return df


def main():
    input_path = "data/raw/ai4i2020.csv"              # raw Kaggle/UCI file lives here
    output_path = "data/processed/processed_data.csv"  # cleaned output goes here

    numeric_cols = [
        "Air temperature",
        "Process temperature",
        "Rotational speed",
        "Torque",
        "Tool wear",
    ]

    df = load_data(input_path)
    explore_data(df)
    df = clean_data(df, numeric_cols)
    df = add_timestamps(df)
    df = normalize_data(df, numeric_cols)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    
    print(f"\nSaved cleaned dataset to {output_path}")


if __name__ == "__main__":
    main()