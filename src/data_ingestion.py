"""
data_ingestion.py
Role: Data Collection & Preprocessing Engineer (Sakshi)
Project: Contextual Predictive Maintenance (IoT Edge AI)

What this does:
- Loads the raw AI4I 2020 dataset from CSV
- Explores it (shape, missing values, duplicates, failure rate)
"""

import pandas as pd


def load_data(filepath: str) -> pd.DataFrame:
    """Load the raw CSV file."""
    df=pd.read_csv("data/raw/ai4i2020.csv")
    
    
    
    print(f"Loaded data: {df.shape[0]} rows, {df.shape[1]} columns")
    return df

def explore_data(df: pd.DataFrame) -> None:
    """Print basic exploration info — missing values, duplicates, failure rate."""
    print("\n--- Missing values per column ---")
    print(df.isnull().sum())
    print(f"\nDuplicate rows: {df.duplicated().sum()}")
    print(f"Failure rate: {df['Machine failure'].mean() * 100:.2f}%")


if __name__ == "__main__":
    # Quick standalone test: just load and explore, no cleaning yet
    raw_df = load_data("data/raw/ai4i2020.csv")
    explore_data(raw_df)