"""
data_fusion.py
Role: Data Fusion
Project: Contextual Predictive Maintenance (IoT Edge AI)

Combines engineered sensor data and prepares the final dataset
for machine learning.
"""

import pandas as pd
from pathlib import Path


def load_data(file_path):
    """Load feature engineered dataset."""
    return pd.read_csv(file_path)


def fuse_data(df):
    """Create fused features."""

    # Overall machine health score
    df["Health_Score"] = (
        df["Air temperature"]
        + df["Process temperature"]
        + df["Rotational speed"]
        + df["Torque"]
        + df["Tool wear"]
    ) / 5

    # Temperature-to-speed ratio
    df["Temp_Speed_Ratio"] = (
        df["Process temperature"]
        / (df["Rotational speed"] + 1)
    )

    # Torque-to-wear ratio
    df["Torque_Wear_Ratio"] = (
        df["Torque"]
        / (df["Tool wear"] + 1)
    )

    return df


def save_data(df, output_path):
    """Save fused dataset."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Fused dataset saved to {output_path}")


def main():
    input_file = "data/processed/featured_data.csv"
    output_file = "data/processed/fused_data.csv"

    df = load_data(input_file)
    df = fuse_data(df)
    save_data(df, output_file)

    print(df.head())


if __name__ == "__main__":
    main()
    
