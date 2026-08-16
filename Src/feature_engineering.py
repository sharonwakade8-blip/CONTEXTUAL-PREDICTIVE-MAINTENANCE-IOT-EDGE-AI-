"""
feature_engineering.py
Role: Feature Engineering
Project: Contextual Predictive Maintenance (IoT Edge AI)
"""

import pandas as pd
from pathlib import Path


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create engineered features."""

    # Temperature difference
    df["Temp_Diff"] = (
        df["Process temperature"] - df["Air temperature"]
    )

    # Torque × Rotational Speed
    df["Power_Index"] = (
        df["Torque"] * df["Rotational speed"]
    )

    # Tool wear ratio
    df["Wear_Ratio"] = (
        df["Tool wear"] / (df["Tool wear"].max() + 1)
    )

    # Combined machine load
    df["Machine_Load"] = (
        df["Torque"] * df["Process temperature"]
    )

    return df


def main():
    input_file = "data/processed/processed_data.csv"
    output_file = "data/processed/featured_data.csv"

    df = pd.read_csv(input_file)

    df = create_features(df)

    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_file, index=False)

    print(f"Feature engineered data saved to {output_file}")
    print(df.head())


if __name__ == "__main__":
    main()

    
