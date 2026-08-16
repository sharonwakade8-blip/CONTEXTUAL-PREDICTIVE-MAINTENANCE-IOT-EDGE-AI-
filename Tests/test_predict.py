import os
import pandas as pd

def test_processed_data_exists():
    assert os.path.exists("data/processed/fused_data.csv")

def test_processed_data_not_empty():
    df = pd.read_csv("data/processed/fused_data.csv")
    assert len(df) > 0
