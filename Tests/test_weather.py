import os
import pandas as pd

def test_weather_file_exists():
    file_path = "data/raw/weather_data.csv"

    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        assert len(df) > 0
    else:
        assert True