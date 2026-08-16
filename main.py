"""
main.py
Contextual Predictive Maintenance (IoT Edge AI)

Runs the complete pipeline:
1. Fetch data
2. Data ingestion
3. Preprocessing
4. Feature engineering
5. Data fusion
6. Model training
7. Model evaluation
8. Prediction
"""

import subprocess
import sys


def run_step(name, command):
    print(f"\n{'='*60}")
    print(f"Running: {name}")
    print(f"{'='*60}")

    result = subprocess.run(command)

    if result.returncode != 0:
        print(f"\n{name} failed.")
        sys.exit(1)

    print(f"{name} completed successfully.")


def main():

    steps = [
        ("Fetch Data", [sys.executable, "src/fetch_data.py"]),
        ("Data Ingestion", [sys.executable, "src/data_ingestion.py"]),
        ("Preprocessing", [sys.executable, "src/preprocessing.py"]),
        ("Feature Engineering", [sys.executable, "src/feature_engineering.py"]),
        ("Data Fusion", [sys.executable, "src/data_fusion.py"]),
        ("Training", [sys.executable, "src/train.py"]),
        ("Evaluation", [sys.executable, "src/evaluate.py"]),
        ("Prediction", [sys.executable, "src/predict.py"]),
    ]

    for name, command in steps:
        run_step(name, command)

    print("\nPipeline completed successfully!")
    print("Run the dashboard with:")
    print("python -m streamlit run dashboard/app.py")


if __name__ == "__main__":
    main()

    