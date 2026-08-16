import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

model = joblib.load("models/model.pkl")
df = pd.read_csv("data/processed/fused_data.csv")

y = df["Machine failure"]
X = df.drop(columns=["Machine failure"])

drop_cols = ["UDI", "Product ID", "Type", "timestamp"]
X = X.drop(columns=[c for c in drop_cols if c in X.columns])
X = X.select_dtypes(include="number")

pred = model.predict(X)

metrics = pd.DataFrame({
    "Metric": ["Accuracy", "Precision", "Recall", "F1 Score"],
    "Value": [
        accuracy_score(y, pred),
        precision_score(y, pred),
        recall_score(y, pred),
        f1_score(y, pred)
    ]
})

metrics.to_csv("reports/metrics.csv", index=False)
print(metrics)
