import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

model = joblib.load("models/model.pkl")
df = pd.read_csv("data/processed/fused_data.csv")

y = df["Machine failure"]
X = df.drop(columns=["Machine failure"])

drop_cols = ["UDI", "Product ID", "Type", "timestamp"]
X = X.drop(columns=[c for c in drop_cols if c in X.columns])
X = X.select_dtypes(include="number")

ConfusionMatrixDisplay.from_estimator(model, X, y)
plt.savefig("reports/confusion_matrix.png")
plt.show()

