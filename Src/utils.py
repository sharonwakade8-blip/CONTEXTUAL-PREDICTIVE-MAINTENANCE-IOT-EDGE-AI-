import os
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import RocCurveDisplay, PrecisionRecallDisplay

def generate_reports(model, X_test, y_test):
    os.makedirs("Reports", exist_ok=True)

    # ROC Curve
    RocCurveDisplay.from_estimator(model, X_test, y_test)
    plt.savefig("Reports/roc_curve.png")
    plt.close()

    # Precision-Recall Curve
    PrecisionRecallDisplay.from_estimator(model, X_test, y_test)
    plt.savefig("Reports/pr_curve.png")
    plt.close()

    # Feature Importance
    plt.figure(figsize=(10, 5))
    plt.bar(range(len(model.feature_importances_)), model.feature_importances_)
    plt.xlabel("Feature Index")
    plt.ylabel("Importance")
    plt.title("LightGBM Feature Importance")
    plt.savefig("Reports/feature_importance.png")
    plt.close()

    print("Reports generated successfully!")
    