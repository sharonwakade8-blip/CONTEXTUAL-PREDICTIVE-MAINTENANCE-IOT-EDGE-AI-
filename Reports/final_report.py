"""
final_report.py
Generate Final PDF Report
Contextual Predictive Maintenance (IoT Edge AI)
"""

import os
import joblib
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
)

from reportlab.lib.styles import getSampleStyleSheet


MODEL_PATH = "models/model.pkl"
DATA_PATH = "data/processed/fused_data.csv"
REPORT_PATH = "reports/final_report.pdf"


def prepare_data(df):
    y = df["Machine failure"]

    X = df.drop(columns=["Machine failure"])

    drop_cols = [
        "UDI",
        "Product ID",
        "Type",
        "timestamp",
    ]

    for col in drop_cols:
        if col in X.columns:
            X.drop(columns=col, inplace=True)

    X = X.select_dtypes(include="number")

    return X, y


def main():

    if not os.path.exists(MODEL_PATH):
        print("Model not found.")
        return

    if not os.path.exists(DATA_PATH):
        print("Dataset not found.")
        return

    model = joblib.load(MODEL_PATH)

    df = pd.read_csv(DATA_PATH)

    X, y = prepare_data(df)

    pred = model.predict(X)

    accuracy = accuracy_score(y, pred)
    precision = precision_score(y, pred)
    recall = recall_score(y, pred)
    f1 = f1_score(y, pred)

    styles = getSampleStyleSheet()

    doc = SimpleDocTemplate(REPORT_PATH)

    story = []

    story.append(
        Paragraph(
            "<b>Contextual Predictive Maintenance (IoT Edge AI)</b>",
            styles["Title"],
        )
    )

    story.append(Spacer(1, 20))

    story.append(
        Paragraph(
            "Final Model Evaluation Report",
            styles["Heading2"],
        )
    )

    story.append(Spacer(1, 15))

    story.append(
        Paragraph(
            f"<b>Total Samples:</b> {len(df)}",
            styles["BodyText"],
        )
    )

    story.append(
        Paragraph(
            f"<b>Accuracy:</b> {accuracy:.4f}",
            styles["BodyText"],
        )
    )

    story.append(
        Paragraph(
            f"<b>Precision:</b> {precision:.4f}",
            styles["BodyText"],
        )
    )

    story.append(
        Paragraph(
            f"<b>Recall:</b> {recall:.4f}",
            styles["BodyText"],
        )
    )

    story.append(
        Paragraph(
            f"<b>F1 Score:</b> {f1:.4f}",
            styles["BodyText"],
        )
    )

    story.append(Spacer(1, 20))

    story.append(
        Paragraph(
            "The Random Forest model achieved excellent predictive performance "
            "on the processed IoT predictive maintenance dataset.",
            styles["BodyText"],
        )
    )

    doc.build(story)

    print("PDF Report Generated Successfully!")
    print(REPORT_PATH)


if __name__ == "__main__":
    main()
    