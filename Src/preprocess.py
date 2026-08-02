import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import SMOTE


def load_and_preprocess_data(file_path="Data/raw/ai4i2020.csv"):
    # Load dataset
    df = pd.read_csv(file_path)

    # Drop unnecessary columns
    df = df.drop(columns=["UDI", "Product ID"])

    # Encode categorical column
    encoder = LabelEncoder()
    df["Type"] = encoder.fit_transform(df["Type"])

    # Features and target
    X = df.drop("Machine failure", axis=1)
    y = df["Machine failure"]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Apply SMOTE
    smote = SMOTE(random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)

    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, scaler
