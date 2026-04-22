"""Shared preprocessing logic for the Churn Predictor project."""

import pandas as pd
from sklearn.preprocessing import LabelEncoder


def preprocess(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Clean and encode the Telco Customer Churn dataframe.

    Returns
    -------
    X : pd.DataFrame   – feature matrix (all columns except Churn)
    y : pd.Series       – binary target (1 = churned, 0 = stayed)
    """
    df = df.copy()

    # Convert TotalCharges to numeric (some rows have whitespace strings)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # Drop rows that ended up with NaN after coercion
    df.dropna(inplace=True)

    # Drop the customer identifier — not a useful feature
    df.drop(columns=["customerID"], inplace=True)

    # Encode target column as binary int
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0}).astype(int)

    # Label-encode every remaining object column
    label_encoders: dict[str, LabelEncoder] = {}
    for col in df.select_dtypes(include="object").columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    y = df["Churn"]
    X = df.drop(columns=["Churn"])

    return X, y
