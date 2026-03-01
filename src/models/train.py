import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from src.data.preprocess import build_preprocessor
from sklearn.pipeline import Pipeline
from config import FILE_PATH

def train_model():

    df = pd.read_csv(FILE_PATH)

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.dropna()

    X = df.drop(columns=["Churn", "customerID"])
    y = df["Churn"].map({"Yes": 1, "No": 0})

    numerical_cols = X.select_dtypes(include=["int64", "float64"]).columns
    categorical_cols = X.select_dtypes(include=["object"]).columns

    preprocessor = build_preprocessor(numerical_cols, categorical_cols)

    model = LogisticRegression(max_iter=1000)

    pipeline = Pipeline([
        ("preprocessing", preprocessor),
        ("model", model)
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.2, random_state=42
    )

    pipeline.fit(X_train, y_train)

    preds = pipeline.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(y_test, preds)

    print(f"ROC-AUC: {auc}")

if __name__ == "__main__":
    train_model()