"""Train a simple XGBoost churn classifier and persist the model + features."""
import joblib
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score
from xgboost import XGBClassifier
from src.utils.logger import get_logger
from src.utils.exceptions import ModelTrainingError

logger = get_logger("train")

RAW_CHURN = Path("D:/GitHub/Sale Insight Assistant/data/raw/telco_churn.csv")
MODEL_PATH = Path("D:/GitHub/Sale Insight Assistant/models/churn_xgb.pkl")
FEATURES_PATH = Path("D:/GitHub/Sale Insight Assistant/models/churn_features.pkl")


def preprocess(df: pd.DataFrame):
    df = df.copy()
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    y = df.pop("Churn").map({"Yes": 1, "No": 0})

    # One-hot encode categorical columns (safer than LabelEncoder for inference)
    X = pd.get_dummies(df)

    return X, y


def main():
    if not RAW_CHURN.exists():
        logger.error("Churn data missing: %s", RAW_CHURN)
        raise FileNotFoundError("Churn data missing in data/raw")

    try:
        df = pd.read_csv(RAW_CHURN)
        X, y = preprocess(df)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        model = XGBClassifier(
            n_estimators=50,
            max_depth=3,
            eval_metric="logloss",
            random_state=42,
        )
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        probs = model.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, preds)
        auc = roc_auc_score(y_test, probs)
        logger.info("Model trained (acc=%.3f, auc=%.3f)", acc, auc)

        # Save model + feature names
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, MODEL_PATH)
        joblib.dump(X.columns.tolist(), FEATURES_PATH)

        logger.info("Model and features saved to %s and %s", MODEL_PATH, FEATURES_PATH)
        return {"accuracy": acc, "auc": auc}

    except Exception as e:
        logger.exception("Training failed")
        raise ModelTrainingError(str(e))


if __name__ == "__main__":
    print(main())
