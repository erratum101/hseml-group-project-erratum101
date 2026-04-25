"""
Classification modeling pipeline for Olist dataset
"""

import pandas as pd
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import f1_score, roc_auc_score


# =========================
# PATHS (FIXED)
# =========================
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "df_model.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")


# =========================
# LOAD DATA
# =========================
def load_data():
    return pd.read_csv(DATA_PATH)


# =========================
# MAIN
# =========================
if __name__ == "__main__":

    print("🚀 Loading dataset...")

    df = load_data()

    features = [
        "payment_value",
        "price",
        "freight_value",
        "delivery_time_days",
        "delivery_delay_days",
        "review_text_length"
    ]

    target = "target"

    X = df[features]
    y = df[target]

    # =========================
    # SPLIT
    # =========================
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # =========================
    # MODELS
    # =========================
    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "RandomForest": RandomForestClassifier(n_estimators=200, random_state=42),
        "GradientBoosting": GradientBoostingClassifier()
    }

    results = []

    print("\n📊 Training models...\n")

    for name, model in models.items():

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        # ROC-AUC (if possible)
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)[:, 1]
            roc = roc_auc_score(y_test, y_proba)
        else:
            roc = None

        f1 = f1_score(y_test, y_pred)

        results.append({
            "model": name,
            "f1": f1,
            "roc_auc": roc
        })

        print(f"{name} | F1 = {f1:.4f}")

    # =========================
    # RESULTS
    # =========================
    results_df = pd.DataFrame(results).sort_values("f1", ascending=False)

    print("\n🏆 FINAL RESULTS")
    print(results_df)

    # =========================
    # BEST MODEL
    # =========================
    best_model_name = results_df.iloc[0]["model"]
    best_model_object = models[best_model_name]

    print("\n✅ BEST MODEL:", best_model_name)

    # =========================
    # SAVE MODEL (FIXED)
    # =========================
    os.makedirs(MODEL_DIR, exist_ok=True)

    model_path = os.path.join(MODEL_DIR, "best_model.pkl")

    joblib.dump(best_model_object, model_path)

    print("\n💾 Model saved to:", model_path)