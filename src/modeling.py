from __future__ import annotations

import os

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from config import FEATURE_COLUMNS, RANDOM_STATE, TARGET_COLUMN, TEST_SIZE, VAL_SIZE

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "df_model_with_time.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")
RESULTS_PATH = os.path.join(BASE_DIR, "data", "processed", "experiments.csv")


def load_data() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH, parse_dates=["order_purchase_timestamp"])


def temporal_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = df.sort_values("order_purchase_timestamp").reset_index(drop=True)
    n = len(df)
    test_start = int(n * (1 - TEST_SIZE))
    train_val = df.iloc[:test_start]
    test_df = df.iloc[test_start:]

    val_start = int(len(train_val) * (1 - VAL_SIZE))
    train_df = train_val.iloc[:val_start]
    val_df = train_val.iloc[val_start:]
    return train_df, val_df, test_df


def evaluate(model, X, y) -> dict:
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[:, 1]
        pred = (proba >= 0.5).astype(int)
        roc = roc_auc_score(y, proba)
    else:
        pred = model.predict(X)
        roc = None
    return {
        "f1": f1_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "roc_auc": roc,
    }


def default_models() -> dict:
    return {
        "LogisticRegression": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=2000,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "KNN": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", KNeighborsClassifier(n_neighbors=15)),
            ]
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=150,
            learning_rate=0.05,
            max_depth=4,
            random_state=RANDOM_STATE,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            scale_pos_weight=5,
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=200,
            learning_rate=0.05,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            verbose=-1,
            n_jobs=-1,
        ),
    }


def run_baseline_experiments(
    X_train, y_train, X_val, y_val, X_test, y_test
) -> pd.DataFrame:
    rows = []
    fitted = {}
    for name, model in default_models().items():
        model.fit(X_train, y_train)
        fitted[name] = model
        val_metrics = evaluate(model, X_val, y_val)
        test_metrics = evaluate(model, X_test, y_test)
        rows.append(
            {
                "model": name,
                "stage": "default",
                "val_f1": val_metrics["f1"],
                "val_roc_auc": val_metrics["roc_auc"],
                "test_f1": test_metrics["f1"],
                "test_precision": test_metrics["precision"],
                "test_recall": test_metrics["recall"],
                "test_roc_auc": test_metrics["roc_auc"],
            }
        )
        print(f"{name:20s} | val F1={val_metrics['f1']:.4f} | test F1={test_metrics['f1']:.4f}")
    return pd.DataFrame(rows), fitted


def run_hyperparameter_search(X_train, y_train) -> list[dict]:
    searches = [
        (
            "RandomForest",
            RandomForestClassifier(
                class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1
            ),
            {
                "n_estimators": [200, 300, 500],
                "max_depth": [8, 12, 16, None],
                "min_samples_leaf": [1, 2, 5],
            },
        ),
        (
            "GradientBoosting",
            GradientBoostingClassifier(random_state=RANDOM_STATE),
            {
                "n_estimators": [150, 200, 300],
                "learning_rate": [0.03, 0.05, 0.1],
                "max_depth": [3, 4, 5],
            },
        ),
        (
            "XGBoost",
            XGBClassifier(
                eval_metric="logloss",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
            {
                "n_estimators": [200, 300],
                "max_depth": [4, 6, 8],
                "learning_rate": [0.03, 0.05, 0.1],
                "scale_pos_weight": [3, 5, 7],
            },
        ),
    ]

    tuned_rows = []
    best_estimators = {}
    for name, estimator, param_grid in searches:
        search = RandomizedSearchCV(
            estimator,
            param_distributions=param_grid,
            n_iter=8,
            scoring="f1",
            cv=2,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=1,
        )
        search.fit(X_train, y_train)
        tuned_rows.append(
            {
                "model": name,
                "stage": "tuned",
                "best_params": search.best_params_,
                "cv_best_f1": search.best_score_,
            }
        )
        best_estimators[name] = search.best_estimator_
        print(f"Tuned {name}: CV F1={search.best_score_:.4f} | params={search.best_params_}")
    return tuned_rows, best_estimators


def select_final_model(results_df: pd.DataFrame) -> str:
    default = results_df[results_df["stage"] == "default"]
    return default.sort_values("val_f1", ascending=False).iloc[0]["model"]


def main() -> None:
    print("Loading dataset...")
    df = load_data()
    train_df, val_df, test_df = temporal_split(df)

    feature_cols = FEATURE_COLUMNS
    X_train, y_train = train_df[feature_cols], train_df[TARGET_COLUMN]
    X_val, y_val = val_df[feature_cols], val_df[TARGET_COLUMN]
    X_test, y_test = test_df[feature_cols], test_df[TARGET_COLUMN]

    print(f"Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")

    results_df, fitted = run_baseline_experiments(
        X_train, y_train, X_val, y_val, X_test, y_test
    )

    tuned_meta, tuned_models = run_hyperparameter_search(X_train, y_train)
    tuned_rows = []
    for name, model in tuned_models.items():
        model.fit(X_train, y_train)
        val_m = evaluate(model, X_val, y_val)
        test_m = evaluate(model, X_test, y_test)
        tuned_rows.append(
            {
                "model": name,
                "stage": "tuned",
                "val_f1": val_m["f1"],
                "val_roc_auc": val_m["roc_auc"],
                "test_f1": test_m["f1"],
                "test_precision": test_m["precision"],
                "test_recall": test_m["recall"],
                "test_roc_auc": test_m["roc_auc"],
            }
        )
    tuned_df = pd.DataFrame(tuned_rows)
    all_results = pd.concat([results_df, tuned_df], ignore_index=True)
    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    all_results.to_csv(RESULTS_PATH, index=False)

    best_name = select_final_model(results_df)
    best_model = fitted[best_name]
    best_model.fit(
        pd.concat([X_train, X_val]),
        pd.concat([y_train, y_val]),
    )

    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, "best_model.pkl")
    joblib.dump(best_model, model_path)

    print("\n=== Results (default models) ===")
    print(results_df.sort_values("val_f1", ascending=False).to_string(index=False))
    print("\n=== Results (tuned models) ===")
    print(tuned_df.sort_values("val_f1", ascending=False).to_string(index=False))
    print(f"\nFinal model (by val F1): {best_name}")
    print(f"Saved to {model_path}")
    print(f"Full table: {RESULTS_PATH}")


if __name__ == "__main__":
    np.random.seed(RANDOM_STATE)
    main()
