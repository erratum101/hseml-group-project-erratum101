from __future__ import annotations

import json
import os

import pandas as pd

from config import FEATURE_COLUMNS, TARGET_COLUMN

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
RAW_PATH = os.path.join(BASE_DIR, "data", "raw")
PROCESSED_PATH = os.path.join(BASE_DIR, "data", "processed")

RANDOM_STATE = 42
OUTLIER_COLUMNS = [
    "payment_value",
    "price",
    "freight_value",
    "delivery_time_days",
    "delivery_delay_days",
]


def load_orders() -> pd.DataFrame:
    return pd.read_csv(os.path.join(RAW_PATH, "olist_orders_dataset.csv"))


def load_reviews() -> pd.DataFrame:
    return pd.read_csv(os.path.join(RAW_PATH, "olist_order_reviews_dataset.csv"))


def load_payments() -> pd.DataFrame:
    return pd.read_csv(os.path.join(RAW_PATH, "olist_order_payments_dataset.csv"))


def load_items() -> pd.DataFrame:
    return pd.read_csv(os.path.join(RAW_PATH, "olist_order_items_dataset.csv"))


def clean_orders(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop_duplicates()
    return df.dropna(subset=["order_id"])


def clean_reviews(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop_duplicates()


def add_datetime_features(orders: pd.DataFrame) -> pd.DataFrame:
    date_cols = [
        "order_purchase_timestamp",
        "order_delivered_customer_date",
        "order_estimated_delivery_date",
    ]
    for col in date_cols:
        orders[col] = pd.to_datetime(orders[col], errors="coerce")

    orders["delivery_time_days"] = (
        orders["order_delivered_customer_date"] - orders["order_purchase_timestamp"]
    ).dt.days
    orders["delivery_delay_days"] = (
        orders["order_delivered_customer_date"] - orders["order_estimated_delivery_date"]
    ).dt.days
    return orders


def add_review_features(reviews: pd.DataFrame) -> pd.DataFrame:
    reviews = reviews.copy()
    reviews["review_text_length"] = reviews["review_comment_message"].fillna("").apply(len)
    return reviews


def clip_outliers_iqr(df: pd.DataFrame, columns: list[str]) -> tuple[pd.DataFrame, dict]:
    stats = {}
    clipped = df.copy()
    for col in columns:
        if col not in clipped.columns:
            continue
        series = clipped[col].dropna()
        q1, q3 = series.quantile(0.25), series.quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        before = ((clipped[col] < lower) | (clipped[col] > upper)).sum()
        clipped[col] = clipped[col].clip(lower=lower, upper=upper)
        stats[col] = {
            "q1": float(q1),
            "q3": float(q3),
            "lower_bound": float(lower),
            "upper_bound": float(upper),
            "rows_clipped": int(before),
        }
    return clipped, stats


def build_merged_table() -> pd.DataFrame:
    print("Loading data...")
    orders = clean_orders(load_orders())
    reviews = clean_reviews(load_reviews())
    payments = load_payments()
    items = load_items()

    orders = add_datetime_features(orders)
    reviews = add_review_features(reviews)

    payments_agg = payments.groupby("order_id").agg(
        payment_value=("payment_value", "sum"),
        payment_installments=("payment_installments", "max"),
    ).reset_index()

    items_agg = items.groupby("order_id").agg(
        price=("price", "sum"),
        freight_value=("freight_value", "sum"),
        items_count=("order_item_id", "count"),
    ).reset_index()

    df = orders.merge(reviews, on="order_id", how="inner")
    df = df.merge(payments_agg, on="order_id", how="left")
    df = df.merge(items_agg, on="order_id", how="left")
    df[TARGET_COLUMN] = (df["review_score"] <= 2).astype(int)
    return df


def build_model_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    model_cols = FEATURE_COLUMNS + [TARGET_COLUMN, "order_purchase_timestamp"]
    df_model = df[model_cols].copy()
    df_model = df_model.dropna(subset=FEATURE_COLUMNS + [TARGET_COLUMN])

    df_model, outlier_stats = clip_outliers_iqr(df_model, OUTLIER_COLUMNS)
    df_model = df_model.sort_values("order_purchase_timestamp").reset_index(drop=True)

    meta = {
        "merged_rows": int(len(df)),
        "merged_columns": int(df.shape[1]),
        "model_rows": int(len(df_model)),
        "model_columns": int(len(FEATURE_COLUMNS) + 1),
        "feature_columns": FEATURE_COLUMNS,
        "target_positive_rate": float(df_model[TARGET_COLUMN].mean()),
        "outlier_handling": outlier_stats,
    }
    return df_model, meta


def save_outputs(df_merged: pd.DataFrame, df_model: pd.DataFrame, meta: dict) -> None:
    os.makedirs(PROCESSED_PATH, exist_ok=True)
    merged_path = os.path.join(PROCESSED_PATH, "df.csv")
    model_path = os.path.join(PROCESSED_PATH, "df_model.csv")
    meta_path = os.path.join(PROCESSED_PATH, "dataset_meta.json")

    df_merged.to_csv(merged_path, index=False)
    df_model.drop(columns=["order_purchase_timestamp"]).to_csv(model_path, index=False)
    df_model.to_csv(os.path.join(PROCESSED_PATH, "df_model_with_time.csv"), index=False)

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("Merged dataset:", df_merged.shape, "->", merged_path)
    print("Model dataset:", df_model.shape, "->", model_path)
    print("Metadata ->", meta_path)


def build_dataset() -> pd.DataFrame:
    df_merged = build_merged_table()
    df_model, meta = build_model_dataset(df_merged)
    save_outputs(df_merged, df_model, meta)
    return df_model.drop(columns=["order_purchase_timestamp"])


if __name__ == "__main__":
    build_dataset()
