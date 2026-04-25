"""
Preprocessing pipeline for Olist dataset
Creates final dataset for classification
"""

import pandas as pd
import os

# =========================
# PATHS
# =========================
RAW_PATH = os.path.join(os.path.dirname(__file__), "../data/raw/")
PROCESSED_PATH = os.path.join(os.path.dirname(__file__), "../data/processed/")


# =========================
# LOAD DATA
# =========================
def load_orders():
    return pd.read_csv(os.path.join(RAW_PATH, "olist_orders_dataset.csv"))


def load_reviews():
    return pd.read_csv(os.path.join(RAW_PATH, "olist_order_reviews_dataset.csv"))


def load_payments():
    return pd.read_csv(os.path.join(RAW_PATH, "olist_order_payments_dataset.csv"))


def load_items():
    return pd.read_csv(os.path.join(RAW_PATH, "olist_order_items_dataset.csv"))


# =========================
# CLEANING
# =========================
def clean_orders(df):
    df = df.drop_duplicates()
    df = df.dropna(subset=["order_id"])
    return df


def clean_reviews(df):
    df = df.drop_duplicates()
    return df


# =========================
# FEATURE ENGINEERING
# =========================
def add_features(orders, reviews):

    # --- FIX datetime ---
    date_cols = [
        "order_purchase_timestamp",
        "order_delivered_customer_date",
        "order_estimated_delivery_date"
    ]

    for col in date_cols:
        orders[col] = pd.to_datetime(orders[col], errors="coerce")

    # delivery time
    orders["delivery_time_days"] = (
        orders["order_delivered_customer_date"]
        - orders["order_purchase_timestamp"]
    ).dt.days

    # delay
    orders["delivery_delay_days"] = (
        orders["order_delivered_customer_date"]
        - orders["order_estimated_delivery_date"]
    ).dt.days

    # review text length
    reviews["review_text_length"] = reviews["review_comment_message"].fillna("").apply(len)

    return orders, reviews


# =========================
# BUILD DATASET
# =========================
def build_dataset():

    print("🚀 Loading data...")

    orders = clean_orders(load_orders())
    reviews = clean_reviews(load_reviews())
    payments = load_payments()
    items = load_items()

    print("📊 Merging datasets...")

    # feature engineering
    orders, reviews = add_features(orders, reviews)

    # aggregate payments
    payments_agg = payments.groupby("order_id")["payment_value"].sum().reset_index()

    # aggregate items (IMPORTANT FIX)
    items_agg = items.groupby("order_id").agg({
        "price": "sum",
        "freight_value": "sum"
    }).reset_index()

    # =========================
    # MERGE ALL TABLES
    # =========================
    df = orders.merge(reviews, on="order_id", how="inner")
    df = df.merge(payments_agg, on="order_id", how="left")
    df = df.merge(items_agg, on="order_id", how="left")

    # =========================
    # TARGET (CLASSIFICATION)
    # =========================
    df["target"] = (df["review_score"] <= 2).astype(int)

    # =========================
    # FINAL DATASET
    # =========================
    df_model = df[[
        "payment_value",
        "price",
        "freight_value",
        "delivery_time_days",
        "delivery_delay_days",
        "review_text_length",
        "target"
    ]]

    df_model = df_model.dropna()

    # =========================
    # SAVE
    # =========================
    os.makedirs(PROCESSED_PATH, exist_ok=True)

    output_path = os.path.join(PROCESSED_PATH, "df_model.csv")
    df_model.to_csv(output_path, index=False)

    print("✅ Dataset saved to:", output_path)
    print("📦 Shape:", df_model.shape)

    return df_model


# =========================
# RUN
# =========================
if __name__ == "__main__":
    build_dataset()