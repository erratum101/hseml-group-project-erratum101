# preprocessing.py
"""
Модуль для загрузки, очистки и подготовки данных Olist
"""
import pandas as pd
import os

RAW_PATH = os.path.join(os.path.dirname(__file__), '../data/raw/')
PROCESSED_PATH = os.path.join(os.path.dirname(__file__), '../data/processed/')

def load_orders():
    return pd.read_csv(os.path.join(RAW_PATH, 'olist_orders_dataset.csv'))

def load_reviews():
    return pd.read_csv(os.path.join(RAW_PATH, 'olist_order_reviews_dataset.csv'))

# TODO: добавить функции для остальных таблиц

def clean_orders(df):
    df = df.drop_duplicates()
    # Пример обработки пропусков
    df = df.dropna(subset=['order_id'])
    return df

def clean_reviews(df):
    df = df.drop_duplicates()
    return df

def add_features(orders, reviews):
    orders['order_purchase_timestamp'] = pd.to_datetime(orders['order_purchase_timestamp'])
    orders['order_delivered_customer_date'] = pd.to_datetime(orders['order_delivered_customer_date'])
    orders['delivery_time'] = (orders['order_delivered_customer_date'] - orders['order_purchase_timestamp']).dt.days
    reviews['review_length'] = reviews['review_comment_message'].fillna('').apply(len)
    return orders, reviews

def time_split(df, time_col, test_size=0.2):
    df = df.sort_values(time_col)
    split_idx = int(len(df) * (1 - test_size))
    train = df.iloc[:split_idx]
    test = df.iloc[split_idx:]
    return train, test
