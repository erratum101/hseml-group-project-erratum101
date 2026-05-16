import json
import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))

from config import FEATURE_COLUMNS, TARGET_COLUMN


def test_feature_columns_defined():
    assert len(FEATURE_COLUMNS) >= 6
    assert TARGET_COLUMN == "target"


def test_dataset_meta_exists_after_preprocess():
    meta_path = os.path.join(ROOT, "data", "processed", "dataset_meta.json")
    assert os.path.isfile(meta_path), "Run: make preprocess"
    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)
    assert meta["model_rows"] > 0
    assert meta["model_columns"] == len(FEATURE_COLUMNS) + 1


def test_model_csv_schema():
    model_path = os.path.join(ROOT, "data", "processed", "df_model.csv")
    assert os.path.isfile(model_path), "Run: make preprocess"
    df = pd.read_csv(model_path)
    assert set(FEATURE_COLUMNS + [TARGET_COLUMN]).issubset(df.columns)
    assert df.isnull().sum().sum() == 0
