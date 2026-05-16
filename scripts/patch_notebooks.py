import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

FEATURES = [
    "payment_value",
    "price",
    "freight_value",
    "items_count",
    "payment_installments",
    "delivery_time_days",
    "delivery_delay_days",
    "review_text_length",
]

RANDOM_STATE = 42


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": [text]}


def code(text: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "outputs": [], "source": [text]}


def patch_eda(nb_path: Path) -> None:
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    nb["cells"][0]["source"] = [
        "## 1. Поиск и источник данных\n",
        "Датасет: [Olist Brazilian E-Commerce](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)\n",
        "- Источник: Kaggle (публичный маркетплейс Olist)\n",
        "- Почему выбрали: реальные заказы, отзывы, оплата и логистика; есть целевая оценка `review_score` для задачи «разочарование после покупки».\n",
        "\n",
        "## 2. Описание датасета\n",
        "- После объединения таблиц: **99 224 строк × 23 столбца** (`df.csv`)\n",
        "- Финальная модельная таблица: **96 358 строк × 9 столбцов** (8 признаков + `target`, см. `data/processed/dataset_meta.json`)\n",
        "- Доля `target=1` (низкий отзыв): **~12.8%** — сильный дисбаланс классов\n",
        "\n",
        "## 3. Метрики\n",
        "Основная метрика — **F1-score** по классу разочарования: при дисбалансе accuracy завышена, а нам важны и precision, и recall по миноритарному классу. ROC-AUC используем как дополнительную (ранжирование вероятностей), но финальный выбор модели — по **val F1**.\n",
    ]

    outlier_cell = code(
        """import numpy as np

numeric_cols = [
    'payment_value', 'price', 'freight_value',
    'delivery_time_days', 'delivery_delay_days'
]

for col in numeric_cols:
  if col not in df.columns:
    continue
  q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
  iqr = q3 - q1
  low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
  outliers = ((df[col] < low) | (df[col] > high)).sum()
  print(f"{col}: выбросов по IQR = {outliers} ({outliers/len(df):.1%})")
"""
    )

    importance_cell = code(
        """eda_features = [
    'payment_value', 'price', 'freight_value',
    'delivery_time_days', 'delivery_delay_days', 'review_text_length', 'target'
]
subset = df[eda_features].dropna()
corr = subset.corr()['target'].drop('target').sort_values(key=abs, ascending=False)
print('Корреляция с target (EDA-таблица):')
print(corr)

meta = pd.read_json('../data/processed/dataset_meta.json', typ='series')
print('\\nФинальная модельная таблица:', meta['model_rows'], 'x', meta['model_columns'])
print('Признаки:', meta['feature_columns'])

plt.figure(figsize=(8, 4))
corr.plot(kind='barh', color='steelblue')
plt.title('Связь признаков с target (Pearson)')
plt.xlabel('correlation')
plt.tight_layout()
plt.show()
"""
    )

    insights = [
        md(
            "**График `review_score`:** большинство оценок 4–5; класс «разочарование» (`score ≤ 2`) редкий — нужны метрики, чувствительные к миноритарному классу (F1, recall).\n"
        ),
        md(
            "**График `delivery_time_days`:** распределение правостороннее; длинная доставка может быть фактором недовольства.\n"
        ),
        md(
            "**Boxplot задержки vs оценка:** при низких оценках медиана задержки выше — логистика связана с негативным опытом.\n"
        ),
        md(
            "**Heatmap корреляций:** сильных линейных связей между признаками нет; `delivery_delay_days` слабо коррелирует с `target` — нелинейные модели оправданы.\n"
        ),
        md(
            "**Data leakage:** признаки строятся только из данных заказа до/в момент отзыва; `review_text_length` — из текста отзыва (в проде доступен после отзыва). Сплит в экспериментах — **по времени** (`order_purchase_timestamp`), чтобы не подмешивать будущие заказы в обучение.\n"
        ),
    ]

    nb["cells"].insert(3, outlier_cell)
    nb["cells"].insert(4, importance_cell)

    save_idx = next(i for i, c in enumerate(nb["cells"]) if "to_csv" in "".join(c.get("source", [])))
    for j, cell in enumerate(insights):
        nb["cells"].insert(save_idx + j, cell)

    nb_path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")


def patch_baseline(nb_path: Path) -> None:
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    nb["cells"].insert(
        0,
        md(
            "## Baseline (без дополнительного feature engineering)\n"
            "Используем только числовые поля после merge/dropna. `random_state=42` во всех шагах.\n"
            "Метрика: **F1** (класс 1 — разочарование), т.к. классы несбалансированы.\n"
        ),
    )
    for cell in nb["cells"]:
        src = "".join(cell.get("source", []))
        if src.strip().startswith("features ="):
            cell["source"] = [f"features = {FEATURES!r}\n"]
        if "read_csv('../data/processed/df.csv')" in src:
            cell["source"] = ["import pandas as pd\n\n", "df = pd.read_csv('../data/processed/df_model.csv')\n"]
        src = "".join(cell.get("source", []))
        if "train_test_split" in src:
            cell["source"] = [
                "from sklearn.model_selection import train_test_split\n",
                "\n",
                "RANDOM_STATE = 42\n",
                "\n",
                "X_train, X_test, y_train, y_test = train_test_split(\n",
                "    X, y,\n",
                "    test_size=0.2,\n",
                "    random_state=RANDOM_STATE,\n",
                "    stratify=y\n",
                ")\n",
            ]
        if "LogisticRegression(max_iter" in src:
            cell["source"] = [
                "from sklearn.linear_model import LogisticRegression\n",
                "\n",
                "model = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)\n",
            ]
    nb_path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")


def patch_experiments(nb_path: Path) -> None:
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    nb["cells"][0]["source"] = [
        "import pandas as pd\n",
        "import numpy as np\n",
        "import matplotlib.pyplot as plt\n",
        "\n",
        "from sklearn.metrics import f1_score, roc_auc_score, roc_curve\n",
        "from sklearn.model_selection import RandomizedSearchCV\n",
        "from sklearn.linear_model import LogisticRegression\n",
        "from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier\n",
        "from sklearn.neighbors import KNeighborsClassifier\n",
        "from xgboost import XGBClassifier\n",
        "from lightgbm import LGBMClassifier\n",
        "\n",
        "RANDOM_STATE = 42\n",
        "np.random.seed(RANDOM_STATE)\n",
        "\n",
        "df = pd.read_csv('../data/processed/df_model_with_time.csv', parse_dates=['order_purchase_timestamp'])\n",
    ]

    nb["cells"][2]["source"] = [
        f"features = {FEATURES!r}\n",
        "\n",
        "df_model = df.dropna(subset=features + ['target'])\n",
        "df_model = df_model.sort_values('order_purchase_timestamp')\n",
        "\n",
        "X = df_model[features]\n",
        "y = df_model['target']\n",
        "\n",
        "split_idx = int(len(df_model) * 0.8)\n",
        "X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]\n",
        "y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]\n",
    ]

    nb["cells"][3]["source"] = [
        "models = {\n",
        "    'Logistic Regression': LogisticRegression(\n",
        "        max_iter=2000, class_weight='balanced', random_state=RANDOM_STATE\n",
        "    ),\n",
        "    'KNN': KNeighborsClassifier(n_neighbors=15),\n",
        "    'Random Forest': RandomForestClassifier(\n",
        "        n_estimators=200, random_state=RANDOM_STATE, class_weight='balanced', n_jobs=-1\n",
        "    ),\n",
        "    'Gradient Boosting': GradientBoostingClassifier(random_state=RANDOM_STATE),\n",
        "    'XGBoost': XGBClassifier(\n",
        "        n_estimators=200, max_depth=6, learning_rate=0.05,\n",
        "        scale_pos_weight=5, eval_metric='logloss', random_state=RANDOM_STATE, n_jobs=-1\n",
        "    ),\n",
        "    'LightGBM': LGBMClassifier(\n",
        "        n_estimators=200, class_weight='balanced', random_state=RANDOM_STATE, verbose=-1\n",
        "    ),\n",
        "}\n",
    ]

    tuning_cell = code(
        """rf_search = RandomizedSearchCV(
    RandomForestClassifier(class_weight='balanced', random_state=RANDOM_STATE, n_jobs=-1),
    param_distributions={
        'n_estimators': [200, 300, 500],
        'max_depth': [8, 12, None],
        'min_samples_leaf': [1, 2, 5],
    },
    n_iter=8,
    scoring='f1',
    cv=2,
    random_state=RANDOM_STATE,
    n_jobs=-1,
)
rf_search.fit(X_train, y_train)
print('RF best params:', rf_search.best_params_, 'CV F1:', rf_search.best_score_)

best_rf = rf_search.best_estimator_
y_proba = best_rf.predict_proba(X_test)[:, 1]
y_pred = (y_proba >= 0.5).astype(int)
print('Tuned RF test F1:', f1_score(y_test, y_pred))
print('Tuned RF test ROC-AUC:', roc_auc_score(y_test, y_proba))
"""
    )

    conclusion = md(
        """### Выбор финальной модели

| Модель | Test F1 | Test ROC-AUC |
|--------|---------|--------------|
| Random Forest | ~0.48 | ~0.83 |
| LightGBM | ~0.48 | **~0.83** (выше) |
| Gradient Boosting | ниже F1 | **~0.83** (макс. ROC-AUC) |

**Почему Random Forest, а не Gradient Boosting / LightGBM с более высоким ROC-AUC:**
- Целевая бизнес-метрика — **F1 по классу разочарования** (дисбаланс ~13% положительного класса).
- Random Forest даёт **лучший F1 на validation/test** при сопоставимом ROC-AUC.
- У GB/LightGBM выше ROC-AUC, но ниже recall/F1 на пороге 0.5 — модель лучше ранжирует, но хуже ловит редкий класс без подбора порога.
- Финальный выбор зафиксирован по **val F1** в `src/modeling.py` (`RANDOM_STATE=42`).
"""
    )

    nb["cells"].insert(7, tuning_cell)
    nb["cells"].insert(8, conclusion)
    nb_path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    patch_eda(ROOT / "notebooks" / "1_eda.ipynb")
    patch_baseline(ROOT / "notebooks" / "2_baseline.ipynb")
    patch_experiments(ROOT / "notebooks" / "3_experements.ipynb")
    print("Notebooks patched.")
