# ML Project — Предсказание разочарования пользователя после покупки в e-commerce

**Студент:** Непорожнев Всеволод Васильевич

**Группа:** БИВ231

## Оглавление

1. Описание задачи
2. Структура репозитория
3. Запуск
4. Данные
5. Результаты
6. Отчёт

## Описание задачи

Цель проекта — построить модель машинного обучения для предсказания вероятности разочарования пользователя после покупки в интернет-магазине на основе данных о заказах, доставке, оплате и отзывах.

Разочарование пользователя определяется через негативный отзыв (низкая оценка заказа).

**Задача:** Бинарная классификация

**Датасет:** Olist Brazilian E-Commerce Dataset (Kaggle)

**Источник:** https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce 

**Целевая переменная:**

* `target = 1`, если `review_score <= 2` (пользователь разочарован)
* `target = 0`, если `review_score > 2`

**Целевая метрика:** F1-score

Дополнительные метрики:

* Accuracy
* Precision
* Recall
* ROC-AUC

## Структура репозитория

```text
.
├── data
│   ├── processed               # Очищенные и обработанные данные
│   └── raw                     # Исходные файлы датасета Olist
├── models                      # Сохранённые модели
├── notebooks
│   ├── 01_eda.ipynb            # Разведочный анализ данных
│   ├── 02_baseline.ipynb       # Базовая модель
│   └── 03_experiments.ipynb    # Эксперименты и улучшения
├── presentation                # Презентация проекта
├── report
│   ├── images                  # Изображения и графики для отчёта
│   └── report.md               # Финальный отчёт
├── src
│   ├── preprocessing.py        # Очистка данных и feature engineering
│   └── modeling.py             # Обучение и оценка моделей
├── tests
│   └── test.py                 # Проверка корректности пайплайна
├── requirements.txt
└── README.md
```

## Запуск

```bash
# 1. Клонировать репозиторий
git clone https://github.com/erratum101/hseml-group-project-erratum101.git
### Скачать данные

# Датасет:
# https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce

# Распаковать в:
# data/raw/

# 2. Перейти в папку проекта
cd hseml-group-project-erratum101

# 3. Создать виртуальное окружение
python -m venv .venv

# 4. Активировать виртуальное окружение
# Windows
.venv\Scripts\activate

# Linux/macOS
source .venv/bin/activate

# 5. Установить зависимости
pip install -r requirements.txt
### Подготовка данных


python src/preprocessing.py
### Обучение модели

python src/modeling.py
# 6. Запустить Jupyter Notebook
jupyter notebook

### Сохранение модели

## Лучшая модель сохраняется в:

models/best_model.pkl
```

## Данные

Используемый датасет содержит реальные данные бразильского маркетплейса Olist:

* `olist_orders_dataset.csv` — заказы
* `olist_order_reviews_dataset.csv` — отзывы пользователей
* `olist_order_items_dataset.csv` — товары в заказах
* `olist_order_payments_dataset.csv` — платежи
* `olist_customers_dataset.csv` — клиенты
* `olist_products_dataset.csv` — товары
* `olist_sellers_dataset.csv` — продавцы
* `olist_geolocation_dataset.csv` — геоданные
* `product_category_name_translation.csv` — перевод категорий товаров

Структура данных:

* `data/raw/` — исходные данные
* `data/processed/` — обработанные данные

## Результаты

В рамках проекта будут протестированы несколько моделей:

| Модель                         | F1-score | ROC-AUC | Примечание |
| ------------------------------ | -------- | ------- | --------------------------------- |
| Baseline (Logistic Regression) | 0.390599| 0.820220 | Базовая модель |
| Random Forest                  | 0.480309 | 0.827283 | Улучшение качества                |
| Gradient Boosting              | 0.474260| 0.845590 | Основная экспериментальная модель |
| Лучшая модель                  | 0.480309| 0.827283  | RandomForest |

## Отчёт

Финальный отчёт находится в:

`report/report.md`

Отчёт включает:

* описание данных
* EDA
* feature engineering
* baseline модель
* эксперименты
* сравнение моделей
* итоговые выводы
