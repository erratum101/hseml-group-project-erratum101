import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

EDA_INTRO = """## 1. Откуда данные
Датасет: [Olist Brazilian E-Commerce](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) (Kaggle).

**Почему подошёл:** есть заказы, оплата, доставка и оценка отзыва — всё нужно, чтобы учиться предсказывать недовольство покупателя.

## 2. Сколько данных
- После объединения таблиц: **99 224 строки × 23 столбца**
- Для модели: **96 358 строк × 9 столбцов** (8 признаков + цель `target`)
- Недовольных заказов (`target = 1`): **около 13%** — класс редкий

## 3. Какую метрику берём главной
**F1-score** для класса «недоволен».

**Простыми словами:** нам важно не пропустить недовольного клиента и не слишком часто ошибаться. F1 как раз это измеряет.

**Почему не accuracy:** если всегда отвечать «всё хорошо», accuracy будет ~87%, но модель бесполезна.

**ROC-AUC** смотрим дополнительно (насколько хорошо модель ранжирует риск), но **модель выбираем по F1**.
"""

INSIGHTS = [
    "**График оценок (`review_score`):** почти все отзывы хорошие (4–5 звёзд). Плохих мало — поэтому нужна метрика F1, а не просто доля правильных ответов.",
    "**График срока доставки:** у части заказов доставка очень долгая. Слишком экстремальные значения мы потом обрезаем при очистке (метод IQR).",
    "**График задержки и оценки:** когда оценка низкая, задержка доставки в среднем больше. Логистика связана с недовольством.",
    "**Тепловая карта:** признаки не повторяют друг друга сильно — можно обучать несколько моделей без проблемы «дублирующих» столбцов.",
    """**Утечка данных (data leakage):** мы **не** подмешиваем в обучение информацию из будущего.
- В признаки **не** кладём саму оценку отзыва.
- В экспериментах данные режем **по дате покупки**: сначала старые заказы — в обучение, новые — в тест.""",
]

BASELINE_MD = """## Baseline — простая модель для сравнения

**Модель:** логистическая регрессия (стандартные настройки, с учётом дисбаланса классов).

**Метрика:** F1 для класса «недоволен» — потому что плохих отзывов мало (~13%).

**Воспроизводимость:** `random_state = 42` при разбиении данных и обучении — результаты можно повторить.
"""

FINAL_MODEL_MD = """### Почему финальная модель — Random Forest

| Модель | F1 на тесте | ROC-AUC на тесте |
|--------|-------------|------------------|
| Random Forest | **~0.48** | ~0.83 |
| LightGBM | ~0.48 | чуть выше |
| Gradient Boosting | ниже | самый высокий ROC-AUC |

**Главное правило:** выбираем модель с **лучшим F1 на проверочной выборке**, а не с максимальным ROC-AUC.

**Почему так:**

1. **F1** отвечает: «находим ли мы недовольных клиентов?»
2. **ROC-AUC** отвечает: «умеем ли мы упорядочить заказы по риску?»
3. У Gradient Boosting ROC-AUC самый высокий, но **F1 сильно ниже** — модель реже ставит метку «недоволен» и пропускает проблемные заказы.
4. Random Forest **стабильно лучший по F1** на validation — поэтому сохраняем его в `models/best_model.pkl`.

**Коротко:** ROC-AUC выше ≠ модель лучше для нашей задачи. Нам важнее **находить недовольных** → Random Forest.
"""


def set_cell_source(nb, index, text):
    nb["cells"][index]["source"] = [text]


def patch_eda():
    p = ROOT / "notebooks" / "1_eda.ipynb"
    nb = json.loads(p.read_text(encoding="utf-8"))
    set_cell_source(nb, 0, EDA_INTRO)

    insight_idx = 0
    for cell in nb["cells"]:
        src = "".join(cell.get("source", []))
        if cell["cell_type"] == "markdown" and (
            src.startswith("**График") or src.startswith("**Тепловая") or src.startswith("**Утечка")
        ):
            if insight_idx < len(INSIGHTS):
                cell["source"] = [INSIGHTS[insight_idx]]
                insight_idx += 1
    p.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")


def patch_baseline():
    p = ROOT / "notebooks" / "2_baseline.ipynb"
    nb = json.loads(p.read_text(encoding="utf-8"))
    for cell in nb["cells"]:
        if cell["cell_type"] == "markdown" and "Baseline" in "".join(cell.get("source", [])):
            cell["source"] = [BASELINE_MD]
            break
    p.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")


def patch_experiments():
    p = ROOT / "notebooks" / "3_experements.ipynb"
    nb = json.loads(p.read_text(encoding="utf-8"))
    for cell in nb["cells"]:
        src = "".join(cell.get("source", []))
        if cell["cell_type"] == "markdown" and "Выбор финальной" in src:
            cell["source"] = [FINAL_MODEL_MD]
    p.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    patch_eda()
    patch_baseline()
    patch_experiments()
    print("Тексты в ноутбуках обновлены.")
