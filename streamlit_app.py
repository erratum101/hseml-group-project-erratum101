"""Простой веб-интерфейс поверх той же модели, что и FastAPI."""

from __future__ import annotations

import os
import sys

import streamlit as st

ROOT = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(ROOT, "src"))

from bird_mood import pick_bird  # noqa: E402
from config import FEATURE_COLUMNS  # noqa: E402
from inference import ModelNotLoadedError, Predictor  # noqa: E402

st.set_page_config(page_title="Предсказание недовольства", layout="wide")

st.markdown(
    """
<style>
  .block-container { padding-top: 1.5rem; max-width: 1100px; }
  .result-card {
    padding: 1.25rem 1.25rem 1rem 1.25rem;
    background: rgba(255,255,255,0.03);
  }
</style>
""",
    unsafe_allow_html=True,
)

st.title("Недоволен ли покупатель?")
st.caption("Главное: вероятность недовольства → пояснение → картинка-подсказка.")

predictor = Predictor()

defaults = {
    "payment_value": 150.0,
    "price": 120.0,
    "freight_value": 25.0,
    "items_count": 2,
    "payment_installments": 3,
    "delivery_time_days": 12.0,
    "delivery_delay_days": 5.0,
    "review_text_length": 80,
}

result_placeholder = st.empty()

with st.container():
    st.subheader("Признаки заказа")
    cols = st.columns(2)
    values: dict[str, float | int] = {}
    for i, name in enumerate(FEATURE_COLUMNS):
        with cols[i % 2]:
            if name in {"items_count", "payment_installments", "review_text_length"}:
                values[name] = st.number_input(name, min_value=0, value=int(defaults[name]), step=1)
            else:
                values[name] = st.number_input(name, value=float(defaults[name]), format="%.2f")

    predict_clicked = st.button("Предсказать", type="primary", use_container_width=True)

if predict_clicked:
    try:
        out = predictor.predict(values)
    except ModelNotLoadedError as exc:
        st.error(str(exc))
        st.stop()

    prob = out["probability"]
    bird = pick_bird(prob)

    with result_placeholder.container():
        st.markdown('<div class="result-card">', unsafe_allow_html=True)
        left, right = st.columns([2, 3], vertical_alignment="center")

        with left:
            st.metric("Вероятность недовольства", f"{prob:.1%}")
            st.progress(min(max(prob, 0.0), 1.0))
            st.markdown(f"### {bird.phrase}")
            st.caption(f"Диапазон: {bird.range_label} · модель: {out['model_name']}")

        with right:
            if os.path.isfile(bird.path):
                st.image(bird.path, use_container_width=True)
            else:
                st.warning(f"Картинка не найдена: {bird.filename}")

        st.markdown("</div>", unsafe_allow_html=True)
else:
    with result_placeholder.container():
        st.markdown(
            '<div class="result-card">Нажмите «Предсказать» — здесь появятся картинка, процент и пояснение.</div>',
            unsafe_allow_html=True,
        )
