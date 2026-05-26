from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from inference import ModelNotLoadedError, get_predictor
from schemas import HealthResponse, OrderFeatures, PredictionResponse

EXAMPLE_ORDER = OrderFeatures(
    payment_value=150.0,
    price=120.0,
    freight_value=25.0,
    items_count=2,
    payment_installments=3,
    delivery_time_days=12.0,
    delivery_delay_days=5.0,
    review_text_length=80,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    predictor = get_predictor()
    try:
        predictor.load()
    except ModelNotLoadedError:
        pass
    yield


app = FastAPI(
    title="Olist Dissatisfaction Predictor",
    description=(
        "API предсказывает, останется ли покупатель недоволен заказом "
        "(оценка отзыва 1–2). Главная метрика при обучении — F1 для класса «недоволен»."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    predictor = get_predictor()
    return HealthResponse(
        status="ok",
        model_loaded=predictor.is_loaded,
        model_path=predictor.model_path,
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(order: OrderFeatures) -> PredictionResponse:
    predictor = get_predictor()
    try:
        result = predictor.predict(order.model_dump())
    except ModelNotLoadedError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return PredictionResponse(**result)


@app.get("/example", response_model=OrderFeatures)
def example_order() -> OrderFeatures:
    """Пример тела запроса для POST /predict."""
    return EXAMPLE_ORDER
