from __future__ import annotations

from pydantic import BaseModel, Field


class OrderFeatures(BaseModel):
    """Признаки одного заказа для предсказания недовольства."""

    payment_value: float = Field(..., ge=0, description="Сумма оплаты, BRL")
    price: float = Field(..., ge=0, description="Сумма товаров, BRL")
    freight_value: float = Field(..., ge=0, description="Стоимость доставки, BRL")
    items_count: int = Field(..., ge=1, description="Число позиций в заказе")
    payment_installments: int = Field(..., ge=1, description="Число платежей")
    delivery_time_days: float = Field(..., ge=0, description="Дней от покупки до доставки")
    delivery_delay_days: float = Field(..., description="Задержка относительно обещанной даты, дней")
    review_text_length: int = Field(..., ge=0, description="Длина текста отзыва, символов")


class PredictionResponse(BaseModel):
    dissatisfied: bool = Field(..., description="True — модель считает клиента недовольным")
    probability: float = Field(..., ge=0, le=1, description="Вероятность класса «недоволен»")
    label: int = Field(..., description="0 — доволен, 1 — недоволен")
    model_name: str = Field(..., description="Имя загруженной модели")


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_path: str
