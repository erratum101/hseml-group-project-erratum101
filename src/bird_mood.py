from __future__ import annotations

import os
from dataclasses import dataclass

BIRDS_DIR = os.path.join(os.path.dirname(__file__), "birds")


@dataclass(frozen=True)
class BirdMood:
    min_prob: float
    max_prob: float
    filename: str
    phrase: str

    @property
    def path(self) -> str:
        return os.path.join(BIRDS_DIR, self.filename)

    @property
    def range_label(self) -> str:
        lo = int(round(self.min_prob * 100))
        hi = int(round(self.max_prob * 100))
        if hi >= 100:
            return f"{lo}–100%"
        return f"{lo}–{hi}%"


BIRD_MOODS: tuple[BirdMood, ...] = (
    BirdMood(0.00, 0.15, "Да.webp", "Да — клиент, скорее всего, доволен."),
    BirdMood(0.15, 0.30, "Врядли.png", "Вряд ли клиент недоволен."),
    BirdMood(0.30, 0.40, "Неточно.jpg", "Неточно — однозначно не скажешь."),
    BirdMood(0.40, 0.49, "Естьсомнения.png", "Есть сомнения — пока рано делать вывод."),
    BirdMood(0.49, 0.51, "50на50.jpg", "50 на 50 — ровно на границе."),
    BirdMood(0.51, 0.60, "возможно.webp", "Возможно, клиент недоволен."),
    BirdMood(0.60, 0.75, "Сомнительно.jpg", "Сомнительно, что клиент доволен."),
    BirdMood(0.75, 1.001, "Неуверен.webp", "Высокий риск недовольства."),
)


def pick_bird(dissatisfaction_prob: float) -> BirdMood:
    p = min(max(dissatisfaction_prob, 0.0), 1.0)
    for mood in BIRD_MOODS:
        if mood.min_prob <= p < mood.max_prob:
            return mood
    return BIRD_MOODS[-1]
