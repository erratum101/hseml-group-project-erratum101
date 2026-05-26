import os
import sys

ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))

from bird_mood import BIRD_MOODS, pick_bird  # noqa: E402


def test_pick_bird_boundaries():
    assert pick_bird(0.0).filename == "Да.webp"
    assert pick_bird(0.149).filename == "Да.webp"
    assert pick_bird(0.15).filename == "Врядли.png"
    assert pick_bird(0.49).filename == "50на50.jpg"
    assert pick_bird(0.50).filename == "50на50.jpg"
    assert pick_bird(0.51).filename == "возможно.webp"
    assert pick_bird(0.999).filename == "Неуверен.webp"


def test_bird_files_exist():
    for mood in BIRD_MOODS:
        assert os.path.isfile(mood.path), mood.filename
