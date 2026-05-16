.PHONY: lint lint-flake8 preprocess train test docker-build

lint: lint-flake8
	ruff check src/ --line-length 120

lint-flake8:
	flake8 src/ --max-line-length=120 --extend-ignore=E501

preprocess:
	python src/preprocessing.py

train:
	python src/modeling.py

test:
	pytest tests/ -q

docker-build:
	docker compose build
