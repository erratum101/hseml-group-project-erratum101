.PHONY: lint lint-flake8 preprocess train test docker-build serve-api serve-ui demo-model report-pdf

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

demo-model:
	python scripts/train_demo_model.py

serve-api:
	uvicorn api:app --host 127.0.0.1 --port 8000 --reload --app-dir src

serve-ui:
	streamlit run streamlit_app.py --server.port 8501

report-pdf:
	pandoc report/report.md -o report/report.pdf --from markdown --pdf-engine=xelatex -V geometry:margin=2.5cm
