# Dockerfile для ML проекта
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["sh", "-c", "python src/preprocessing.py && python src/modeling.py && pytest tests/ -q"]
