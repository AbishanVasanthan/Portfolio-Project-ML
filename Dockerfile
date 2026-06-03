FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p data/raw/kaggle data/raw/weather data/raw/economic data/raw/calendar \
    data/interim data/processed results mlruns

EXPOSE 8000

CMD ["python", "pipeline.py", "--mode", "serve"]
