# Tokyo Cement Demand Forecasting System

AI-driven weekly depot-level cement demand forecasting for Tokyo Cement Sri Lanka.

## Setup

Copy `.env.example` to `.env` and fill in any blank values:

```bash
cp .env.example .env
```

`DATABASE_URL` and `KAGGLE_API_TOKEN` are pre-filled. Set `MLFLOW_TRACKING_URI`, `MLFLOW_TRACKING_USERNAME`, and `MLFLOW_TRACKING_PASSWORD` when enabling DagsHub hosting (leave blank to use local `mlruns/`).

## Pipeline Commands

```bash
# First-time setup: ingest all data, augment, build features, seed the database
python pipeline.py --mode setup

# Train (or retrain) the model on whatever is currently in the database
python pipeline.py --mode train

# Start the API server
python pipeline.py --mode serve
```

Once live and accumulating real sales data:

```bash
# Fetch fresh weather + economic data for any weeks not yet in the DB
python pipeline.py --mode update

# Retrain after update
python pipeline.py --mode train
```
