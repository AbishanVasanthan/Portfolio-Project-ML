# Tokyo Cement Demand Forecasting System

An end-to-end AI-driven pipeline that forecasts weekly cement demand at depot level across Sri Lanka, 6 weeks ahead. Built for Tokyo Cement as a LoCoders Data Science portfolio project.

The system ingests public data (Kaggle, Open-Meteo, World Bank), transforms it to a Sri Lankan business context, trains a global XGBoost model, serves forecasts via a REST API, and continuously improves as depot managers submit real sales data.

---

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Prerequisites](#prerequisites)
- [Environment Setup](#environment-setup)
- [Installation](#installation)
- [Pipeline Commands](#pipeline-commands)
  - [--mode setup](#--mode-setup)
  - [--mode update](#--mode-update)
  - [--mode train](#--mode-train)
  - [--mode serve](#--mode-serve)
- [Project Structure](#project-structure)
- [Data Sources](#data-sources)
- [Augmentation Pipeline](#augmentation-pipeline)
- [Feature Engineering](#feature-engineering)
- [Model Design](#model-design)
- [API Reference](#api-reference)
- [Database Schema](#database-schema)
- [MLflow & DagsHub](#mlflow--dagshub)
- [Configuration Reference](#configuration-reference)
- [Known Data Gaps](#known-data-gaps)
- [Success Criteria](#success-criteria)

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        pipeline.py                              │
│              Single entry point — four modes                    │
└────────┬──────────┬──────────┬──────────────────────────────────┘
         │          │          │
    setup │    update│    train │    serve
         ▼          ▼          ▼          ▼
┌──────────────┐  ┌──────┐  ┌────────────────────┐  ┌──────────┐
│  Ingestion   │  │Tier3 │  │  train.py          │  │ app.py   │
│  ─────────── │  │Weather  │  ─────────────────  │  │ FastAPI  │
│  Kaggle CSV  │  │only  │  │  Rolling-window CV │  │ 19 REST  │
│  Open-Meteo  │  └──────┘  │  Optuna tuning     │  │ endpoints│
│  World Bank  │            │  6 XGBoost models  │  └────┬─────┘
│  CBSL PMI    │            │  MLflow logging    │       │
│  LKA Calendar│            └─────────┬──────────┘       │
└──────┬───────┘                      │                   │
       │                              ▼                   │
       ▼                    ┌──────────────────┐          │
┌──────────────┐            │  evaluate.py     │          │
│ Augmentation │            │  ─────────────── │          │
│  ──────────  │            │  MAPE / bias     │          │
│  Replace econ│            │  SHAP analysis   │          │
│  Scale → LKA │            │  9 plot types    │          │
│  → weekly    │            │  → model_plots   │          │
│  Seasonal adj│            └─────────┬────────┘          │
│  24 depots   │                      │                   │
└──────┬───────┘                      │                   │
       │                              │                   │
       ▼                              ▼                   ▼
┌──────────────────────────────────────────────────────────────┐
│                  Supabase PostgreSQL                          │
│  depots · demand_panel · forecasts · stock_levels            │
│  purchase_orders · alerts · sales_actuals                    │
│  retrain_log · model_plots                                   │
└──────────────────────────────────────────────────────────────┘
                              ▲
                    MLflow Model Registry
               (local mlruns/ or DagsHub hosted)
```

**Data flow in one sentence:** Public data is ingested → augmented to Sri Lankan context → joined into a ~16,000-row panel → used to train 6 XGBoost models → served via FastAPI → continuously retrained as real sales data arrives.

---

## Prerequisites

- Python 3.11+
- Access to the shared Supabase PostgreSQL database (connection string is pre-filled in `.env.example`)
- Internet access for Open-Meteo and World Bank API calls during `--mode setup`
- No GPU required — XGBoost runs on CPU

---

## Environment Setup

Copy the example env file:

```bash
cp .env.example .env
```

The `.env.example` already contains the shared project credentials:

```
# Database — Supabase PostgreSQL
DATABASE_URL=postgresql://postgres:...@db.hcpyyeitixyvcoeritct.supabase.co:5432/postgres

# Kaggle — KGAT-style token (no username needed, picked up automatically by kagglehub)
KAGGLE_API_TOKEN=KGAT_66b7b9b660696d7f3936a7443fe27c73

# MLflow — leave blank to use local mlruns/
MLFLOW_TRACKING_URI=
MLFLOW_TRACKING_USERNAME=
MLFLOW_TRACKING_PASSWORD=

# API server
API_HOST=0.0.0.0
API_PORT=8000
```

`DATABASE_URL` and `KAGGLE_API_TOKEN` are filled in — you do not need to change them. The MLflow variables are only needed when you are ready to push runs to DagsHub (see [MLflow & DagsHub](#mlflow--dagshub)).

---

## Installation

```bash
pip install -r requirements.txt
```

Key dependencies: `kagglehub`, `pandas`, `numpy`, `xgboost`, `scikit-learn`, `optuna`, `mlflow`, `shap`, `fastapi`, `uvicorn`, `psycopg2-binary`, `wbgapi`, `python-dotenv`, `pyyaml`, `matplotlib`, `seaborn`.

---

## Pipeline Commands

Every stage of the system runs as a single command. There are exactly four modes.

### First-time run

```bash
python pipeline.py --mode setup    # run once — takes 10–20 min (API calls for 24 depots)
python pipeline.py --mode train    # run once after setup
python pipeline.py --mode serve    # start the API
```

### Ongoing cadence (once live)

```bash
python pipeline.py --mode update   # run weekly — fetches new weather + economic data
python pipeline.py --mode train    # run after update to refresh the model
python pipeline.py --mode serve    # keep running as a long-lived process
```

---

### `--mode setup`

**Purpose:** Full first-time initialisation. Downloads all data, runs the augmentation pipeline, builds the feature panel, creates the database schema, and seeds the database.

**Steps executed in order:**

| Step | What happens |
|---|---|
| 1 | Download Kaggle cement dataset (`kishorkhengare/cement-sales-demand`) to `data/raw/kaggle/` |
| 2 | Pull weather data from Open-Meteo for all 24 depots — ERA5 (2010–2022), historical forecast archive (2022–92 days ago), and current rolling window |
| 3 | Pull World Bank annual indicators for Sri Lanka (GDP, population, lending rate, govt consumption) and interpolate to weekly; download Pink Sheet metals index as clinker proxy |
| 4 | Build Sri Lanka ISO-week calendar table (2010–2030) with monsoon flags, holiday flags, and fiscal quarter markers |
| 5 | Run all 5 augmentation steps (see [Augmentation Pipeline](#augmentation-pipeline)) |
| 6 | Join all sources into the final modelling panel (`data/processed/panel_modelling.csv`) |
| 7 | Run `src/db/schema.sql` against Supabase to create all 8 tables (idempotent — safe to re-run) |
| 8 | Seed the `depots` table (24 rows) and `demand_panel` table (~16,000 rows) using bulk insert with `ON CONFLICT DO NOTHING` |
| 9 | Print summary: rows written, date range, depots seeded |

If any step fails, the pipeline stops immediately and prints which step failed and why. It does not silently continue.

**Idempotent:** Running `--mode setup` a second time is safe. Files that already exist are skipped; DB rows that already exist are skipped.

---

### `--mode update`

**Purpose:** Fetch fresh data for weeks not yet in the database. Run weekly to keep the system current.

**Steps executed in order:**

| Step | What happens |
|---|---|
| 1 | Query `demand_panel` to find the latest `week_start` in the DB |
| 2 | Pull Tier 3 weather data (Open-Meteo current rolling window, past 92 days) for all 24 depots |
| 3 | Refresh World Bank economic data (removes cached CSV and re-fetches) |
| 4 | Append new weekly rows to `demand_panel` with `data_source = 'augmented'` — these will be overwritten with `data_source = 'actual'` when a manager submits real sales |
| 5 | Print summary: weeks added, depots updated |

**Append-only:** This mode never modifies existing rows. It only inserts new weeks that are not yet in the database.

---

### `--mode train`

**Purpose:** Train (or retrain) the XGBoost forecasting models on whatever is currently in the database and evaluate them.

**Steps executed in order:**

| Step | What happens |
|---|---|
| 1 | Pull the full `demand_panel` from the database and rebuild all lag features |
| 2 | Run rolling-window cross-validation (104-week train window, 6-week val window, 6-week step, minimum 5 folds) |
| 3 | Tune XGBoost hyperparameters with Optuna (50 trials, minimising mean MAPE across CV folds) |
| 4 | Train 6 final XGBoost models — one per forecast horizon (t+1 through t+6) |
| 5 | Evaluate: MAPE per depot, per horizon, per season; bias analysis; SHAP; feature importance |
| 6 | Compare to the current Production model in MLflow. If new MAPE ≤ previous MAPE, promote to Production |
| 7 | Save all evaluation plots as base64 PNG to the `model_plots` table in Supabase |
| 8 | Write a row to `retrain_log` with full audit trail |
| 9 | Print summary line |

**Output:**
```
[TRAIN] Current production model: version 3 | trained 2025-05-21 | MAPE 11.4% | promoted: yes
[TRAIN] Previous MAPE: 12.1%
```

**Promotion rule:** A new model is promoted to Production only if its average MAPE across all 6 horizons is equal to or better than the current Production model. A worse model is logged but not promoted — the previous model keeps serving.

---

### `--mode serve`

**Purpose:** Load all 6 horizon models from the MLflow registry and start the FastAPI server.

```
[PIPELINE] Starting FastAPI on 0.0.0.0:8000
```

The API is then available at `http://localhost:8000`. Interactive docs at `http://localhost:8000/docs`.

The serve mode loads models at startup. If models are not yet trained, it starts with a 503 on forecast endpoints and logs a warning. All other endpoints (depots, stock, sales, alerts) work without trained models.

---

## Project Structure

```
.
├── pipeline.py                  # Single entry point — all four modes
├── config.yaml                  # Single source of truth for all parameters
├── requirements.txt
├── .env.example                 # Committed — DB and Kaggle creds pre-filled
├── .env                         # Gitignored — copy from .env.example
│
├── src/
│   ├── ingestion/
│   │   ├── kaggle_ingest.py     # Download Kaggle dataset via kagglehub
│   │   ├── weather_ingest.py    # 3-tier Open-Meteo weather fetcher (ERA5, Tier2, Tier3)
│   │   ├── economic_ingest.py   # World Bank wbgapi + Pink Sheet + CBSL PMI loader
│   │   └── calendar_build.py   # Sri Lanka ISO-week calendar table (2010–2030)
│   │
│   ├── augmentation/
│   │   ├── replace_economics.py # Drop Indian econ columns, join World Bank LKA equivalents
│   │   ├── scale_to_lka.py      # Scale Indian volumes → Tokyo Cement LKA; apply 2022 crisis shock
│   │   ├── disaggregate_weekly.py # Monthly → weekly using within-month demand weights
│   │   ├── seasonal_override.py # Strip Indian seasonality; apply LKA monsoon-driven curve
│   │   └── split_to_depots.py   # National series → 24 depots via population weights + noise
│   │
│   ├── features/
│   │   └── build_features.py    # Join all sources; lag features; interaction terms; depot encoding
│   │
│   ├── model/
│   │   ├── train.py             # Rolling-window CV, Optuna, XGBoost × 6 horizons, MLflow logging
│   │   ├── evaluate.py          # MAPE/bias/SHAP breakdowns; 9 plot types → model_plots DB table
│   │   └── predict.py           # Load models from MLflow registry; construct feature row; inference
│   │
│   ├── db/
│   │   ├── schema.sql           # 8-table PostgreSQL schema (idempotent)
│   │   ├── db.py                # Shared psycopg2 connection pool (ThreadedConnectionPool)
│   │   └── seed.py              # Bulk-insert depots and demand_panel (ON CONFLICT DO NOTHING)
│   │
│   └── serve/
│       └── app.py               # FastAPI app — 19 endpoints, background retrain, alert logic
│
├── data/                        # Gitignored — generated by --mode setup
│   ├── raw/
│   │   ├── kaggle/              # Downloaded Kaggle CSV
│   │   ├── weather/             # One CSV per depot (weekly aggregated)
│   │   ├── economic/            # worldbank_lka.csv, pink_sheet_metals.csv, cbsl_pmi_weekly.csv
│   │   └── calendar/            # lka_calendar.csv
│   ├── interim/                 # Augmentation stage outputs
│   └── processed/               # panel_modelling.csv — final feature panel
│
└── results/                     # Gitignored — temporary plot scratch space during training
```

---

## Data Sources

### Kaggle — Cement Sales Dataset

Source: `kishorkhengare/cement-sales-demand`

Monthly CSV with columns: `Month`, `Production`, `Sales`, `Demand`, `Population`, `GDP`, `Disbursement`, `Interest_Rate`. Date range January 2010 – November 2022.

This is Indian national cement data. The augmentation pipeline (see below) transforms it to a Sri Lankan, Tokyo Cement-specific context.

### Open-Meteo Weather API

Free, no API key required. Weather data is pulled for all 24 depot coordinates using a three-tier strategy to cover the full date range without gaps:

| Tier | Endpoint | Coverage |
|---|---|---|
| Tier 1 — ERA5 historical | `archive-api.open-meteo.com/v1/archive` | 2010-01-01 → 2022-11-30 |
| Tier 2 — Historical forecast archive | `historical-forecast-api.open-meteo.com/v1/forecast` | 2022-12-01 → 92 days ago |
| Tier 3 — Current rolling window | `api.open-meteo.com/v1/forecast` | Past 92 days (used by `--mode update`) |

Hourly variables: `temperature_2m`, `relative_humidity_2m`, `precipitation`, `rain`, `cloud_cover`. Aggregated to weekly (precipitation as **sum**, all others as **mean**). One CSV per depot saved to `data/raw/weather/`.

### World Bank API

Fetched via the `wbgapi` Python library — no API key required. Four annual indicators for Sri Lanka (`LKA`):

| Indicator | Code | Used for |
|---|---|---|
| GDP in current LKR | `NY.GDP.MKTP.CN` | Economic context feature |
| Total population | `SP.POP.TOTL` | Scale validation |
| Lending interest rate | `FR.INR.LNDP` | Credit cost feature |
| Government consumption | `NE.CON.GOVT.CN` | Construction activity proxy |

Annual data is linearly interpolated to monthly, then forward-filled to weekly.

**Pink Sheet:** The World Bank Metals & Minerals index is downloaded from the CMO Pink Sheet Excel file and used as a clinker input cost proxy (no clinker-specific series exists in public data).

### CBSL Construction PMI

Downloaded manually from the Central Bank of Sri Lanka website and placed at `data/raw/economic/cbsl_pmi_construction.csv` (columns: `month, pmi_construction`). Available from approximately 2018 onwards. Pre-2018 weeks are backward-filled from the mean of the first 6 available readings — an acknowledged approximation. If the file is absent, the pipeline drops the PMI column and continues.

### Sri Lanka Calendar Table

Generated programmatically in `src/ingestion/calendar_build.py`. Covers 2010–2030. Includes monsoon season flags, Sinhala/Tamil New Year, Vesak Poya, Christmas, post-holiday lags, and fiscal Q4 marker. Never needs to be regenerated.

---

## Augmentation Pipeline

The Kaggle dataset is Indian national data. Five sequential steps transform it to a Tokyo Cement Sri Lanka context. Each step reads from the previous step's output.

### Step 1 — Replace Economics (`replace_economics.py`)

Drops the Indian economic columns (`Population`, `GDP`, `Disbursement`, `Interest_Rate`) and joins in the World Bank LKA equivalents aligned by month. The physical volumes (Production, Sales, Demand) remain unchanged at this step — they are still Indian quantities.

### Step 2 — Scale to LKA (`scale_to_lka.py`)

Computes a scalar factor from Tokyo Cement's known financials:

```
Tokyo Cement annual revenue:  Rs 50.1 Bn
Average market price:         Rs 52,000 / tonne (Rs 1,300 per 50kg bag × 40 bags)
Implied annual volume:        50,100,000,000 / 52,000 ≈ 963,000 tonnes/year ≈ 80,250 t/month

scale_factor = 80,250 / mean(Kaggle Sales column)
```

Applies `scale_factor` to Production, Sales, and Demand. Then applies a **2022 crisis shock** — Sales and Demand are multiplied by `0.72` for all months in 2022, reflecting the severe construction collapse during Sri Lanka's economic crisis that year.

### Step 3 — Monthly to Weekly Disaggregation (`disaggregate_weekly.py`)

Expands each monthly row to 4 or 5 weekly rows using within-month demand weights that reflect front-loading of construction activity:

- 4-week month: `[0.29, 0.26, 0.25, 0.20]`
- 5-week month: `[0.24, 0.22, 0.21, 0.19, 0.14]`

Volume columns are split by weight. Economic columns are forward-filled (constant within the month).

### Step 4 — Seasonal Override (`seasonal_override.py`)

Strips the Indian seasonal pattern and replaces it with a Sri Lanka monsoon-driven curve:

1. Compute Indian seasonal index per calendar month: `SI[m] = mean_sales[m] / overall_mean_sales`
2. Deseasonalise: `Sales_deseas = Sales / SI[month]`
3. Build LKA multiplier from Colombo ERA5 precipitation data: `multiplier[week] = 1.15 - (precip_norm × 0.40)` — this gives a range of **0.75** (peak SW monsoon) to **1.15** (dry season peak)
4. Re-apply: `Sales_lka = Sales_deseas × multiplier[calendar_week]`

### Step 5 — Split to 24 Depots (`split_to_depots.py`)

Allocates the national weekly series across 24 depots using district population weights from the 2012 Sri Lanka Census. A ±3% depot-specific noise term (seeded for reproducibility) is added per depot per week to simulate real operational variation and prevent all depots from being perfectly correlated.

```python
depot_sales = national_sales × pop_weight × (1 + noise)
```

Output: long-format table `[week_start, depot, sales_tonnes, demand_tonnes, production_tonnes]` — approximately 672 weeks × 24 depots = ~16,000 rows.

---

## Feature Engineering

All feature engineering runs in `src/features/build_features.py` after joining the four data sources on `(week_start, depot)`.

### Target Variable

```
y = demand_tonnes (for that depot, that week)
```

### Lag Features (computed per depot, in week order)

| Feature | Description |
|---|---|
| `demand_lag_1` | Demand 1 week prior |
| `demand_lag_2` | Demand 2 weeks prior |
| `demand_lag_3` | Demand 3 weeks prior |
| `demand_lag_4` | Demand 4 weeks prior |
| `demand_lag_6` | Demand 6 weeks prior (matches the maximum forecast horizon) |
| `demand_lag_52` | Demand same week last year — captures annual seasonality |
| `demand_rolling_mean_4` | 4-week rolling average (short-term trend) |
| `demand_rolling_std_4` | 4-week rolling standard deviation (volatility signal) |
| `demand_rolling_mean_12` | 12-week rolling average (quarterly trend) |

All lag features are computed with `shift(1)` before rolling — no data leakage from the target week itself.

### Weather Features

| Feature | Kept | Reason |
|---|---|---|
| `precip_sum` | Yes | Primary monsoon signal |
| `rain_sum` | Yes | Distinction from total precipitation matters |
| `temp_mean` | Yes | Heat affects construction activity |
| `humidity_mean` | Yes | Correlated with monsoon intensity |
| `cloud_cover_mean` | Optional | Dropped if collinear with precipitation |

### Calendar Features

All columns from the calendar table: `is_sw_monsoon`, `is_ne_monsoon`, `is_dry_season`, `is_sinhala_tamil_new_year`, `is_vesak`, `is_christmas_week`, `post_holiday_lag_1`, `post_holiday_lag_2`, `is_year_end_quarter`.

### Economic Features

`gdp_lka`, `lending_rate`, `cbsl_pmi_construction`, `govt_consumption` — all weekly-interpolated from annual/monthly sources.

### Interaction Features

| Feature | Formula | Business intuition |
|---|---|---|
| `precip_x_monsoon` | `precip_sum × is_sw_monsoon` | Monsoon rainfall has a stronger dampening effect than off-season rain |
| `post_holiday_demand_boost` | `post_holiday_lag_1 × demand_rolling_mean_4` | Captures the construction restart surge after major holidays |

### Dropped Columns

`sales_tonnes` and `production_tonnes` are excluded from the feature set — they would be either data leakage or unavailable at prediction time.

---

## Model Design

### Global XGBoost — One Model for All 24 Depots

A single XGBoost model is trained on data from all 24 depots simultaneously. Depot identity is encoded as an integer feature (`depot_enc` — alphabetical label encoding). A global model learns cross-depot patterns and generalises better than 24 separate models on limited data.

### Direct Multi-Step Forecasting — 6 Separate Horizon Models

Six independent XGBoost regressors are trained, one per forecast horizon (t+1 through t+6). The target for horizon h is `demand_tonnes` shifted back by h weeks per depot. This **direct multi-step** approach is more stable than recursive forecasting over a 6-week horizon because errors do not compound.

MLflow model names: `cement_demand_forecaster_h1` through `cement_demand_forecaster_h6`.

### Rolling-Window Cross-Validation

| Parameter | Value |
|---|---|
| Training window | 104 weeks (2 years) |
| Validation window | 6 weeks (the forecast horizon) |
| Step size | 6 weeks |
| Minimum folds | 5 |

No random splits — the time ordering is always respected.

### Hyperparameter Tuning (Optuna)

50 trials of TPE (Tree-structured Parzen Estimator) search, minimising mean MAPE across CV folds.

| Hyperparameter | Search range |
|---|---|
| `n_estimators` | 200 – 800 |
| `max_depth` | 3 – 8 |
| `learning_rate` | 0.01 – 0.15 (log scale) |
| `subsample` | 0.6 – 1.0 |
| `colsample_bytree` | 0.6 – 1.0 |
| `min_child_weight` | 1 – 10 |

Tuning is run on the second-to-last CV fold for speed, then the best params are evaluated across all folds.

### MLflow Logging

Every training run logs:

- **Parameters:** all XGBoost hyperparameters, horizon number, number of features, training window size
- **Metrics:** `mape_val`, `mae_val`, `bias_val`, `mape_per_depot_{name}` for each of the 24 depots
- **Artifacts:** trained model (registered in MLflow Model Registry)

### Model Registry and Promotion

- Models are registered under `cement_demand_forecaster_h{1-6}` in the MLflow registry
- A new model is promoted to `Production` stage only if its average MAPE ≤ the current Production model's MAPE
- The `retrain_log` table records the MLflow version that was active as Production at the time of every training run — full audit trail
- Last 3 versions are always retained; old versions are archived rather than deleted

### Saved Evaluation Plots

After every training run, 9 plot types are rendered (headless, `Agg` backend) and saved as base64-encoded PNG to the `model_plots` database table. The frontend fetches them directly from the API — no file serving required.

| `plot_type` | Scope | Description |
|---|---|---|
| `mape_by_depot` | Global | Bar chart — MAPE per depot averaged across all 6 horizons |
| `mape_by_horizon` | Global | Line chart — accuracy decay from t+1 to t+6 |
| `mape_by_season` | Global | Bar chart — MAPE in SW monsoon vs non-monsoon |
| `forecast_vs_actual` | Global | Line chart — aggregated forecast vs actual, last CV fold |
| `bias_by_depot` | Global | Bar chart — signed error per depot (positive = overforecast) |
| `feature_importance` | Global | Top 20 features by XGBoost gain (horizon 1 model) |
| `shap_summary` | Global | SHAP beeswarm plot on 500-row sample |
| `retrain_history` | Global | MAPE trend across all retraining runs |
| `depot_forecast` | Per depot | 6-week forecast ribbon vs actuals — one per depot (24 rows) |

---

## API Reference

Base URL: `http://localhost:8000`  
Interactive docs: `http://localhost:8000/docs`

CORS is enabled for all origins during development (`allow_origins=["*"]`). Restrict to the frontend domain before any public deployment.

### Depots

| Method | Path | Description |
|---|---|---|
| `GET` | `/depots` | List all 24 depots with metadata (id, name, district, province, lat/lon) |

### Forecasting

| Method | Path | Description |
|---|---|---|
| `POST` | `/forecast` | Generate a fresh 6-week forecast for a depot as of a given date. Writes to `forecasts` table and triggers alert + PO evaluation. |
| `GET` | `/forecasts/{depot}` | Retrieve stored forecasts for a depot. Optional `?as_of_date=YYYY-MM-DD` filter; returns latest if omitted. |

`POST /forecast` request body:
```json
{ "depot": "Colombo", "as_of_date": "2022-10-01" }
```

`POST /forecast` response:
```json
{
  "depot": "Colombo",
  "as_of_date": "2022-10-01",
  "forecasts": [
    { "horizon": 1, "forecast_week": "2022-10-08", "demand_tonnes": 842.3 },
    { "horizon": 2, "forecast_week": "2022-10-15", "demand_tonnes": 791.1 },
    ...
  ],
  "generated_at": "2022-10-01T08:00:00+00:00"
}
```

### Stock Management

| Method | Path | Description |
|---|---|---|
| `POST` | `/stock` | Submit current stock level for a depot. Triggers alert re-evaluation. |
| `GET` | `/stock/{depot}` | Latest stock level + last 12 weeks of history. |

### Purchase Orders

| Method | Path | Description |
|---|---|---|
| `GET` | `/purchase-orders/{depot}` | List PO recommendations. Optional `?status=pending\|approved\|dismissed\|all`. |
| `PATCH` | `/purchase-orders/{po_id}` | Approve or dismiss a PO recommendation. |

PO quantity is computed as:
```
recommended_qty = max(0, forecast_week_1 × 1.25 - current_stock)
```

### Alerts

| Method | Path | Description |
|---|---|---|
| `GET` | `/alerts/{depot}` | Active (unresolved) alerts for a depot. Optional `?resolved=true` to include resolved. |
| `PATCH` | `/alerts/{alert_id}/resolve` | Mark an alert as resolved. |

Alert conditions evaluated after every forecast or stock update:
- **Critical low stock:** current stock < 80% of 2-week forecast demand
- **Warning low stock:** current stock < 90% of 4-week forecast demand
- **Demand spike:** week-1 forecast > 130% of 4-week rolling average
- **Overstock:** current stock > 150% of 6-week forecast demand

### Dashboard

| Method | Path | Description |
|---|---|---|
| `GET` | `/dashboard/{depot}` | All dashboard data in one call — depot metadata, latest stock, 6-week forecast, pending POs, active alerts. |

### Sales Actuals

| Method | Path | Description |
|---|---|---|
| `POST` | `/sales` | Submit actual sales figures for a depot-week. Syncs to `demand_panel` with `data_source = 'actual'`. Enqueues retrain. |
| `PUT` | `/sales/{depot}/{week_start}` | Correct a previously submitted sales entry. |
| `GET` | `/sales/{depot}` | Historical sales actuals. Optional `?weeks=12` (max 52). |

### Retraining

| Method | Path | Description |
|---|---|---|
| `POST` | `/retrain` | Manually trigger a retraining run (runs in background). |
| `GET` | `/retrain/status/{retrain_id}` | Poll the status of a specific retrain run. |
| `GET` | `/retrain/history` | Last 10 retraining runs with MAPE before/after and promotion status. |

**Auto-retrain rule:** Every sales submission (`POST /sales`, `PUT /sales`) increments a pending counter. When 5 or more new rows have accumulated since the last completed retrain, retraining triggers automatically in a background task. The threshold is configurable in `config.yaml` (`model.retrain_batch_size`).

### Plots

| Method | Path | Description |
|---|---|---|
| `GET` | `/plots/latest` | All global plots from the most recent completed retrain. Each item has `plot_type` and `image_data` (base64 PNG, usable directly as `<img src="...">` in the frontend). |
| `GET` | `/plots/depot/{depot}` | The `depot_forecast` plot for a specific depot from the latest retrain. |
| `GET` | `/plots/{retrain_id}` | All plots for a specific historical training run. Optional `?plot_type=` filter. |

---

## Database Schema

All tables live in the shared Supabase PostgreSQL instance. The schema is version-controlled at [src/db/schema.sql](src/db/schema.sql) and created by `--mode setup`.

| Table | Purpose |
|---|---|
| `depots` | Static reference — 24 depots with coordinates and population weights |
| `demand_panel` | Full weekly panel — one row per depot per week; the training data source and feature store for inference |
| `forecasts` | Every generated forecast, stored for audit and dashboard display |
| `stock_levels` | Current stock at each depot, submitted by depot managers |
| `purchase_orders` | Auto-generated order quantity recommendations |
| `alerts` | Low-stock, demand spike, and overstock alerts |
| `sales_actuals` | Real sales figures entered by depot managers — the live data that replaces augmented rows over time |
| `retrain_log` | Audit trail of every retraining run |
| `model_plots` | Base64-encoded PNG plots from every training run |

### `data_source` column in `demand_panel`

Every row in `demand_panel` has a `data_source` field:

- `'augmented'` — generated by the Kaggle augmentation pipeline
- `'actual'` — written or updated via `POST /sales` or `PUT /sales`

**Rule:** An `actual` row is never overwritten by `augmented` data. As depot managers submit real weekly sales, the augmented rows are replaced week by week. Once 6–12 months of real data accumulates, the Kaggle-derived rows become irrelevant to recent forecasts.

---

## MLflow & DagsHub

### Local development (default)

By default, all MLflow runs are logged to `mlruns/` in the project root. Start the MLflow UI with:

```bash
mlflow ui
```

Then open `http://localhost:5000` to browse experiments, compare runs, and inspect model versions.

### Deploying to DagsHub

When you are ready to host MLflow remotely:

1. Create a DagsHub account at `https://dagshub.com` and create a new repo linked to this GitHub repository
2. In your `.env`, fill in the three MLflow values from your DagsHub repo settings:
   ```
   MLFLOW_TRACKING_URI=https://dagshub.com/{username}/{repo}.mlflow
   MLFLOW_TRACKING_USERNAME={your_dagshub_username}
   MLFLOW_TRACKING_PASSWORD={your_dagshub_token}
   ```
3. Re-run `python pipeline.py --mode train` — runs will push to DagsHub automatically
4. Register the best model in the DagsHub Model Registry under `cement_demand_forecaster_h{1-6}`
5. `--mode serve` will pick up models from DagsHub with zero code changes

No code changes are needed at any point — the tracking URI is always read from the environment variable with a local fallback:

```python
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "mlruns"))
```

---

## Configuration Reference

All parameters live in [`config.yaml`](config.yaml). The pipeline reads this at startup and passes it down to every module. Nothing is hardcoded in scripts.

Key sections:

| Section | What it controls |
|---|---|
| `paths` | All input/output directory paths |
| `kaggle` | Dataset identifier |
| `weather` | Open-Meteo API URLs, date ranges, tier boundaries |
| `worldbank` | Indicator codes, year range, Pink Sheet URL |
| `augmentation` | Scale factors, crisis shock, weekly weights, seasonal formula, noise seed |
| `features` | Lag weeks list, rolling windows, drop columns |
| `model` | Horizons, CV parameters, Optuna trials, XGBoost search ranges, registry name, retrain batch size |
| `api` | Host and port |
| `alerts` | Thresholds for low-stock, spike, and overstock alert conditions |
| `purchase_orders` | Safety stock percentage |
| `depots` | All 24 depots — name, district, province, lat/lon, population weight |

---

## Known Data Gaps

These are deliberate engineering decisions, not bugs. The goal is a working, trainable pipeline — not perfect data.

| Gap | Handling |
|---|---|
| Weather Nov 2022 – 92 days ago | Filled via `historical-forecast-api.open-meteo.com` (Tier 2) |
| CBSL PMI 2010–2017 | Backward-filled from mean of first 6 available 2018 readings |
| CBSL PMI file absent entirely | Column dropped; pipeline continues with a warning |
| World Bank 2024+ publication lag (12–18 months) | Forward-filled from last available year |
| No clinker price series exists | World Bank Metals & Minerals index used as proxy |
| Kaggle data is Indian, not Sri Lankan | Full 5-step augmentation pipeline |
| No real Tokyo Cement historical sales data | Synthetic depot-level data from augmented Kaggle series; replaced week by week as managers submit actuals via `POST /sales` |

---

## Success Criteria

| Metric | Target |
|---|---|
| Average MAPE across all depots, all horizons | < 15% |
| MAPE at t+1 (1-week ahead) | < 10% |
| MAPE at t+6 (6-week ahead) | < 20% |
| No single depot with MAPE > 25% | All 24 depots |
| SHAP top-5 features make business sense | Manual review — expect `demand_lag_1`, `demand_lag_52`, `precip_sum`, `is_sw_monsoon`, and a calendar feature |
| `POST /forecast` response time | < 500ms |
| DB writes idempotent on re-run | Verified by running `--mode setup` twice |
| Auto-retrain triggers after 5 new sales submissions | Verified via `retrain_log` |
| Retrained model promoted only if MAPE improves | Verified via `mape_before` vs `mape_after` in `retrain_log` |
| All plots saved to `model_plots` after every train run | Minimum 32 rows: 8 global + 24 per-depot |
| Frontend can render plots from a single `GET /plots/latest` | No file serving needed — base64 inline |
