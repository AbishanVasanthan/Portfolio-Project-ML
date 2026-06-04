# Tokyo Cement Demand Forecasting — System Architecture

End-to-end documentation for the two-repo system:

- **Portfolio-Project-ML** — the XGBoost forecasting engine (training + serving).
- **demand-forecast-dashboard** — the CemCast AI web app (FastAPI backend +
  React frontend) used by depot managers and planners.

They are independent apps connected by **one shared Supabase database** and an
**HTTP trigger** from the dashboard to the ML service.

---

## 1. High-level diagram

```
                         ┌──────────────────────────┐
                         │        DagsHub           │
                         │   MLflow model registry  │
                         │  cement_demand_forecaster_h1..h6   (depot, MAPE ~22%) │
                         │  cement_sku_forecaster_h1..h6      (SKU,   MAPE ~3.5%)│
                         └────────────▲───────┬─────┘
                            writes     │       │  loads models
                          (weekly      │       │
                           training)   │       ▼
   GitHub Actions ───────────►  Portfolio-Project-ML  ◄───── Render (always-on API)
   (Sun 01:00 UTC cron)         pipeline.py modes            https://portfolio-project-ml.onrender.com
        update→train            update / train / forecast            │ POST /forecast/all
                                train_sku / forecast_sku_all          │ POST /retrain
                                       │                              │ (X-Admin-Key)
                                       │ writes forecasts             │
                                       ▼                              │
                         ┌────────────────────────────────────────┐  │
                         │                Supabase                 │  │
                         │  tc_depots, tc_demand_panel             │  │
                         │  tc_forecasts          (depot forecasts)│  │
                         │  tc_skus, tc_sku_demand_panel           │  │
                         │  tc_sku_forecasts      (SKU forecasts)  │  │
                         │  tc_sales_actuals, tc_retrain_log, …    │  │
                         └────────────▲────────────────▲──────────┘  │
                            SQLAlchemy │                │ reads        │ trigger
                                       │                │              │
                         ┌─────────────┴────────────────┴──────────┐  │
                         │      demand-forecast-dashboard           │──┘
                         │      backend (FastAPI, CemCast AI)       │
                         │  reads tc_* forecasts, writes actuals    │
                         │  triggers ML API on Generate / Retrain   │
                         └────────────────────▲─────────────────────┘
                                              │ REST (JWT)
                         ┌────────────────────┴─────────────────────┐
                         │      frontend (React / TanStack Start)    │
                         │  Forecast page: depot + per-product views │
                         │  + "Enter Weekly Actual Demand" form      │
                         └──────────────────────────────────────────┘
```

---

## 2. Components

| Component | Tech | Hosting | Role |
|---|---|---|---|
| ML engine | Python, XGBoost, Optuna, MLflow | Render (web service) + GitHub Actions (cron) | Train models, write forecasts to Supabase, serve on-demand forecasts |
| Model registry | MLflow on DagsHub | DagsHub (free) | Versioned Production models, shared by Actions (writer) and Render (reader) |
| Database | PostgreSQL | Supabase | Single source of truth shared by both apps |
| Dashboard backend | FastAPI, SQLAlchemy | Docker / any host | Auth, business logic, reads forecasts, writes actuals, triggers ML |
| Dashboard frontend | React, TanStack, Vite | Cloudflare / static host | UI for managers and planners |

---

## 3. Two model families

| Family | Registry names | Grain | Rows trained | MAPE |
|---|---|---|---|---|
| Depot total | `cement_demand_forecaster_h1..h6` | depot × week | ~20.5k | ~22 % |
| Per product | `cement_sku_forecaster_h1..h6` | depot × SKU × week | ~123k (capped 156 wks/group at train) | ~3.5 % |

Each family is **6 direct-multistep XGBoost models** (one per horizon t+1…t+6),
trained with rolling-window CV and Optuna, promoted to MLflow **Production** only
if MAPE improves.

Products: SuperMix, SuperFix, SuperSeal, SuperSet, SuperScreed, SuperFlow.

---

## 4. Data flow

### Reads (what the user sees)
```
Frontend → GET /api/forecast/forecasts/{depot_id}     → tc_forecasts        (depot)
Frontend → GET /api/forecast/sku-summary/{depot_id}    → tc_sku_forecasts    (per product)
```
The dashboard never calls DagsHub or Render for reads — it reads forecasts that
already landed in Supabase.

### Writes (what keeps the DB consistent)
```
Manager enters weekly sales  → POST /api/forecast/sku-actuals
                             → tc_sku_demand_panel rows flip to data_source='actual'
                             → used as ground truth on the next train_sku run
```

### On-demand (button click)
```
"Generate Forecast"  → dashboard POST /api/forecast/generate
   1. fires ML POST /forecast/all (non-blocking) → XGBoost writes tc_forecasts
   2. dashboard's own sklearn model returns immediately (fallback)
   3. seconds later XGBoost results overwrite sklearn in the shared DB
```

---

## 5. Scheduler (automated weekly training)

**GitHub Actions** — `.github/workflows/weekly_pipeline.yml` in the ML repo.

- **Trigger:** cron `0 1 * * 0` (Sundays 01:00 UTC) or manual *Run workflow*.
- **Steps:**
  1. `python pipeline.py --mode update` — fetch fresh weather (Open-Meteo) +
     economics (World Bank) into `tc_demand_panel`.
  2. `python pipeline.py --mode train` — retrain depot models → DagsHub, then
     auto `forecast_all` → `tc_forecasts`.
  3. `curl POST {RENDER_API_URL}/reload-models` (if secret set) — tell the live
     Render server to hot-load the new models.

> SKU models are **not** in the weekly cron by default. To include them, add a
> step running `python pipeline.py --mode train_sku` (it also pushes SKU
> forecasts). It is heavier (~20 min) than the depot run.

**Required GitHub Secrets** (ML repo → Settings → Secrets → Actions):
```
DATABASE_URL, MLFLOW_TRACKING_URI, MLFLOW_TRACKING_USERNAME,
MLFLOW_TRACKING_PASSWORD, RENDER_API_URL, ADMIN_API_KEY
```

---

## 6. Manual training methods

Run from the ML repo with the venv activated and `.env` populated.

| Command | Use when |
|---|---|
| `python pipeline.py --mode setup` | First-time only — download Kaggle data, augment, seed depots + `tc_demand_panel`. |
| `python pipeline.py --mode update` | Pull latest weather/economics into the DB. |
| `python pipeline.py --mode train` | Retrain the 6 depot models + push depot forecasts. |
| `python pipeline.py --mode forecast_all` | Re-push depot forecasts without retraining. |
| `python pipeline.py --mode setup_skus` | One-time — run `src/db/sku_schema.sql` first, then seed `tc_sku_demand_panel`. |
| `python pipeline.py --mode train_sku` | Retrain the 6 SKU models + push SKU forecasts. |
| `python pipeline.py --mode forecast_sku_all` | Re-push SKU forecasts without retraining. |
| `python pipeline.py --mode serve` | Start the API locally. |

**Trigger a retrain from the dashboard UI** ("Retrain" button) → calls ML
`POST /retrain`. **Trigger from the API directly:**
```
POST {ML_API_URL}/retrain          (depot)   header X-Admin-Key: <ADMIN_API_KEY>
POST {ML_API_URL}/forecast/sku/all (SKU)      header X-Admin-Key: <ADMIN_API_KEY>
```

**Windows training notes** (low-RAM / full C: drive):
- Set BLAS threads to 1 (`OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`,
  `MKL_NUM_THREADS=1`) — `train_sku.py` already sets these.
- Redirect temp to a drive with space: `$env:TEMP="D:\tmp"`.
- All writes are idempotent (upsert) — a network drop mid-run is safe to re-run.

---

## 7. Deployment notes

- **Render (ML API):** deploy from the ML repo (`render.yaml` / `Dockerfile`).
  Set env vars: `DATABASE_URL`, `MLFLOW_TRACKING_*`, `ADMIN_API_KEY`.
- **IPv6 gotcha:** Supabase's direct host `db.<ref>.supabase.co` is **IPv6-only**;
  Render free tier has no outbound IPv6. Use the **session pooler** host on
  Render:
  ```
  postgresql+psycopg://postgres.<ref>:<pw>@aws-1-<region>.pooler.supabase.com:5432/postgres?sslmode=require
  ```
  (Local machines with IPv6 can use either host.)
- **DagsHub:** create a free repo, copy the `.mlflow` tracking URI, generate a
  token. Run one `--mode train` locally to seed Production models before the
  first Render deploy.

---

## 8. Shared secret summary

One `ADMIN_API_KEY` value is used in three places and must match:
```
ML repo .env / Render env   → ADMIN_API_KEY        (server validates)
GitHub Actions secret       → ADMIN_API_KEY        (cron sends on /reload-models)
dashboard backend .env      → ML_ADMIN_API_KEY     (dashboard sends on /forecast/all, /retrain)
```

`JWT_SECRET` (dashboard backend only) is **separate** — it signs user login
tokens and must not be reused as the admin key.
