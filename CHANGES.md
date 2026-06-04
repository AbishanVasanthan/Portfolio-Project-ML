# ML Project — Changes Log

This document records everything added/changed in the **Portfolio-Project-ML**
service (the XGBoost forecasting engine) on top of the original code base.

---

## 1. Database layer — moved off `supabase-py` to direct PostgreSQL

**File:** `src/db/db.py` (rewritten)

- Removed the `supabase-py` REST client. The project now talks to Postgres
  directly with **`psycopg`**, using a thin drop-in wrapper so the rest of the
  code keeps the same fluent API:
  `get_client().table("x").select().eq().order().range().execute()`.
- **Thread-local connection** (`_get_conn`) — one connection reused per thread
  instead of opening a new socket per query. This prevents Windows socket-quota
  exhaustion when paginating 100k+ rows.
- **`Decimal → float` coercion** on every result row so pandas arithmetic works.
- Added `.range(start, end)` (maps to `LIMIT/OFFSET`) for pagination.

**Why:** one credential (`DATABASE_URL`) instead of `SUPABASE_URL` + `SUPABASE_KEY`,
and the same connection string the dashboard backend already uses.

**Requirements:** `requirements.txt` — `supabase` replaced by `psycopg[binary]`.

---

## 2. Environment variables

`SUPABASE_URL` / `SUPABASE_KEY` → **`DATABASE_URL`** everywhere
(`pipeline.py` now requires `DATABASE_URL`).

```
DATABASE_URL=postgresql+psycopg://<user>:<pw>@<host>:5432/postgres?sslmode=require
MLFLOW_TRACKING_URI=https://dagshub.com/<user>/<repo>.mlflow
MLFLOW_TRACKING_USERNAME=<dagshub-user>
MLFLOW_TRACKING_PASSWORD=<dagshub-token>
ADMIN_API_KEY=<random-secret>            # protects POST /reload-models, /forecast/all, /forecast/sku/all
API_HOST=0.0.0.0
API_PORT=8000
```

> **Deployment note:** Supabase's *direct* host (`db.<ref>.supabase.co`) is
> **IPv6-only**. Render's free tier can't do outbound IPv6, so on Render you must
> use the **session pooler** host (IPv4):
> `postgresql+psycopg://postgres.<ref>:<pw>@aws-1-<region>.pooler.supabase.com:5432/postgres?sslmode=require`

---

## 3. New pipeline modes — `pipeline.py`

| Mode | What it does |
|---|---|
| `--mode forecast_all` | Loads the 6 depot models from MLflow and writes 6-week forecasts for **all depots** → `tc_forecasts`. Also called automatically at the end of `--mode train`. |
| `--mode setup_skus` | Seeds `tc_skus` + generates synthetic per-product demand into `tc_sku_demand_panel` from `tc_demand_panel`. |
| `--mode train_sku` | Trains 6 XGBoost **SKU** models and pushes per-product forecasts. |
| `--mode forecast_sku_all` | Writes 6-week × 6-product forecasts for all depots → `tc_sku_forecasts`. Also called at the end of `--mode train_sku`. |

Other fixes:
- Reads Railway/Render injected `PORT` in serve mode (`PORT` → `API_PORT` → config).
- Evaluation step in `run_train` wrapped in try/except (non-fatal if
  `tc_model_plots` is absent).

---

## 4. Per-SKU forecasting system (new)

Forecasts demand for the 6 Tokyo Cement products per depot per week, instead of
just one depot total.

**Products & baseline mix:** SuperMix 35 %, SuperFix 25 %, SuperSeal 15 %,
SuperSet 12 %, SuperScreed 8 %, SuperFlow 5 %.

| File | Purpose |
|---|---|
| `src/db/sku_schema.sql` | Creates `tc_skus`, `tc_sku_demand_panel`, `tc_sku_forecasts`. **Run once in Supabase SQL Editor.** |
| `src/db/seed_skus.py` | Disaggregates each depot-week into 6 SKU rows using mix-ratio × product-specific seasonal curve × ±5 % noise, normalised so the 6 rows sum to the depot total. |
| `src/features/build_sku_features.py` | Joins SKU demand with weather/econ/calendar from `tc_demand_panel`; adds lags, rolling stats, `sku_enc`, and SKU×season interactions. |
| `src/model/train_sku.py` | Rolling-CV + Optuna, trains `cement_sku_forecaster_h1..h6`, promotes to Production in MLflow. Caps training to last 156 weeks and pins BLAS threads to avoid OOM. |
| `src/model/predict_sku.py` | Loads SKU models from MLflow, builds the feature row per (depot, SKU) and returns 6-week forecasts. |

**Achieved MAPE:** ~3.5 % overall (vs ~22 % for the depot-total model — the
per-product seasonal signal is much cleaner).

---

## 5. API — `src/serve/app.py`

New endpoints (depot is passed by **name**):

| Method | Path | Auth | Purpose |
|---|---|---|---|
| GET | `/health` | – | liveness + `models_loaded` |
| POST | `/reload-models` | `X-Admin-Key` | hot-reload depot models from MLflow |
| POST | `/forecast/all` | `X-Admin-Key` | regenerate all depot forecasts (background) |
| GET | `/skus` | – | list the 6 products |
| GET | `/forecast/sku/{depot}/{sku_code}` | – | one product, 6-week forecast |
| GET | `/forecast/sku-summary/{depot}` | – | all 6 products × 6 weeks |
| POST | `/forecast/sku/all` | `X-Admin-Key` | regenerate all SKU forecasts (background) |

---

## 6. Deployment artifacts (new)

| File | Purpose |
|---|---|
| `Dockerfile` | Python 3.11 slim, installs requirements, `CMD python pipeline.py --mode serve`. |
| `render.yaml` | Render web-service blueprint (env vars, health check on `/health`). |
| `.github/workflows/weekly_pipeline.yml` | Weekly cron (Sun 01:00 UTC): `update` → `train` → ping `/reload-models`. |

---

## 7. How to run locally

```bash
python pipeline.py --mode setup          # first-time: Kaggle + augment + seed depots
python pipeline.py --mode setup_skus      # seed per-SKU panel (needs sku_schema.sql first)
python pipeline.py --mode train           # train + push depot forecasts
python pipeline.py --mode train_sku       # train + push SKU forecasts
python pipeline.py --mode serve           # start the API
```

See `ARCHITECTURE.md` for the full data flow, scheduler, and manual-training guide.
