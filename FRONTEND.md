# Frontend Integration Guide

This document is the single reference for the frontend team. It covers every table the frontend reads from Supabase, the exact column names and shapes to expect, and the two calls that must go to the Python backend.

---

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│                      Frontend                            │
│              (Next.js / React / any)                     │
└──────┬────────────────────────────────────┬──────────────┘
       │                                    │
       │  DIRECT (all reads + most writes)  │  BACKEND ONLY (2 calls)
       ▼                                    ▼
┌─────────────────────┐           ┌──────────────────────┐
│      Supabase        │           │   FastAPI backend    │
│  (JS client / REST)  │           │   localhost:8000     │
│                      │           │                      │
│  tc_forecasts        │           │  POST /forecast      │
│  tc_model_plots      │           │  POST /retrain       │
│  tc_alerts           │           │                      │
│  tc_purchase_orders  │           └──────────────────────┘
│  tc_stock_levels     │                    │
│  tc_retrain_log      │                    │ writes results
│  tc_sales_actuals    │◄───────────────────┘  back to DB
│  tc_demand_panel     │
│  tc_depots           │
└─────────────────────┘
```

**Rule of thumb:** If it is a read, it comes from Supabase. If it is a write with no ML computation involved, it also goes directly to Supabase. The backend is contacted only when the frontend needs the ML model to do something — generate a forecast or retrain.

---

## Supabase Client Setup

```bash
npm install @supabase/supabase-js
```

```ts
// lib/supabase.ts
import { createClient } from '@supabase/supabase-js'

export const supabase = createClient(
  process.env.NEXT_PUBLIC_SUPABASE_URL!,
  process.env.NEXT_PUBLIC_SUPABASE_PUBLISHABLE_KEY!
)
```

Environment variables (`.env.local`):
```
NEXT_PUBLIC_SUPABASE_URL=https://<project-ref>.supabase.co
NEXT_PUBLIC_SUPABASE_PUBLISHABLE_KEY=<your-anon-key>
```

The same Supabase project that the backend writes to. RLS is disabled on all `tc_` tables so the anon key can read and write freely.

---

## Table Reference

### `tc_depots`

Static lookup table. Load once at app startup and cache — it never changes unless a new depot is added.

**Columns:**

| Column | Type | Description |
|---|---|---|
| `depot_id` | integer | Primary key |
| `name` | text | Display name — e.g. `"Colombo"` |
| `district` | text | e.g. `"Western"` |
| `province` | text | e.g. `"Western"` |
| `latitude` | number | For map pins |
| `longitude` | number | For map pins |
| `pop_weight` | number | Population weight (0–1, sums to 1 across 24 depots) |
| `created_at` | timestamptz | |

**Fetch all depots:**

```ts
const { data: depots } = await supabase
  .from('tc_depots')
  .select('depot_id, name, district, province, latitude, longitude')
  .order('name')
```

---

### `tc_forecasts`

One row per depot per forecast horizon per `as_of_date`. The backend writes here after every `POST /forecast` call. The frontend reads from here to display the demand forecast chart.

**Columns:**

| Column | Type | Description |
|---|---|---|
| `id` | integer | Primary key |
| `depot_id` | integer | FK → `tc_depots` |
| `generated_at` | timestamptz | When the forecast was computed |
| `as_of_date` | date | The date the forecast was made from |
| `horizon_weeks` | smallint | 1 through 6 |
| `forecast_week` | date | The week being forecast (`as_of_date + horizon_weeks`) |
| `demand_forecast` | number | Predicted demand in tonnes |
| `model_version` | text | MLflow model version string |

**Fetch the latest 6-week forecast for a depot:**

```ts
// Get the most recent as_of_date first
const { data: latest } = await supabase
  .from('tc_forecasts')
  .select('as_of_date')
  .eq('depot_id', depotId)
  .order('as_of_date', { ascending: false })
  .limit(1)
  .single()

// Then fetch all 6 horizons for that date
const { data: forecasts } = await supabase
  .from('tc_forecasts')
  .select('horizon_weeks, forecast_week, demand_forecast')
  .eq('depot_id', depotId)
  .eq('as_of_date', latest.as_of_date)
  .order('horizon_weeks')
```

**Shape of `forecasts`:**

```json
[
  { "horizon_weeks": 1, "forecast_week": "2022-12-05", "demand_forecast": 2071.52 },
  { "horizon_weeks": 2, "forecast_week": "2022-12-12", "demand_forecast": 1881.58 },
  { "horizon_weeks": 3, "forecast_week": "2022-12-19", "demand_forecast": 1844.22 },
  { "horizon_weeks": 4, "forecast_week": "2022-12-26", "demand_forecast": 2147.10 },
  { "horizon_weeks": 5, "forecast_week": "2023-01-02", "demand_forecast": 2311.48 },
  { "horizon_weeks": 6, "forecast_week": "2023-01-09", "demand_forecast": 2192.82 }
]
```

---

### `tc_model_plots`

One row per plot. Every training run produces ~31 rows: 7 global plots + 24 per-depot plots. Images are stored as base64-encoded PNG strings and render directly in an `<img>` tag.

**Columns:**

| Column | Type | Description |
|---|---|---|
| `id` | integer | Primary key |
| `retrain_id` | integer | FK → `tc_retrain_log` |
| `plot_type` | text | See plot type reference below |
| `depot_id` | integer \| null | Non-null only for per-depot plots |
| `image_data` | text | Base64-encoded PNG — use directly as `src` |
| `created_at` | timestamptz | |

**Plot types:**

| `plot_type` | Scope | What it shows |
|---|---|---|
| `forecast_vs_actual` | Global | Aggregated forecast vs actual demand, last CV fold |
| `mape_by_depot` | Global | Bar chart — MAPE per depot across all 6 horizons |
| `mape_by_horizon` | Global | Line chart — accuracy decay from t+1 to t+6 |
| `mape_by_season` | Global | MAPE in SW monsoon vs non-monsoon periods |
| `bias_by_depot` | Global | Signed error per depot (positive = model overestimates) |
| `feature_importance` | Global | Top 20 features by XGBoost gain |
| `shap_summary` | Global | SHAP beeswarm — which features push forecasts up/down |
| `retrain_history` | Global | MAPE trend across all historical retraining runs |
| `depot_forecast` | Per depot | 6-week forecast ribbon vs actuals for one depot (one per depot, `depot_id` set) |

**Fetch all global plots from the latest retrain:**

```ts
// Get the most recent completed retrain id
const { data: latestRetrain } = await supabase
  .from('tc_retrain_log')
  .select('id')
  .eq('status', 'completed')
  .order('triggered_at', { ascending: false })
  .limit(1)
  .single()

// Fetch all global plots (depot_id is null for global)
const { data: plots } = await supabase
  .from('tc_model_plots')
  .select('plot_type, image_data')
  .eq('retrain_id', latestRetrain.id)
  .is('depot_id', null)
```

**Render a plot:**

```tsx
<img
  src={`data:image/png;base64,${plot.image_data}`}
  alt={plot.plot_type}
/>
```

**Fetch the per-depot forecast plot for one depot:**

```ts
const { data: depotPlot } = await supabase
  .from('tc_model_plots')
  .select('image_data')
  .eq('retrain_id', latestRetrain.id)
  .eq('plot_type', 'depot_forecast')
  .eq('depot_id', depotId)
  .single()
```

---

### `tc_alerts`

Written by the backend after every forecast or stock update. The frontend reads these to show the notification/alert panel.

**Columns:**

| Column | Type | Description |
|---|---|---|
| `id` | integer | Primary key |
| `depot_id` | integer | FK → `tc_depots` |
| `created_at` | timestamptz | |
| `alert_type` | text | `critical_low_stock` / `warning_low_stock` / `demand_spike` / `overstock` |
| `severity` | text | `critical` / `warning` / `info` |
| `message` | text | Human-readable description |
| `resolved` | boolean | `false` = active |
| `resolved_at` | timestamptz \| null | |

**Fetch active alerts for a depot:**

```ts
const { data: alerts } = await supabase
  .from('tc_alerts')
  .select('id, alert_type, severity, message, created_at')
  .eq('depot_id', depotId)
  .eq('resolved', false)
  .order('created_at', { ascending: false })
```

**Resolve an alert (write directly to Supabase):**

```ts
await supabase
  .from('tc_alerts')
  .update({ resolved: true, resolved_at: new Date().toISOString() })
  .eq('id', alertId)
```

---

### `tc_purchase_orders`

Auto-generated by the backend after every forecast. Quantity is `max(0, week_1_forecast × 1.25 − current_stock)`.

**Columns:**

| Column | Type | Description |
|---|---|---|
| `id` | integer | Primary key |
| `depot_id` | integer | FK → `tc_depots` |
| `created_at` | timestamptz | |
| `week_start` | date | Week the order is for |
| `recommended_qty` | number | Suggested order quantity in tonnes |
| `current_stock` | number \| null | Stock at time of recommendation |
| `forecast_demand` | number \| null | Forecast demand used to compute qty |
| `status` | text | `pending` / `approved` / `dismissed` |
| `approved_by` | text \| null | |
| `approved_at` | timestamptz \| null | |

**Fetch pending orders for a depot:**

```ts
const { data: orders } = await supabase
  .from('tc_purchase_orders')
  .select('id, week_start, recommended_qty, current_stock, forecast_demand, status')
  .eq('depot_id', depotId)
  .eq('status', 'pending')
  .order('week_start')
```

**Approve or dismiss an order (write directly to Supabase):**

```ts
// Approve
await supabase
  .from('tc_purchase_orders')
  .update({ status: 'approved', approved_by: userName, approved_at: new Date().toISOString() })
  .eq('id', orderId)

// Dismiss
await supabase
  .from('tc_purchase_orders')
  .update({ status: 'dismissed' })
  .eq('id', orderId)
```

---

### `tc_stock_levels`

Submitted by depot managers. One row per depot per week.

**Columns:**

| Column | Type | Description |
|---|---|---|
| `id` | integer | Primary key |
| `depot_id` | integer | FK → `tc_depots` |
| `reported_at` | timestamptz | |
| `week_start` | date | ISO week start (Monday) |
| `stock_tonnes` | number | Current stock on hand |
| `reported_by` | text \| null | Manager name or ID |

**Fetch latest stock for a depot:**

```ts
const { data: stock } = await supabase
  .from('tc_stock_levels')
  .select('week_start, stock_tonnes, reported_by, reported_at')
  .eq('depot_id', depotId)
  .order('week_start', { ascending: false })
  .limit(1)
  .single()
```

**Submit a stock reading (write directly to Supabase):**

```ts
await supabase
  .from('tc_stock_levels')
  .upsert(
    { depot_id: depotId, week_start: weekStart, stock_tonnes: qty, reported_by: userName },
    { onConflict: 'depot_id,week_start' }
  )
```

> Note: Writing stock directly to Supabase does **not** auto-evaluate alert conditions. If you want alerts to fire immediately after a stock update, call `POST /stock` on the backend instead (see the backend reference at the end of this document). For non-real-time use, a nightly backend job can sweep and re-evaluate alerts.

---

### `tc_sales_actuals`

Real sales figures entered by depot managers. These rows replace the synthetic `augmented` rows in `tc_demand_panel` over time.

**Columns:**

| Column | Type | Description |
|---|---|---|
| `id` | integer | Primary key |
| `depot_id` | integer | FK → `tc_depots` |
| `week_start` | date | ISO week start |
| `sales_tonnes` | number | Actual sales for that week |
| `demand_tonnes` | number \| null | Actual demand (can differ from sales if stock-outs occurred) |
| `notes` | text \| null | Optional manager note |
| `entered_by` | text \| null | |
| `entered_at` | timestamptz | |
| `updated_by` | text \| null | |
| `updated_at` | timestamptz \| null | |

**Fetch recent sales history for a depot:**

```ts
const { data: sales } = await supabase
  .from('tc_sales_actuals')
  .select('week_start, sales_tonnes, demand_tonnes, entered_by, entered_at')
  .eq('depot_id', depotId)
  .order('week_start', { ascending: false })
  .limit(12)
```

**Submit a sales entry (write directly to Supabase):**

```ts
await supabase
  .from('tc_sales_actuals')
  .upsert(
    {
      depot_id: depotId,
      week_start: weekStart,
      sales_tonnes: salesQty,
      demand_tonnes: demandQty ?? null,
      entered_by: userName,
    },
    { onConflict: 'depot_id,week_start' }
  )
```

> Same note as stock: writing here directly does **not** trigger auto-retrain or sync to `tc_demand_panel`. If you want those side effects, route submissions through the backend (`POST /sales`). For a simpler integration, a nightly backend cron can sync `tc_sales_actuals` → `tc_demand_panel` and count rows for retrain triggering.

---

### `tc_retrain_log`

One row per training run. Use this to show the model version history and current model health.

**Columns:**

| Column | Type | Description |
|---|---|---|
| `id` | integer | Primary key |
| `triggered_at` | timestamptz | When training started |
| `triggered_by` | text \| null | `"manual"` or `"auto"` |
| `trigger_reason` | text \| null | Human-readable reason |
| `rows_added` | integer \| null | New sales rows that triggered auto-retrain |
| `training_data_up_to` | date \| null | Latest week in the training set |
| `mape_before` | number \| null | MAPE of the previous Production model |
| `mape_after` | number \| null | MAPE of the newly trained model |
| `new_model_version` | text \| null | MLflow version string |
| `status` | text | `pending` / `running` / `completed` / `failed` |
| `error_message` | text \| null | Set if `status = 'failed'` |
| `mlflow_version` | integer \| null | MLflow registry version number |
| `promoted` | boolean | Whether the new model replaced the Production model |

**Fetch the last 10 training runs:**

```ts
const { data: history } = await supabase
  .from('tc_retrain_log')
  .select('id, triggered_at, triggered_by, mape_before, mape_after, status, promoted')
  .order('triggered_at', { ascending: false })
  .limit(10)
```

**Fetch the current production model info:**

```ts
const { data: current } = await supabase
  .from('tc_retrain_log')
  .select('triggered_at, mape_after, mlflow_version, promoted')
  .eq('status', 'completed')
  .eq('promoted', true)
  .order('triggered_at', { ascending: false })
  .limit(1)
  .single()
```

---

### `tc_demand_panel`

The raw training panel — 16,200 rows, one per depot per week from 2009 to 2022. The frontend would typically not display this directly, but it is useful for rendering historical demand charts.

**Columns relevant to the frontend:**

| Column | Type | Description |
|---|---|---|
| `depot_id` | integer | FK → `tc_depots` |
| `week_start` | date | ISO week start |
| `demand_tonnes` | number | Actual (or augmented) demand |
| `sales_tonnes` | number \| null | |
| `data_source` | text | `'augmented'` or `'actual'` |

**Fetch demand history for a chart (last 52 weeks for one depot):**

```ts
const { data: history } = await supabase
  .from('tc_demand_panel')
  .select('week_start, demand_tonnes, data_source')
  .eq('depot_id', depotId)
  .order('week_start', { ascending: false })
  .limit(52)
```

> `tc_demand_panel` has 16,200 rows — always filter by `depot_id` and use `.limit()`. Never select the full table from the frontend.

---

## The Two Backend Calls

These are the only two endpoints the frontend needs to call on the Python backend (`http://localhost:8000` in development; swap for the deployed URL in production).

---

### 1. `POST /forecast` — Generate a Fresh Forecast

**When to call:** When the user clicks a "Run Forecast" or "Refresh" button for a depot. Do not call this on every page load — the stored results in `tc_forecasts` are sufficient for display. Call it when the user explicitly wants a new prediction from the current model.

**Request:**

```ts
const response = await fetch(`${BACKEND_URL}/forecast`, {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    depot: 'Colombo',           // depot name exactly as in tc_depots.name
    as_of_date: '2022-11-28',   // YYYY-MM-DD — the week to forecast from
  }),
})
const data = await response.json()
```

**Response:**

```json
{
  "depot": "Colombo",
  "as_of_date": "2022-11-28",
  "forecasts": [
    { "horizon": 1, "forecast_week": "2022-12-05", "demand_tonnes": 2071.52 },
    { "horizon": 2, "forecast_week": "2022-12-12", "demand_tonnes": 1881.58 },
    { "horizon": 3, "forecast_week": "2022-12-19", "demand_tonnes": 1844.22 },
    { "horizon": 4, "forecast_week": "2022-12-26", "demand_tonnes": 2147.10 },
    { "horizon": 5, "forecast_week": "2023-01-02", "demand_tonnes": 2311.48 },
    { "horizon": 6, "forecast_week": "2023-01-09", "demand_tonnes": 2192.82 }
  ],
  "generated_at": "2026-05-22T08:15:54.056717+00:00"
}
```

**What the backend does internally:**
1. Pulls the most recent 52 weeks of `tc_demand_panel` for this depot from Supabase
2. Constructs the feature vector (lag features, weather, calendar)
3. Runs all 6 XGBoost horizon models
4. Writes the 6 forecast rows to `tc_forecasts` in Supabase
5. Evaluates alert conditions and writes to `tc_alerts` if triggered
6. Creates a purchase order recommendation in `tc_purchase_orders`
7. Returns the 6 forecasts in the response

After this call returns, `tc_forecasts`, `tc_alerts`, and `tc_purchase_orders` in Supabase are already updated — the frontend can re-query them directly.

---

### 2. `POST /retrain` — Trigger Model Retraining

**When to call:** When the user clicks a "Retrain Model" button in an admin/settings page. This starts a background job on the backend — it returns immediately with a `retrain_id` and the training runs asynchronously (typically 10–15 minutes).

**Request:**

```ts
const response = await fetch(`${BACKEND_URL}/retrain`, {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    reason: 'Manual retrain requested by admin',  // optional
  }),
})
const data = await response.json()
// { "retrain_id": 4, "status": "started", "message": "Retraining started in background" }
```

**Poll for completion:**

Poll `tc_retrain_log` directly in Supabase — no need to call the backend for status:

```ts
const { data: log } = await supabase
  .from('tc_retrain_log')
  .select('status, mape_before, mape_after, promoted, error_message')
  .eq('id', retrainId)
  .single()

// log.status: 'pending' | 'running' | 'completed' | 'failed'
```

Poll every 30 seconds until `status` is `completed` or `failed`. When `completed`:
- New plots are in `tc_model_plots` — re-fetch them
- `mape_after` shows the new model's accuracy
- `promoted: true` means the new model replaced the Production model

**What the backend does internally:**
1. Creates a row in `tc_retrain_log` with `status = 'running'`
2. Pulls all 16,200+ rows from `tc_demand_panel`
3. Runs rolling-window cross-validation (95 folds) + Optuna tuning (50 trials)
4. Trains 6 XGBoost models (one per horizon)
5. Evaluates MAPE — promotes to Production only if better than current model
6. Saves ~31 plots to `tc_model_plots`
7. Updates the `tc_retrain_log` row to `status = 'completed'`

---

## Recommended Page → Data Mapping

| Page / Component | Data source | Notes |
|---|---|---|
| Depot list / map | `tc_depots` | Load once, cache |
| Demand forecast chart | `tc_forecasts` | Filter by `depot_id`, latest `as_of_date` |
| Historical demand chart | `tc_demand_panel` | Filter by `depot_id`, limit 52 |
| Alert panel | `tc_alerts` | Filter `resolved = false` |
| Purchase order list | `tc_purchase_orders` | Filter by `depot_id`, `status = 'pending'` |
| Stock display | `tc_stock_levels` | Latest row per `depot_id` |
| Sales history table | `tc_sales_actuals` | Filter by `depot_id`, last 12 weeks |
| Model performance plots | `tc_model_plots` | Filter by latest `retrain_id`, `depot_id IS NULL` for global plots |
| Per-depot plot | `tc_model_plots` | Filter by `plot_type = 'depot_forecast'`, `depot_id` |
| Model version history | `tc_retrain_log` | Last 10 rows, descending |
| "Run Forecast" button | **→ Backend** `POST /forecast` | Call backend, then re-read `tc_forecasts` |
| "Retrain Model" button | **→ Backend** `POST /retrain` | Call backend, poll `tc_retrain_log` via Supabase |
| "Resolve alert" button | `tc_alerts` UPDATE | Direct Supabase write |
| "Approve order" button | `tc_purchase_orders` UPDATE | Direct Supabase write |
| Stock submission form | `tc_stock_levels` UPSERT | Direct Supabase write |
| Sales submission form | `tc_sales_actuals` UPSERT | Direct Supabase write |

---

## Environment Variables Summary

```
# .env.local (frontend)
NEXT_PUBLIC_SUPABASE_URL=https://<project-ref>.supabase.co
NEXT_PUBLIC_SUPABASE_PUBLISHABLE_KEY=<anon-key>

NEXT_PUBLIC_BACKEND_URL=http://localhost:8000   # or deployed backend URL
```

The `BACKEND_URL` is only used in two places: the forecast button and the retrain button. Every other data access uses the Supabase client directly.
