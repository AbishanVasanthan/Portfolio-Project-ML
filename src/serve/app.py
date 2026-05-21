import asyncio
import logging
import os
from datetime import date, datetime, timezone
from typing import Optional

import pandas as pd
import psycopg2.extras
import yaml
from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.db.db import get_conn, release_conn

logger = logging.getLogger(__name__)

app = FastAPI(title="Tokyo Cement Demand Forecasting API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Startup: load config and models ──────────────────────────

_cfg: dict = {}
_models_loaded = False


@app.on_event("startup")
async def startup():
    global _cfg, _models_loaded
    with open("config.yaml") as f:
        _cfg = yaml.safe_load(f)

    from src.model.predict import load_models
    try:
        load_models(_cfg)
        _models_loaded = True
        logger.info("[SERVE] Models loaded successfully")
    except Exception as e:
        logger.warning("[SERVE] Could not load models at startup (train first): %s", e)


# ── Helper: resolve depot name → depot_id ────────────────────

def _resolve_depot(name: str) -> tuple[int, str]:
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT depot_id, name FROM depots WHERE name = %s", (name,))
            row = cur.fetchone()
            if not row:
                raise HTTPException(status_code=404, detail=f"Depot '{name}' not found")
            return row[0], row[1]
    finally:
        release_conn(conn)


def _get_recent_panel(depot_id: int, n_weeks: int = 52) -> pd.DataFrame:
    conn = get_conn()
    try:
        df = pd.read_sql(
            """
            SELECT dp.*, de.name AS depot
            FROM demand_panel dp
            JOIN depots de ON dp.depot_id = de.depot_id
            WHERE dp.depot_id = %s
            ORDER BY dp.week_start DESC
            LIMIT %s
            """,
            conn,
            params=(depot_id, n_weeks),
        )
        return df.sort_values("week_start")
    finally:
        release_conn(conn)


# ── GET /depots ───────────────────────────────────────────────

@app.get("/depots")
def get_depots():
    conn = get_conn()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("SELECT depot_id, name, district, province, latitude, longitude FROM depots ORDER BY name")
            return cur.fetchall()
    finally:
        release_conn(conn)


# ── POST /forecast ────────────────────────────────────────────

class ForecastRequest(BaseModel):
    depot: str
    as_of_date: date


@app.post("/forecast")
def create_forecast(req: ForecastRequest, background_tasks: BackgroundTasks):
    if not _models_loaded:
        raise HTTPException(status_code=503, detail="Models not loaded. Run `python pipeline.py --mode train` first.")

    depot_id, depot_name = _resolve_depot(req.depot)
    recent = _get_recent_panel(depot_id, 52)
    if recent.empty:
        raise HTTPException(status_code=422, detail=f"No panel data for depot '{depot_name}'")

    from src.model.predict import forecast_depot
    forecasts = forecast_depot(depot_name, req.as_of_date, recent, _cfg)

    # Write to forecasts table
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            for fc in forecasts:
                cur.execute(
                    """
                    INSERT INTO forecasts (depot_id, as_of_date, horizon_weeks, forecast_week, demand_forecast)
                    VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT (depot_id, as_of_date, horizon_weeks) DO UPDATE
                        SET demand_forecast = EXCLUDED.demand_forecast, generated_at = NOW()
                    """,
                    (depot_id, req.as_of_date, fc["horizon"], fc["forecast_week"], fc["demand_tonnes"]),
                )
        conn.commit()
    finally:
        release_conn(conn)

    background_tasks.add_task(_run_alert_evaluation, depot_id, forecasts)
    background_tasks.add_task(_run_po_generation, depot_id, forecasts, req.as_of_date)

    return {
        "depot": depot_name,
        "as_of_date": req.as_of_date,
        "forecasts": forecasts,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


# ── GET /forecasts/{depot} ────────────────────────────────────

@app.get("/forecasts/{depot}")
def get_forecasts(depot: str, as_of_date: Optional[date] = None):
    depot_id, depot_name = _resolve_depot(depot)
    conn = get_conn()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            if as_of_date:
                cur.execute(
                    """SELECT horizon_weeks AS horizon, forecast_week, demand_forecast AS demand_tonnes
                       FROM forecasts WHERE depot_id=%s AND as_of_date=%s ORDER BY horizon_weeks""",
                    (depot_id, as_of_date),
                )
            else:
                cur.execute(
                    """SELECT horizon_weeks AS horizon, forecast_week, demand_forecast AS demand_tonnes
                       FROM forecasts WHERE depot_id=%s
                       ORDER BY generated_at DESC, horizon_weeks LIMIT 6""",
                    (depot_id,),
                )
            rows = cur.fetchall()
        return {"depot": depot_name, "forecasts": rows}
    finally:
        release_conn(conn)


# ── POST /stock ───────────────────────────────────────────────

class StockRequest(BaseModel):
    depot: str
    week_start: date
    stock_tonnes: float
    reported_by: Optional[str] = None


@app.post("/stock")
def submit_stock(req: StockRequest, background_tasks: BackgroundTasks):
    depot_id, _ = _resolve_depot(req.depot)
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """INSERT INTO stock_levels (depot_id, week_start, stock_tonnes, reported_by)
                   VALUES (%s, %s, %s, %s)
                   ON CONFLICT (depot_id, week_start) DO UPDATE
                       SET stock_tonnes = EXCLUDED.stock_tonnes, reported_at = NOW()
                   RETURNING id""",
                (depot_id, req.week_start, req.stock_tonnes, req.reported_by),
            )
            stock_id = cur.fetchone()[0]
        conn.commit()
    finally:
        release_conn(conn)

    # Re-evaluate alerts after stock update
    background_tasks.add_task(_run_alert_evaluation, depot_id, None)
    return {"status": "saved", "stock_id": stock_id}


# ── GET /stock/{depot} ────────────────────────────────────────

@app.get("/stock/{depot}")
def get_stock(depot: str):
    depot_id, depot_name = _resolve_depot(depot)
    conn = get_conn()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """SELECT week_start, stock_tonnes, reported_at
                   FROM stock_levels WHERE depot_id=%s ORDER BY week_start DESC LIMIT 12""",
                (depot_id,),
            )
            rows = cur.fetchall()
        latest = rows[0] if rows else None
        return {"depot": depot_name, "latest": latest, "history": rows}
    finally:
        release_conn(conn)


# ── GET /purchase-orders/{depot} ─────────────────────────────

@app.get("/purchase-orders/{depot}")
def get_purchase_orders(depot: str, status: str = "pending"):
    depot_id, depot_name = _resolve_depot(depot)
    conn = get_conn()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            if status == "all":
                cur.execute(
                    "SELECT id AS po_id, week_start, recommended_qty, current_stock, forecast_demand, status, created_at "
                    "FROM purchase_orders WHERE depot_id=%s ORDER BY created_at DESC",
                    (depot_id,),
                )
            else:
                cur.execute(
                    "SELECT id AS po_id, week_start, recommended_qty, current_stock, forecast_demand, status, created_at "
                    "FROM purchase_orders WHERE depot_id=%s AND status=%s ORDER BY created_at DESC",
                    (depot_id, status),
                )
            rows = cur.fetchall()
        for r in rows:
            r["depot"] = depot_name
        return rows
    finally:
        release_conn(conn)


# ── PATCH /purchase-orders/{po_id} ───────────────────────────

class POPatch(BaseModel):
    status: str
    approved_by: Optional[str] = None


@app.patch("/purchase-orders/{po_id}")
def patch_purchase_order(po_id: int, req: POPatch):
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """UPDATE purchase_orders
                   SET status=%s, approved_by=%s, approved_at=NOW()
                   WHERE id=%s RETURNING id""",
                (req.status, req.approved_by, po_id),
            )
            if not cur.fetchone():
                raise HTTPException(status_code=404, detail="Purchase order not found")
        conn.commit()
    finally:
        release_conn(conn)
    return {"po_id": po_id, "status": req.status, "approved_at": datetime.now(timezone.utc).isoformat()}


# ── GET /alerts/{depot} ───────────────────────────────────────

@app.get("/alerts/{depot}")
def get_alerts(depot: str, resolved: bool = False):
    depot_id, depot_name = _resolve_depot(depot)
    conn = get_conn()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """SELECT id AS alert_id, alert_type, severity, message, created_at
                   FROM alerts WHERE depot_id=%s AND resolved=%s ORDER BY created_at DESC""",
                (depot_id, resolved),
            )
            rows = cur.fetchall()
        for r in rows:
            r["depot"] = depot_name
        return rows
    finally:
        release_conn(conn)


# ── PATCH /alerts/{alert_id}/resolve ─────────────────────────

@app.patch("/alerts/{alert_id}/resolve")
def resolve_alert(alert_id: int):
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE alerts SET resolved=TRUE, resolved_at=NOW() WHERE id=%s RETURNING id",
                (alert_id,),
            )
            if not cur.fetchone():
                raise HTTPException(status_code=404, detail="Alert not found")
        conn.commit()
    finally:
        release_conn(conn)
    return {"alert_id": alert_id, "resolved": True, "resolved_at": datetime.now(timezone.utc).isoformat()}


# ── GET /dashboard/{depot} ────────────────────────────────────

@app.get("/dashboard/{depot}")
def get_dashboard(depot: str):
    depot_id, depot_name = _resolve_depot(depot)

    conn = get_conn()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            # Depot metadata
            cur.execute("SELECT * FROM depots WHERE depot_id=%s", (depot_id,))
            depot_meta = cur.fetchone()

            # Latest stock
            cur.execute(
                "SELECT week_start, stock_tonnes, reported_at FROM stock_levels "
                "WHERE depot_id=%s ORDER BY week_start DESC LIMIT 1", (depot_id,)
            )
            latest_stock = cur.fetchone()

            # Latest forecast
            cur.execute(
                "SELECT horizon_weeks AS horizon, forecast_week, demand_forecast AS demand_tonnes "
                "FROM forecasts WHERE depot_id=%s ORDER BY generated_at DESC, horizon_weeks LIMIT 6",
                (depot_id,),
            )
            forecast = cur.fetchall()

            # Pending POs
            cur.execute(
                "SELECT id AS po_id, week_start, recommended_qty, current_stock, forecast_demand, status "
                "FROM purchase_orders WHERE depot_id=%s AND status='pending'", (depot_id,)
            )
            pending_pos = cur.fetchall()

            # Active alerts
            cur.execute(
                "SELECT id AS alert_id, alert_type, severity, message, created_at "
                "FROM alerts WHERE depot_id=%s AND resolved=FALSE", (depot_id,)
            )
            active_alerts = cur.fetchall()

    finally:
        release_conn(conn)

    return {
        "depot": depot_meta,
        "latest_stock": latest_stock,
        "forecast": forecast,
        "pending_pos": pending_pos,
        "active_alerts": active_alerts,
    }


# ── POST /sales ───────────────────────────────────────────────

class SalesSubmit(BaseModel):
    depot: str
    week_start: date
    sales_tonnes: float
    demand_tonnes: Optional[float] = None
    entered_by: Optional[str] = None
    notes: Optional[str] = None


@app.post("/sales")
def submit_sales(req: SalesSubmit, background_tasks: BackgroundTasks):
    depot_id, _ = _resolve_depot(req.depot)
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """INSERT INTO sales_actuals
                       (depot_id, week_start, sales_tonnes, demand_tonnes, notes, entered_by)
                   VALUES (%s, %s, %s, %s, %s, %s)
                   ON CONFLICT (depot_id, week_start) DO UPDATE
                       SET sales_tonnes=EXCLUDED.sales_tonnes,
                           demand_tonnes=EXCLUDED.demand_tonnes,
                           notes=EXCLUDED.notes,
                           updated_by=EXCLUDED.entered_by,
                           updated_at=NOW()
                   RETURNING id""",
                (depot_id, req.week_start, req.sales_tonnes,
                 req.demand_tonnes, req.notes, req.entered_by),
            )
            sales_id = cur.fetchone()[0]

            # Sync to demand_panel
            cur.execute(
                """UPDATE demand_panel
                   SET sales_tonnes=%s, demand_tonnes=%s, data_source='actual'
                   WHERE depot_id=%s AND week_start=%s""",
                (req.sales_tonnes, req.demand_tonnes, depot_id, req.week_start),
            )
        conn.commit()
    finally:
        release_conn(conn)

    background_tasks.add_task(_maybe_trigger_retrain, "auto", f"New sales for {req.depot} week {req.week_start}")
    return {"status": "saved", "sales_id": sales_id, "retrain_scheduled": True}


# ── PUT /sales/{depot}/{week_start} ──────────────────────────

class SalesUpdate(BaseModel):
    sales_tonnes: float
    demand_tonnes: Optional[float] = None
    updated_by: Optional[str] = None
    notes: Optional[str] = None


@app.put("/sales/{depot}/{week_start}")
def update_sales(depot: str, week_start: date, req: SalesUpdate, background_tasks: BackgroundTasks):
    depot_id, _ = _resolve_depot(depot)
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """UPDATE sales_actuals
                   SET sales_tonnes=%s, demand_tonnes=%s, notes=%s,
                       updated_by=%s, updated_at=NOW()
                   WHERE depot_id=%s AND week_start=%s
                   RETURNING id""",
                (req.sales_tonnes, req.demand_tonnes, req.notes,
                 req.updated_by, depot_id, week_start),
            )
            row = cur.fetchone()
            if not row:
                raise HTTPException(status_code=404, detail="Sales record not found")
            sales_id = row[0]

            cur.execute(
                """UPDATE demand_panel
                   SET sales_tonnes=%s, demand_tonnes=%s, data_source='actual'
                   WHERE depot_id=%s AND week_start=%s""",
                (req.sales_tonnes, req.demand_tonnes, depot_id, week_start),
            )
        conn.commit()
    finally:
        release_conn(conn)

    background_tasks.add_task(_maybe_trigger_retrain, "auto", f"Updated sales for {depot} week {week_start}")
    return {"status": "updated", "sales_id": sales_id, "retrain_scheduled": True}


# ── GET /sales/{depot} ────────────────────────────────────────

@app.get("/sales/{depot}")
def get_sales(depot: str, weeks: int = 12):
    depot_id, _ = _resolve_depot(depot)
    weeks = min(weeks, 52)
    conn = get_conn()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                "SELECT week_start, sales_tonnes, demand_tonnes, notes FROM sales_actuals "
                "WHERE depot_id=%s ORDER BY week_start DESC LIMIT %s",
                (depot_id, weeks),
            )
            return cur.fetchall()
    finally:
        release_conn(conn)


# ── POST /retrain ─────────────────────────────────────────────

class RetrainRequest(BaseModel):
    triggered_by: Optional[str] = "admin"


@app.post("/retrain")
async def trigger_retrain(req: RetrainRequest, background_tasks: BackgroundTasks):
    retrain_id = _create_retrain_log_row(req.triggered_by, "Manual trigger via API")
    background_tasks.add_task(_run_retrain, retrain_id)
    return {
        "status": "started",
        "retrain_id": retrain_id,
        "message": f"Retraining in progress. Check /retrain/status/{retrain_id} for updates.",
    }


# ── GET /retrain/status/{retrain_id} ─────────────────────────

@app.get("/retrain/status/{retrain_id}")
def get_retrain_status(retrain_id: int):
    conn = get_conn()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("SELECT * FROM retrain_log WHERE id=%s", (retrain_id,))
            row = cur.fetchone()
            if not row:
                raise HTTPException(status_code=404, detail="Retrain log not found")
            return row
    finally:
        release_conn(conn)


# ── GET /retrain/history ──────────────────────────────────────

@app.get("/retrain/history")
def get_retrain_history():
    conn = get_conn()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                "SELECT id AS retrain_id, triggered_at, mape_before, mape_after, promoted, status "
                "FROM retrain_log ORDER BY triggered_at DESC LIMIT 10"
            )
            return cur.fetchall()
    finally:
        release_conn(conn)


# ── GET /plots/latest ─────────────────────────────────────────

@app.get("/plots/latest")
def get_latest_plots():
    conn = get_conn()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """SELECT mp.plot_type, mp.image_data
                   FROM model_plots mp
                   JOIN retrain_log rl ON mp.retrain_id = rl.id
                   WHERE rl.status = 'completed' AND mp.depot_id IS NULL
                   AND rl.id = (SELECT MAX(id) FROM retrain_log WHERE status='completed')
                   ORDER BY mp.plot_type"""
            )
            return cur.fetchall()
    finally:
        release_conn(conn)


# ── GET /plots/depot/{depot} ──────────────────────────────────

@app.get("/plots/depot/{depot}")
def get_depot_plot(depot: str):
    depot_id, depot_name = _resolve_depot(depot)
    conn = get_conn()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """SELECT mp.plot_type, mp.image_data, mp.retrain_id, mp.created_at
                   FROM model_plots mp
                   JOIN retrain_log rl ON mp.retrain_id = rl.id
                   WHERE mp.plot_type = 'depot_forecast' AND mp.depot_id = %s
                   AND rl.status = 'completed'
                   ORDER BY mp.retrain_id DESC LIMIT 1""",
                (depot_id,),
            )
            row = cur.fetchone()
            if not row:
                raise HTTPException(status_code=404, detail=f"No depot forecast plot found for '{depot_name}'")
            row["depot"] = depot_name
            return row
    finally:
        release_conn(conn)


# ── GET /plots/{retrain_id} ───────────────────────────────────

@app.get("/plots/{retrain_id}")
def get_plots_for_run(retrain_id: int, plot_type: Optional[str] = None):
    conn = get_conn()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            if plot_type:
                cur.execute(
                    "SELECT plot_type, depot_id, image_data, created_at FROM model_plots "
                    "WHERE retrain_id=%s AND plot_type=%s ORDER BY plot_type",
                    (retrain_id, plot_type),
                )
            else:
                cur.execute(
                    "SELECT plot_type, depot_id, image_data, created_at FROM model_plots "
                    "WHERE retrain_id=%s ORDER BY plot_type",
                    (retrain_id,),
                )
            return cur.fetchall()
    finally:
        release_conn(conn)


# ── Internal: alert evaluation ────────────────────────────────

def _run_alert_evaluation(depot_id: int, forecasts: Optional[list]) -> None:
    """Evaluate alert conditions and write new alerts (no duplicates)."""
    conn = get_conn()
    try:
        # Get current stock
        with conn.cursor() as cur:
            cur.execute(
                "SELECT stock_tonnes FROM stock_levels WHERE depot_id=%s ORDER BY week_start DESC LIMIT 1",
                (depot_id,),
            )
            row = cur.fetchone()
            current_stock = float(row[0]) if row else None

        if forecasts is None:
            # Fetch latest stored forecasts
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(
                    "SELECT demand_forecast FROM forecasts WHERE depot_id=%s "
                    "ORDER BY generated_at DESC, horizon_weeks LIMIT 6",
                    (depot_id,),
                )
                rows = cur.fetchall()
                forecasts = [{"demand_tonnes": r["demand_forecast"], "horizon": i + 1}
                             for i, r in enumerate(rows)]

        if not forecasts:
            return

        demand_2w = sum(f["demand_tonnes"] for f in forecasts[:2])
        demand_4w = sum(f["demand_tonnes"] for f in forecasts[:4])
        demand_6w = sum(f["demand_tonnes"] for f in forecasts[:6])
        demand_w1 = forecasts[0]["demand_tonnes"] if forecasts else 0

        # Rolling mean from demand_panel
        with conn.cursor() as cur:
            cur.execute(
                "SELECT demand_tonnes FROM demand_panel WHERE depot_id=%s "
                "ORDER BY week_start DESC LIMIT 4",
                (depot_id,),
            )
            recent = [r[0] for r in cur.fetchall() if r[0] is not None]
        rolling_mean = float(sum(recent) / len(recent)) if recent else demand_w1

        alerts_to_create = []
        alert_cfg = _cfg.get("alerts", {})

        if current_stock is not None:
            # Low stock critical
            threshold_crit = demand_2w * alert_cfg.get("low_stock_critical_buffer", 0.80)
            if current_stock < threshold_crit:
                alerts_to_create.append((
                    "low_stock", "critical",
                    f"Projected stockout in 2 weeks. Current stock {current_stock:.0f}t vs 2-week forecast {demand_2w:.0f}t."
                ))
            # Low stock warning
            elif current_stock < demand_4w * alert_cfg.get("low_stock_warning_buffer", 0.90):
                alerts_to_create.append((
                    "low_stock", "warning",
                    f"Stock may run low within 4 weeks. Current stock {current_stock:.0f}t vs 4-week forecast {demand_4w:.0f}t."
                ))

            # Overstock
            overstock_thresh = demand_6w * alert_cfg.get("overstock_multiplier", 1.50)
            if current_stock > overstock_thresh:
                alerts_to_create.append((
                    "overstock", "warning",
                    f"Excess stock detected. {current_stock:.0f}t held vs {demand_6w:.0f}t forecast over 6 weeks."
                ))

        # Demand spike
        spike_mult = alert_cfg.get("demand_spike_multiplier", 1.30)
        if demand_w1 > rolling_mean * spike_mult:
            pct = (demand_w1 / rolling_mean - 1) * 100 if rolling_mean else 0
            alerts_to_create.append((
                "demand_spike", "warning",
                f"Demand spike forecast: {demand_w1:.0f}t vs 4-week avg {rolling_mean:.0f}t (+{pct:.0f}%)."
            ))

        with conn.cursor() as cur:
            for alert_type, severity, message in alerts_to_create:
                # Check no unresolved duplicate
                cur.execute(
                    "SELECT id FROM alerts WHERE depot_id=%s AND alert_type=%s AND resolved=FALSE",
                    (depot_id, alert_type),
                )
                if cur.fetchone():
                    continue
                cur.execute(
                    "INSERT INTO alerts (depot_id, alert_type, severity, message) VALUES (%s,%s,%s,%s)",
                    (depot_id, alert_type, severity, message),
                )
        conn.commit()
    except Exception as e:
        logger.warning("[SERVE] Alert evaluation failed for depot_id=%d: %s", depot_id, e)
        conn.rollback()
    finally:
        release_conn(conn)


# ── Internal: PO generation ───────────────────────────────────

def _run_po_generation(depot_id: int, forecasts: list, as_of_date: date) -> None:
    conn = get_conn()
    try:
        # Get current stock
        with conn.cursor() as cur:
            cur.execute(
                "SELECT stock_tonnes FROM stock_levels WHERE depot_id=%s ORDER BY week_start DESC LIMIT 1",
                (depot_id,),
            )
            row = cur.fetchone()
            current_stock = float(row[0]) if row else 0.0

        if not forecasts:
            return

        forecast_w1 = forecasts[0]["demand_tonnes"]
        safety_pct = _cfg.get("purchase_orders", {}).get("safety_stock_pct", 0.25)
        safety_stock = forecast_w1 * safety_pct
        recommended_qty = max(0.0, forecast_w1 + safety_stock - current_stock)

        if recommended_qty <= 0:
            return

        from datetime import timedelta
        week_start = as_of_date + timedelta(weeks=1)

        with conn.cursor() as cur:
            cur.execute(
                """INSERT INTO purchase_orders
                       (depot_id, week_start, recommended_qty, current_stock, forecast_demand)
                   VALUES (%s, %s, %s, %s, %s)
                   ON CONFLICT DO NOTHING""",
                (depot_id, week_start, recommended_qty, current_stock, forecast_w1),
            )
        conn.commit()
    except Exception as e:
        logger.warning("[SERVE] PO generation failed for depot_id=%d: %s", depot_id, e)
        conn.rollback()
    finally:
        release_conn(conn)


# ── Internal: retrain helpers ─────────────────────────────────

def _pending_sales_count() -> int:
    """Count sales_actuals rows added since last completed retrain."""
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """SELECT entered_at FROM sales_actuals
                   WHERE entered_at > COALESCE(
                       (SELECT triggered_at FROM retrain_log WHERE status='completed' ORDER BY id DESC LIMIT 1),
                       '2000-01-01'
                   )"""
            )
            return cur.rowcount if cur.rowcount >= 0 else len(cur.fetchall())
    finally:
        release_conn(conn)


def _create_retrain_log_row(triggered_by: str, reason: str) -> int:
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """INSERT INTO retrain_log (triggered_by, trigger_reason, status)
                   VALUES (%s, %s, 'pending') RETURNING id""",
                (triggered_by, reason),
            )
            retrain_id = cur.fetchone()[0]
        conn.commit()
        return retrain_id
    finally:
        release_conn(conn)


def _maybe_trigger_retrain(triggered_by: str, reason: str) -> None:
    batch_size = _cfg.get("model", {}).get("retrain_batch_size", 5)
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """SELECT COUNT(*) FROM sales_actuals
                   WHERE entered_at > COALESCE(
                       (SELECT triggered_at FROM retrain_log WHERE status='completed' ORDER BY id DESC LIMIT 1),
                       '2000-01-01'
                   )"""
            )
            pending = cur.fetchone()[0]
    finally:
        release_conn(conn)

    if pending >= batch_size:
        retrain_id = _create_retrain_log_row(triggered_by, reason)
        _run_retrain(retrain_id)
    else:
        # Log as pending
        _create_retrain_log_row(triggered_by, f"{reason} (pending, {pending}/{batch_size} rows)")
        logger.info("[SERVE] Retrain pending: %d/%d new rows", pending, batch_size)


def _run_retrain(retrain_id: int) -> None:
    """Execute a full retrain cycle synchronously (called from background task)."""
    from src.model.train import train_all_horizons
    from src.model.evaluate import run_evaluation
    from src.features.build_features import rebuild_lag_features_for_depots

    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("UPDATE retrain_log SET status='running' WHERE id=%s", (retrain_id,))
        conn.commit()
    finally:
        release_conn(conn)

    try:
        # Get MAPE before
        conn2 = get_conn()
        try:
            with conn2.cursor() as cur:
                cur.execute(
                    "SELECT mape_after FROM retrain_log WHERE status='completed' ORDER BY id DESC LIMIT 1"
                )
                row = cur.fetchone()
                mape_before = float(row[0]) if row else None
        finally:
            release_conn(conn2)

        result = train_all_horizons(_cfg, retrain_id=retrain_id)

        # Evaluate and save plots
        df_full = rebuild_lag_features_for_depots([], _cfg)
        eval_result = run_evaluation(result, df_full, retrain_id, _cfg)

        # Find latest week in training data
        conn3 = get_conn()
        try:
            with conn3.cursor() as cur:
                cur.execute("SELECT MAX(week_start) FROM demand_panel")
                latest_week = cur.fetchone()[0]
        finally:
            release_conn(conn3)

        conn4 = get_conn()
        try:
            with conn4.cursor() as cur:
                cur.execute(
                    """UPDATE retrain_log
                       SET status='completed', mape_before=%s, mape_after=%s,
                           promoted=%s, training_data_up_to=%s
                       WHERE id=%s""",
                    (mape_before, result["overall_mape"], result["promoted"], latest_week, retrain_id),
                )
            conn4.commit()
        finally:
            release_conn(conn4)

        # Reload models if promoted
        if result["promoted"]:
            global _models_loaded
            from src.model.predict import load_models
            try:
                load_models(_cfg)
                _models_loaded = True
                logger.info("[SERVE] Models reloaded after promotion")
            except Exception as e:
                logger.warning("[SERVE] Model reload failed: %s", e)

        logger.info("[SERVE] Retrain %d complete: MAPE %.2f%% (promoted=%s)",
                    retrain_id, result["overall_mape"], result["promoted"])

    except Exception as e:
        logger.error("[SERVE] Retrain %d failed: %s", retrain_id, e)
        conn5 = get_conn()
        try:
            with conn5.cursor() as cur:
                cur.execute(
                    "UPDATE retrain_log SET status='failed', error_message=%s WHERE id=%s",
                    (str(e), retrain_id),
                )
            conn5.commit()
        finally:
            release_conn(conn5)
