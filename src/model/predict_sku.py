"""
predict_sku.py — Load SKU XGBoost models and generate per-product forecasts.
"""

import logging
import os
from datetime import date, timedelta

import mlflow
import mlflow.xgboost
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_sku_models: dict[int, mlflow.pyfunc.PyFuncModel] = {}

SKU_REGISTRY = "cement_sku_forecaster"
LAG_WEEKS       = [1, 2, 3, 4, 6, 8, 12, 52]
ROLLING_WINDOWS = [4, 12]


def load_sku_models(cfg: dict) -> None:
    """Load all 6 SKU horizon models from MLflow Production registry."""
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "mlruns"))
    global _sku_models
    _sku_models = {}
    for h in cfg["model"]["horizons"]:
        model_name = f"{SKU_REGISTRY}_h{h}"
        uri = f"models:/{model_name}/Production"
        try:
            _sku_models[h] = mlflow.xgboost.load_model(uri)
            logger.info("[PREDICT_SKU] Loaded model horizon=%d", h)
        except Exception as e:
            logger.error("[PREDICT_SKU] Failed to load h%d: %s", h, e)
            raise


def _build_sku_feature_row(
    depot_name: str,
    sku_code: str,
    sku_enc: int,
    depot_enc: int,
    mix_ratio: float,
    as_of_date: date,
    recent_panel: pd.DataFrame,   # last 52 rows of tc_sku_demand_panel for this depot+SKU
    shared_row: pd.Series,        # latest tc_demand_panel row for this depot (weather/econ/cal)
    cfg: dict,
) -> pd.DataFrame:
    """Construct one feature row for a single (depot, SKU, as_of_date)."""

    panel = recent_panel.sort_values("week_start").copy()
    panel["demand_tonnes"] = pd.to_numeric(panel["demand_tonnes"], errors="coerce")
    demands = panel["demand_tonnes"].values

    row: dict = {}

    # Lag features
    for lag in LAG_WEEKS:
        row[f"demand_lag_{lag}"] = float(demands[-lag]) if len(demands) >= lag else np.nan

    # Rolling features
    for w in ROLLING_WINDOWS:
        window_vals = demands[-w:] if len(demands) >= w else demands
        row[f"demand_rolling_mean_{w}"] = float(np.mean(window_vals))

    std_window = demands[-4:] if len(demands) >= 4 else demands
    row["demand_rolling_std_4"] = float(np.std(std_window)) if len(std_window) > 1 else 0.0

    # Shared weather / econ / calendar from latest tc_demand_panel row
    carry_cols = [
        "precip_sum", "rain_sum", "temp_mean", "humidity_mean", "cloud_cover_mean",
        "gdp_lka", "lending_rate", "govt_consumption",
        "is_sw_monsoon", "is_ne_monsoon", "is_dry_season",
        "is_sinhala_tamil_new_year", "is_vesak", "is_christmas_week",
        "post_holiday_lag_1", "post_holiday_lag_2", "is_year_end_quarter",
        "week_of_year", "month",
    ]
    for col in carry_cols:
        try:
            val = shared_row.get(col)
            if val is None or (isinstance(val, float) and np.isnan(val)):
                row[col] = 0.0
            else:
                row[col] = float(val)
        except (KeyError, TypeError):
            row[col] = 0.0

    # Encodings
    row["depot_enc"] = depot_enc
    row["sku_enc"]   = sku_enc
    row["mix_ratio"] = mix_ratio

    # Interaction features — use _safe to guard against NaN
    def _safe(v):
        return 0.0 if (v is None or (isinstance(v, float) and np.isnan(v))) else float(v)

    precip  = _safe(row.get("precip_sum", 0))
    monsoon = _safe(row.get("is_sw_monsoon", 0))
    row["precip_x_monsoon"]          = precip * monsoon
    row["post_holiday_demand_boost"] = _safe(row.get("post_holiday_lag_1", 0)) * _safe(row.get("demand_rolling_mean_4", 0))
    row["sku_x_sw_monsoon"]          = float(sku_enc) * monsoon
    row["sku_x_ne_monsoon"]          = float(sku_enc) * _safe(row.get("is_ne_monsoon", 0))
    row["sku_x_dry_season"]          = float(sku_enc) * _safe(row.get("is_dry_season", 0))
    row["sku_x_mix_ratio"]           = float(sku_enc) * mix_ratio

    row["quarter"] = ((int(row.get("month", 1)) - 1) // 3) + 1

    return pd.DataFrame([row])


def forecast_sku_depot(
    depot_name: str,
    depot_id: int,
    as_of_date: date,
    cfg: dict,
) -> list[dict]:
    """
    Generate 6-week × 6-SKU forecasts for one depot.
    Returns list of dicts with depot_id, sku_id, sku_code, horizon, forecast_week, demand_tonnes.
    """
    if not _sku_models:
        raise RuntimeError("[PREDICT_SKU] Models not loaded. Call load_sku_models() first.")

    from src.db.db import get_client
    sb = get_client()

    # Fetch SKU catalogue
    sku_res = sb.table("tc_skus").select("sku_id,sku_code,name,mix_ratio").order("sku_code").execute()
    skus = sku_res.data

    # Depot encoding (alphabetical order of depot names)
    depot_list = sb.table("tc_depots").select("name").order("name").execute()
    depot_names = [r["name"] for r in depot_list.data]
    depot_enc_map = {n: i for i, n in enumerate(depot_names)}
    depot_enc = depot_enc_map.get(depot_name, 0)

    sku_codes_sorted = sorted(s["sku_code"] for s in skus)
    sku_enc_map = {c: i for i, c in enumerate(sku_codes_sorted)}

    # Latest shared row from tc_demand_panel for weather/econ/cal
    shared_res = (
        sb.table("tc_demand_panel")
        .select("*")
        .eq("depot_id", depot_id)
        .order("week_start", desc=True)
        .limit(1)
        .execute()
    )
    shared_row = pd.Series(shared_res.data[0]) if shared_res.data else pd.Series(dtype=float)

    results: list[dict] = []

    for sku in skus:
        sku_id    = sku["sku_id"]
        sku_code  = sku["sku_code"]
        sku_enc   = sku_enc_map.get(sku_code, 0)
        mix_ratio = float(sku["mix_ratio"])

        # Last 52 weeks of SKU demand for this depot
        sku_panel_res = (
            sb.table("tc_sku_demand_panel")
            .select("week_start,demand_tonnes")
            .eq("depot_id", depot_id)
            .eq("sku_id", sku_id)
            .order("week_start", desc=True)
            .limit(52)
            .execute()
        )
        recent = pd.DataFrame(sku_panel_res.data)
        if recent.empty:
            logger.warning("[PREDICT_SKU] No panel data for depot=%s sku=%s", depot_name, sku_code)
            continue

        recent["week_start"] = pd.to_datetime(recent["week_start"])
        recent = recent.sort_values("week_start").reset_index(drop=True)

        X = _build_sku_feature_row(
            depot_name, sku_code, sku_enc, depot_enc, mix_ratio,
            as_of_date, recent, shared_row, cfg,
        )

        for h in cfg["model"]["horizons"]:
            model = _sku_models[h]
            try:
                feat_names = list(model.feature_names_in_)
                for col in feat_names:
                    if col not in X.columns:
                        X[col] = 0.0
                pred = float(model.predict(X[feat_names])[0])
                pred = max(pred, 0.0)
            except Exception as e:
                logger.warning("[PREDICT_SKU] h%d sku=%s failed: %s", h, sku_code, e)
                pred = 0.0

            forecast_week = as_of_date + timedelta(weeks=h)
            results.append({
                "depot_id":      depot_id,
                "sku_id":        sku_id,
                "sku_code":      sku_code,
                "sku_name":      sku["name"],
                "horizon":       h,
                "forecast_week": forecast_week,
                "demand_tonnes": round(pred, 2),
            })

    return results
