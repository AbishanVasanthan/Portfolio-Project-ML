"""
build_sku_features.py — Build the training feature panel for SKU-level models.

Joins tc_sku_demand_panel (demand per product) with tc_demand_panel
(weather, economics, calendar) and adds:
  - Lag features per (depot, SKU)
  - Rolling statistics per (depot, SKU)
  - SKU encoding
  - SKU × season interaction features
"""

import logging
import os

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

LAG_WEEKS       = [1, 2, 3, 4, 6, 8, 12, 52]
ROLLING_WINDOWS = [4, 12]

SHARED_COLS = [
    "precip_sum", "rain_sum", "temp_mean", "humidity_mean", "cloud_cover_mean",
    "gdp_lka", "lending_rate", "govt_consumption",
    "is_sw_monsoon", "is_ne_monsoon", "is_dry_season",
    "is_sinhala_tamil_new_year", "is_vesak", "is_christmas_week",
    "post_holiday_lag_1", "post_holiday_lag_2", "is_year_end_quarter",
    "week_of_year", "month",
]


def rebuild_sku_features(cfg: dict) -> pd.DataFrame:
    """
    Pull tc_sku_demand_panel + tc_demand_panel from DB, join, engineer
    features and return a ready-to-train DataFrame.
    """
    from src.db.db import get_client
    sb = get_client()

    # ── 1. Load SKU demand panel (paginated) ──────────────────
    logger.info("[SKU_FEAT] Loading tc_sku_demand_panel from DB")
    page, all_rows = 0, []
    page_size = 1000
    while True:
        res = (
            sb.table("tc_sku_demand_panel")
            .select("depot_id,sku_id,week_start,demand_tonnes,sales_tonnes,data_source")
            .range(page, page + page_size - 1)
            .execute()
        )
        all_rows.extend(res.data)
        if len(res.data) < page_size:
            break
        page += page_size

    if not all_rows:
        logger.error("[SKU_FEAT] tc_sku_demand_panel is empty — run --mode setup_skus first")
        return pd.DataFrame()

    sku_df = pd.DataFrame(all_rows)
    sku_df["week_start"]     = pd.to_datetime(sku_df["week_start"])
    sku_df["demand_tonnes"]  = pd.to_numeric(sku_df["demand_tonnes"], errors="coerce").fillna(0.0)
    logger.info("[SKU_FEAT] %d SKU-panel rows loaded", len(sku_df))

    # ── 2. Load shared features from tc_demand_panel (paginated) ─
    logger.info("[SKU_FEAT] Loading shared features from tc_demand_panel")
    page, panel_rows = 0, []
    select_cols = "depot_id,week_start," + ",".join(SHARED_COLS)
    while True:
        res = (
            sb.table("tc_demand_panel")
            .select(select_cols)
            .range(page, page + page_size - 1)
            .execute()
        )
        panel_rows.extend(res.data)
        if len(res.data) < page_size:
            break
        page += page_size

    panel_df = pd.DataFrame(panel_rows)
    panel_df["week_start"] = pd.to_datetime(panel_df["week_start"])
    for col in SHARED_COLS:
        if col in panel_df.columns:
            panel_df[col] = pd.to_numeric(panel_df[col], errors="coerce")

    # ── 3. Load depot and SKU name maps ───────────────────────
    depot_res = sb.table("tc_depots").select("depot_id,name").execute()
    depot_map = {r["depot_id"]: r["name"] for r in depot_res.data}

    sku_res = sb.table("tc_skus").select("sku_id,sku_code,name,mix_ratio").execute()
    sku_map    = {r["sku_id"]: r["sku_code"]  for r in sku_res.data}
    sku_name   = {r["sku_id"]: r["name"]      for r in sku_res.data}
    sku_ratio  = {r["sku_id"]: float(r["mix_ratio"]) for r in sku_res.data}

    # ── 4. Join shared features ───────────────────────────────
    df = sku_df.merge(panel_df, on=["depot_id", "week_start"], how="left")
    df["depot"]    = df["depot_id"].map(depot_map)
    df["sku_code"] = df["sku_id"].map(sku_map)
    df["mix_ratio"]= df["sku_id"].map(sku_ratio).astype(float)
    df = df.sort_values(["depot", "sku_code", "week_start"]).reset_index(drop=True)

    # ── 5. Lag + rolling features per (depot, SKU) ───────────
    group = df.groupby(["depot", "sku_code"])["demand_tonnes"]

    for lag in LAG_WEEKS:
        df[f"demand_lag_{lag}"] = group.shift(lag)

    for w in ROLLING_WINDOWS:
        df[f"demand_rolling_mean_{w}"] = group.transform(
            lambda s: s.shift(1).rolling(w, min_periods=1).mean()
        )

    df["demand_rolling_std_4"] = group.transform(
        lambda s: s.shift(1).rolling(4, min_periods=2).std(ddof=0)
    ).fillna(0.0)

    # ── 6. SKU encoding ───────────────────────────────────────
    sku_codes_sorted = sorted(df["sku_code"].unique())
    sku_enc_map = {c: i for i, c in enumerate(sku_codes_sorted)}
    df["sku_enc"] = df["sku_code"].map(sku_enc_map).astype(int)

    # ── 7. Depot encoding ─────────────────────────────────────
    depot_names_sorted = sorted(df["depot"].unique())
    depot_enc_map = {n: i for i, n in enumerate(depot_names_sorted)}
    df["depot_enc"] = df["depot"].map(depot_enc_map).astype(int)

    # ── 8. Interaction features ───────────────────────────────
    precip  = df["precip_sum"].fillna(0.0)
    monsoon = df["is_sw_monsoon"].fillna(0).astype(float)
    df["precip_x_monsoon"] = precip * monsoon

    ph_lag1   = df["post_holiday_lag_1"].fillna(0).astype(float)
    roll_mean = df["demand_rolling_mean_4"].fillna(0.0)
    df["post_holiday_demand_boost"] = ph_lag1 * roll_mean

    # SKU × season interactions (product-specific seasonal signals)
    df["sku_x_sw_monsoon"]  = df["sku_enc"].astype(float) * monsoon
    df["sku_x_ne_monsoon"]  = df["sku_enc"].astype(float) * df["is_ne_monsoon"].fillna(0).astype(float)
    df["sku_x_dry_season"]  = df["sku_enc"].astype(float) * df["is_dry_season"].fillna(0).astype(float)
    df["sku_x_mix_ratio"]   = df["sku_enc"].astype(float) * df["mix_ratio"]

    df["week_of_year"] = df["week_start"].dt.isocalendar().week.astype(int)
    df["month"]        = df["week_start"].dt.month.astype(int)
    df["quarter"]      = df["week_start"].dt.quarter.astype(int)

    logger.info(
        "[SKU_FEAT] Feature panel ready: %d rows, %d depots × %d SKUs",
        len(df), df["depot"].nunique(), df["sku_code"].nunique(),
    )
    return df
