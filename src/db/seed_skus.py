"""
seed_skus.py — Populate tc_sku_demand_panel from tc_demand_panel.

For every (depot, week) row in tc_demand_panel, disaggregates the
total demand_tonnes into 6 SKU rows using:

    sku_demand = depot_demand × mix_ratio × seasonal_mult × noise
    (then normalised so the 6 SKU rows sum exactly to depot_demand)

Seasonal multipliers capture product-specific construction patterns
for Sri Lanka (see SKU_SEASONAL dict below).
Inserts in batches of 500; ON CONFLICT DO UPDATE to be idempotent.
"""

import logging
import os
import random
from decimal import Decimal

import pandas as pd

from .db import get_client

logger = logging.getLogger(__name__)

# ── Product seasonal multipliers (month → relative demand) ────
# All rows are normalised during seeding so they always sum to
# the parent depot demand — these are *relative* shapes only.
SKU_SEASONAL: dict[str, dict[int, float]] = {
    "SUPERMIX": {          # General-purpose mix — mirrors base seasonal
        1: 0.97, 2: 1.08, 3: 1.12, 4: 1.10, 5: 1.05,
        6: 0.88, 7: 0.85, 8: 0.85, 9: 0.90,
        10: 1.05, 11: 1.05, 12: 1.00,
    },
    "SUPERFIX": {          # Tile adhesive — strong in dry finishing season
        1: 1.00, 2: 1.20, 3: 1.25, 4: 1.20, 5: 1.05,
        6: 0.82, 7: 0.78, 8: 0.78, 9: 0.85,
        10: 1.15, 11: 1.20, 12: 1.05,
    },
    "SUPERSEAL": {         # Waterproofing — spikes pre-monsoon (Apr–May)
        1: 0.95, 2: 1.05, 3: 1.15, 4: 1.35, 5: 1.30,
        6: 0.75, 7: 0.70, 8: 0.70, 9: 0.80,
        10: 1.00, 11: 0.95, 12: 0.90,
    },
    "SUPERSET": {          # Rapid-setting — post-monsoon construction restart
        1: 0.95, 2: 1.05, 3: 1.10, 4: 1.05, 5: 1.00,
        6: 0.85, 7: 0.80, 8: 0.80, 9: 0.90,
        10: 1.30, 11: 1.25, 12: 1.05,
    },
    "SUPERSCREED": {       # Floor levelling — dry-season finishing work
        1: 0.95, 2: 1.15, 3: 1.20, 4: 1.10, 5: 1.00,
        6: 0.80, 7: 0.75, 8: 0.75, 9: 0.85,
        10: 1.10, 11: 1.15, 12: 1.00,
    },
    "SUPERFLOW": {         # Self-levelling grout — specialised, relatively flat
        1: 0.95, 2: 1.05, 3: 1.10, 4: 1.05, 5: 1.00,
        6: 0.90, 7: 0.88, 8: 0.88, 9: 0.92,
        10: 1.00, 11: 1.05, 12: 1.00,
    },
}

NOISE_PCT = 0.05   # ±5 % uniform noise per row
BATCH_SIZE = 500


def seed_sku_demand_panel(seed: int = 42) -> int:
    """
    Read tc_demand_panel + tc_skus, disaggregate by product,
    and upsert into tc_sku_demand_panel.
    Returns number of rows written.
    """
    rng = random.Random(seed)
    sb = get_client()

    # ── Load SKU catalogue ────────────────────────────────────
    sku_result = sb.table("tc_skus").select("sku_id,sku_code,mix_ratio").execute()
    if not sku_result.data:
        logger.error("[SEED_SKU] tc_skus is empty — run sku_schema.sql first")
        return 0

    skus = sku_result.data  # list of {sku_id, sku_code, mix_ratio}
    logger.info("[SEED_SKU] %d SKUs loaded", len(skus))

    # ── Page through tc_demand_panel ─────────────────────────
    page_size = 1000
    start = 0
    all_panel: list[dict] = []
    while True:
        res = (
            sb.table("tc_demand_panel")
            .select("depot_id,week_start,demand_tonnes,sales_tonnes,data_source")
            .range(start, start + page_size - 1)
            .execute()
        )
        all_panel.extend(res.data)
        if len(res.data) < page_size:
            break
        start += page_size

    logger.info("[SEED_SKU] %d depot-week rows loaded from tc_demand_panel", len(all_panel))

    # ── Disaggregate ─────────────────────────────────────────
    rows: list[dict] = []

    for panel_row in all_panel:
        depot_id    = panel_row["depot_id"]
        week_start  = panel_row["week_start"]
        depot_demand = float(panel_row["demand_tonnes"] or 0.0)
        depot_sales  = float(panel_row["sales_tonnes"]  or 0.0)
        source       = panel_row["data_source"] or "augmented"
        month        = pd.Timestamp(week_start).month

        # Raw (un-normalised) values
        raw_demand: list[float] = []
        raw_sales:  list[float] = []
        for sku in skus:
            code        = sku["sku_code"]
            mix         = float(sku["mix_ratio"])
            seas        = SKU_SEASONAL.get(code, {}).get(month, 1.0)
            noise_d     = 1.0 + rng.uniform(-NOISE_PCT, NOISE_PCT)
            noise_s     = 1.0 + rng.uniform(-NOISE_PCT, NOISE_PCT)
            raw_demand.append(max(depot_demand * mix * seas * noise_d, 0.0))
            raw_sales.append( max(depot_sales  * mix * seas * noise_s, 0.0))

        # Normalise so SKU demands sum to depot total
        sum_d = sum(raw_demand) or 1.0
        sum_s = sum(raw_sales)  or 1.0

        for i, sku in enumerate(skus):
            norm_demand = round(raw_demand[i] / sum_d * depot_demand, 2)
            norm_sales  = round(raw_sales[i]  / sum_s * depot_sales,  2)
            rows.append({
                "depot_id":      depot_id,
                "sku_id":        sku["sku_id"],
                "week_start":    week_start if isinstance(week_start, str)
                                 else week_start.strftime("%Y-%m-%d"),
                "demand_tonnes": norm_demand,
                "sales_tonnes":  norm_sales,
                "data_source":   source,
            })

    # ── Batch upsert ─────────────────────────────────────────
    total = 0
    for i in range(0, len(rows), BATCH_SIZE):
        batch = rows[i: i + BATCH_SIZE]
        sb.table("tc_sku_demand_panel").upsert(
            batch, on_conflict="depot_id,sku_id,week_start"
        ).execute()
        total += len(batch)
        if (i // BATCH_SIZE) % 10 == 0:
            logger.info("[SEED_SKU] Upserted %d / %d rows", total, len(rows))

    logger.info("[SEED_SKU] ✓ Done — %d rows written to tc_sku_demand_panel", total)
    return total
