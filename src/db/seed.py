import logging
import pandas as pd
import psycopg2.extras
from src.db.db import get_conn, release_conn

logger = logging.getLogger(__name__)


def seed_depots(depots_cfg: list[dict]) -> dict[str, int]:
    """Insert depot rows (idempotent). Returns {name: depot_id} map."""
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            for d in depots_cfg:
                cur.execute(
                    """
                    INSERT INTO depots (name, district, province, latitude, longitude, pop_weight)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    ON CONFLICT (name) DO NOTHING
                    """,
                    (d["name"], d["district"], d["province"],
                     d["lat"], d["lon"], d["pop_weight"]),
                )
            conn.commit()
            cur.execute("SELECT name, depot_id FROM depots")
            rows = cur.fetchall()
        depot_map = {r[0]: r[1] for r in rows}
        logger.info("[SEED] Depots seeded: %d", len(depot_map))
        return depot_map
    except Exception:
        conn.rollback()
        raise
    finally:
        release_conn(conn)


def seed_demand_panel(panel_path: str, depot_map: dict[str, int]) -> int:
    """Bulk-insert panel rows into demand_panel. Idempotent (ON CONFLICT DO NOTHING)."""
    df = pd.read_csv(panel_path, parse_dates=["week_start"])
    df["depot_id"] = df["depot"].map(depot_map)

    missing = df[df["depot_id"].isna()]["depot"].unique()
    if len(missing):
        raise ValueError(f"Unknown depots in panel: {missing}")

    col_order = [
        "depot_id", "week_start", "demand_tonnes", "sales_tonnes", "production_tonnes",
        "precip_sum", "rain_sum", "temp_mean", "humidity_mean", "cloud_cover_mean",
        "gdp_lka", "lending_rate", "cbsl_pmi_construction", "govt_consumption",
        "is_sw_monsoon", "is_ne_monsoon", "is_dry_season",
        "is_sinhala_tamil_new_year", "is_vesak", "is_christmas_week",
        "post_holiday_lag_1", "post_holiday_lag_2", "is_year_end_quarter",
    ]
    # fill missing optional columns with None
    for c in col_order:
        if c not in df.columns:
            df[c] = None

    records = [tuple(row) for row in df[col_order].itertuples(index=False, name=None)]

    conn = get_conn()
    try:
        with conn.cursor() as cur:
            psycopg2.extras.execute_values(
                cur,
                """
                INSERT INTO demand_panel
                    (depot_id, week_start, demand_tonnes, sales_tonnes, production_tonnes,
                     precip_sum, rain_sum, temp_mean, humidity_mean, cloud_cover_mean,
                     gdp_lka, lending_rate, cbsl_pmi_construction, govt_consumption,
                     is_sw_monsoon, is_ne_monsoon, is_dry_season,
                     is_sinhala_tamil_new_year, is_vesak, is_christmas_week,
                     post_holiday_lag_1, post_holiday_lag_2, is_year_end_quarter)
                VALUES %s
                ON CONFLICT (depot_id, week_start) DO NOTHING
                """,
                records,
                page_size=500,
            )
        conn.commit()
        logger.info("[SEED] demand_panel rows inserted: %d", len(records))
        return len(records)
    except Exception:
        conn.rollback()
        raise
    finally:
        release_conn(conn)
