import logging
import os
import time
from datetime import date, timedelta

import pandas as pd
import requests

logger = logging.getLogger(__name__)

HOURLY_VARS = "temperature_2m,relative_humidity_2m,precipitation,rain,cloud_cover"


def _fetch_open_meteo(url: str, params: dict, retries: int = 3) -> pd.DataFrame:
    """Fetch CSV from Open-Meteo and return as DataFrame."""
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, timeout=60)
            r.raise_for_status()
            from io import StringIO
            lines = r.text.splitlines()
            # Skip header comment lines (start with #)
            data_lines = [l for l in lines if not l.startswith("#")]
            df = pd.read_csv(StringIO("\n".join(data_lines)))
            return df
        except Exception as e:
            if attempt == retries - 1:
                raise
            logger.warning("[WEATHER] Attempt %d failed: %s — retrying", attempt + 1, e)
            time.sleep(2 ** attempt)


def _agg_hourly_to_weekly(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate hourly weather data to weekly (Monday-anchored)."""
    time_col = next((c for c in df.columns if "time" in c.lower()), df.columns[0])
    df[time_col] = pd.to_datetime(df[time_col])
    df = df.rename(columns={time_col: "timestamp"})
    df["week_start"] = df["timestamp"].dt.to_period("W-SUN").apply(lambda p: p.start_time.date())

    agg = {}
    for col in df.columns:
        if col in ("timestamp", "week_start"):
            continue
        if "precipitation" in col or "rain" in col:
            agg[col] = "sum"
        else:
            agg[col] = "mean"

    weekly = df.groupby("week_start").agg(agg).reset_index()
    # Rename columns to standard names
    rename = {}
    for c in weekly.columns:
        if "precipitation" in c:
            rename[c] = "precip_sum"
        elif "rain" in c:
            rename[c] = "rain_sum"
        elif "temperature" in c:
            rename[c] = "temp_mean"
        elif "relative_humidity" in c:
            rename[c] = "humidity_mean"
        elif "cloud_cover" in c:
            rename[c] = "cloud_cover_mean"
    weekly = weekly.rename(columns=rename)
    weekly["week_start"] = pd.to_datetime(weekly["week_start"])
    return weekly


def fetch_depot_weather(depot: dict, cfg: dict, tier: str = "all") -> pd.DataFrame:
    """
    Fetch weather for a single depot across all applicable tiers.
    tier: 'all' (setup) | 'tier3' (update)
    Returns weekly-aggregated DataFrame.
    """
    lat, lon = depot["lat"], depot["lon"]
    tz = cfg["weather"]["timezone"]
    urls = cfg["weather"]["urls"]
    lag_days = int(cfg["weather"]["tier2_lag_days"])
    today = date.today()
    tier2_end = today - timedelta(days=lag_days)
    tier2_start = cfg["weather"]["tier2_start"]
    era5_end = cfg["weather"]["era5_end"]

    base_params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": HOURLY_VARS,
        "timezone": tz,
        "format": "csv",
    }

    frames = []

    if tier in ("all",):
        # Tier 1 — ERA5 historical
        logger.info("[WEATHER] Tier1 ERA5 for depot %s", depot["name"])
        p = {**base_params, "start_date": cfg["weather"]["era5_start"],
             "end_date": era5_end, "models": "era5"}
        try:
            df1 = _fetch_open_meteo(urls["era5"], p)
            frames.append(_agg_hourly_to_weekly(df1))
        except Exception as e:
            logger.error("[WEATHER] Tier1 failed for %s: %s", depot["name"], e)
            raise

        # Tier 2 — historical forecast archive
        if tier2_end > pd.Timestamp(tier2_start).date():
            logger.info("[WEATHER] Tier2 historical-forecast for depot %s", depot["name"])
            p2 = {**base_params, "start_date": tier2_start,
                  "end_date": tier2_end.isoformat()}
            try:
                df2 = _fetch_open_meteo(urls["tier2"], p2)
                frames.append(_agg_hourly_to_weekly(df2))
            except Exception as e:
                logger.warning("[WEATHER] Tier2 failed for %s (non-fatal): %s", depot["name"], e)

    # Tier 3 — current rolling window (used by both setup and update)
    logger.info("[WEATHER] Tier3 current for depot %s", depot["name"])
    p3 = {**base_params, "past_days": lag_days, "forecast_days": 0}
    try:
        df3 = _fetch_open_meteo(urls["current"], p3)
        frames.append(_agg_hourly_to_weekly(df3))
    except Exception as e:
        logger.warning("[WEATHER] Tier3 failed for %s (non-fatal): %s", depot["name"], e)

    if not frames:
        raise RuntimeError(f"[WEATHER] All tiers failed for depot {depot['name']}")

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.drop_duplicates(subset=["week_start"]).sort_values("week_start").reset_index(drop=True)
    return combined


def fetch_all_depots_weather(cfg: dict, tier: str = "all") -> None:
    """Fetch weather for all 24 depots and save per-depot CSVs."""
    out_dir = cfg["paths"]["raw_weather"]
    os.makedirs(out_dir, exist_ok=True)

    depots = cfg["depots"]
    for depot in depots:
        out_path = os.path.join(out_dir, f"{depot['name'].lower().replace(' ', '_')}.csv")
        if tier == "all" and os.path.exists(out_path):
            logger.info("[WEATHER] Already exists, skipping: %s", out_path)
            continue
        logger.info("[WEATHER] Fetching depot: %s", depot["name"])
        df = fetch_depot_weather(depot, cfg, tier=tier)
        df.to_csv(out_path, index=False)
        logger.info("[WEATHER] Saved %d weekly rows -> %s", len(df), out_path)
        time.sleep(0.5)  # gentle rate limiting
