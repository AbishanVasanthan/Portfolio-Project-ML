import logging
import os

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

VOL_COLS = ["Production", "Sales", "Demand"]


def _load_colombo_weekly_precip(cfg: dict) -> pd.Series:
    """Load Colombo weather CSV and return mean weekly precipitation indexed by calendar week."""
    weather_dir = cfg["paths"]["raw_weather"]
    colombo_path = os.path.join(weather_dir, "colombo.csv")
    if not os.path.exists(colombo_path):
        logger.warning("[SEASONAL] Colombo weather not found — using flat multiplier")
        return None

    df = pd.read_csv(colombo_path, parse_dates=["week_start"])
    if "precip_sum" not in df.columns:
        logger.warning("[SEASONAL] precip_sum not in Colombo weather — using flat multiplier")
        return None

    df["calendar_week"] = df["week_start"].dt.isocalendar().week.astype(int)
    return df.groupby("calendar_week")["precip_sum"].mean()


def seasonal_override(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    Strip Indian seasonal pattern, apply Sri Lanka monsoon-driven seasonal curve.
    Reads:  kaggle_weekly.csv DataFrame
    Writes: data/interim/kaggle_seasadj.csv
    """
    out_path = os.path.join(cfg["paths"]["interim"], "kaggle_seasadj.csv")
    aug = cfg["augmentation"]

    result = df.copy()
    result["week_start"] = pd.to_datetime(result["week_start"])
    result["_month"] = result["week_start"].dt.month
    result["_week"] = result["week_start"].dt.isocalendar().week.astype(int)

    # Step 1: Compute Indian seasonal index per calendar month
    if "Sales" in result.columns:
        global_mean = result["Sales"].mean()
        monthly_means = result.groupby("_month")["Sales"].mean()
        indian_si = (monthly_means / global_mean).to_dict()
    else:
        indian_si = {m: 1.0 for m in range(1, 13)}

    # Step 2: Deseasonalise
    for col in VOL_COLS:
        if col in result.columns:
            si = result["_month"].map(indian_si).fillna(1.0)
            result[col] = result[col] / si

    # Step 3: Build LKA seasonal multiplier from Colombo weather
    peak = float(aug["seasonal_peak"])
    rng = float(aug["seasonal_range"])

    precip_by_week = _load_colombo_weekly_precip(cfg)
    if precip_by_week is not None:
        max_precip = precip_by_week.max()
        if max_precip > 0:
            norm = precip_by_week / max_precip
        else:
            norm = pd.Series(0.0, index=precip_by_week.index)
        lka_mult = (peak - norm * rng).to_dict()
    else:
        # Flat multiplier if no weather data
        lka_mult = {w: 1.0 for w in range(1, 54)}

    # Step 4: Re-apply LKA seasonal curve
    for col in VOL_COLS:
        if col in result.columns:
            mult = result["_week"].map(lka_mult).fillna(1.0)
            result[col] = result[col] * mult

    result = result.drop(columns=["_month", "_week"])

    os.makedirs(cfg["paths"]["interim"], exist_ok=True)
    result.to_csv(out_path, index=False)
    logger.info("[AUG] seasonal_override: %d rows -> %s", len(result), out_path)
    return result
