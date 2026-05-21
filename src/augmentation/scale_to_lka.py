import logging
import os

import pandas as pd

logger = logging.getLogger(__name__)


def scale_to_lka(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    Scale Indian cement volumes to Tokyo Cement Sri Lanka volumes.
    Applies 2022 crisis shock multiplier.
    Reads:  economic_lka.csv DataFrame (monthly)
    Writes: data/interim/kaggle_scaled.csv
    """
    out_path = os.path.join(cfg["paths"]["interim"], "kaggle_scaled.csv")
    aug = cfg["augmentation"]

    # Identify volume columns (case-insensitive)
    col_map = {}
    for c in df.columns:
        lc = c.lower()
        if lc == "sales":
            col_map[c] = "Sales"
        elif lc == "demand":
            col_map[c] = "Demand"
        elif lc == "production":
            col_map[c] = "Production"
    df = df.rename(columns=col_map)

    vol_cols = [c for c in ["Production", "Sales", "Demand"] if c in df.columns]
    if not vol_cols:
        raise ValueError("[SCALE] No volume columns found in DataFrame")

    # Compute scale factor
    mean_sales = df["Sales"].mean() if "Sales" in df.columns else df[vol_cols[0]].mean()
    scale_factor = float(aug["tokyo_monthly_tonnes"]) / mean_sales
    logger.info("[SCALE] Scale factor: %.4f (target monthly tonnes: %s, Kaggle mean: %.1f)",
                scale_factor, aug["tokyo_monthly_tonnes"], mean_sales)

    result = df.copy()
    for col in vol_cols:
        result[col] = result[col] * scale_factor

    # 2022 crisis shock
    crisis_year = int(aug["crisis_year"])
    crisis_mult = float(aug["crisis_multiplier"])
    mask = result["Month"].dt.year == crisis_year
    for col in ["Sales", "Demand"]:
        if col in result.columns:
            result.loc[mask, col] *= crisis_mult
    logger.info("[SCALE] Applied %.0f%% crisis shock to %d rows in %d",
                crisis_mult * 100, mask.sum(), crisis_year)

    os.makedirs(cfg["paths"]["interim"], exist_ok=True)
    result.to_csv(out_path, index=False)
    logger.info("[SCALE] scale_to_lka: %d rows -> %s", len(result), out_path)
    return result
