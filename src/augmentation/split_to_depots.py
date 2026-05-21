import logging
import os

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def split_to_depots(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    Split national weekly series to 24 depot-level series using district population weights.
    Adds ±3% depot-specific noise per week for realism.
    Reads:  kaggle_seasadj.csv DataFrame (national weekly)
    Writes: data/interim/kaggle_depot_split.csv
    """
    out_path = os.path.join(cfg["paths"]["interim"], "kaggle_depot_split.csv")
    aug = cfg["augmentation"]
    rng = np.random.default_rng(int(aug["depot_noise_seed"]))
    noise_pct = float(aug["depot_noise_pct"])

    depots = cfg["depots"]
    total_weight = sum(d["pop_weight"] for d in depots)
    if abs(total_weight - 1.0) > 0.01:
        logger.warning("[SPLIT] Depot weights sum to %.4f (expected 1.0)", total_weight)

    rows = []
    for _, week_row in df.iterrows():
        week_start = week_row["week_start"]
        national_sales = week_row.get("Sales", 0)
        national_demand = week_row.get("Demand", 0)
        national_prod = week_row.get("Production", 0)

        for depot in depots:
            weight = depot["pop_weight"]
            noise = 1.0 + rng.uniform(-noise_pct, noise_pct)
            rows.append({
                "week_start":       week_start,
                "depot":            depot["name"],
                "sales_tonnes":     round(national_sales * weight * noise, 2),
                "demand_tonnes":    round(national_demand * weight * noise, 2),
                "production_tonnes": round(national_prod * weight * noise, 2),
            })

    result = pd.DataFrame(rows).sort_values(["week_start", "depot"]).reset_index(drop=True)

    os.makedirs(cfg["paths"]["interim"], exist_ok=True)
    result.to_csv(out_path, index=False)
    logger.info("[SPLIT] Depot split: %d rows (%d depots × %d weeks) -> %s",
                len(result), len(depots), len(df), out_path)
    return result
