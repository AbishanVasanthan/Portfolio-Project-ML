import logging
import os

import pandas as pd

logger = logging.getLogger(__name__)


def replace_economics(kaggle_df: pd.DataFrame, worldbank_weekly: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    Drop Indian economic columns from the Kaggle CSV and join Sri Lanka
    World Bank equivalents aligned by month.
    Reads:  raw Kaggle DataFrame (monthly)
    Writes: data/interim/economic_lka.csv
    """
    out_path = os.path.join(cfg["paths"]["interim"], "economic_lka.csv")

    # Drop Indian economic columns
    drop_cols = [c for c in ["Population", "GDP", "Disbursement", "Interest_Rate",
                              "population", "gdp", "disbursement", "interest_rate"]
                 if c in kaggle_df.columns]
    df = kaggle_df.drop(columns=drop_cols)

    # Align World Bank weekly → monthly (pick first week of each month)
    wb = worldbank_weekly.copy()
    wb["month_key"] = wb["week_start"].dt.to_period("M").dt.to_timestamp()
    wb_monthly = wb.drop_duplicates(subset=["month_key"]).set_index("month_key")
    wb_cols = [c for c in ["gdp_lka", "population_lka", "lending_rate", "govt_consumption"]
               if c in wb_monthly.columns]
    wb_monthly = wb_monthly[wb_cols]

    df["month_key"] = df["Month"].dt.to_period("M").dt.to_timestamp()
    df = df.merge(wb_monthly.reset_index(), on="month_key", how="left")
    df = df.drop(columns=["month_key"])

    os.makedirs(cfg["paths"]["interim"], exist_ok=True)
    df.to_csv(out_path, index=False)
    logger.info("[AUG] replace_economics: %d rows -> %s", len(df), out_path)
    return df
