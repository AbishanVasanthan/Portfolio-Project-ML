import logging
import os

import numpy as np
import pandas as pd
import requests

logger = logging.getLogger(__name__)


# ── World Bank via wbgapi ─────────────────────────────────────

def fetch_worldbank(cfg: dict) -> pd.DataFrame:
    """Fetch World Bank annual indicators for Sri Lanka and interpolate to weekly."""
    import wbgapi as wb

    out_path = os.path.join(cfg["paths"]["raw_economic"], "worldbank_lka.csv")
    if os.path.exists(out_path):
        logger.info("[ECONOMIC] World Bank CSV already exists — skipping fetch")
        return pd.read_csv(out_path, parse_dates=["week_start"])

    wb_cfg = cfg["worldbank"]
    indicators = list(wb_cfg["indicators"].values())
    years = range(int(wb_cfg["start_year"]), int(wb_cfg["end_year"]) + 1)

    logger.info("[ECONOMIC] Fetching World Bank indicators: %s", indicators)
    try:
        df = wb.data.DataFrame(indicators, economy="LKA", time=years)
        df = df.reset_index()
    except Exception as e:
        raise RuntimeError(f"[ECONOMIC] World Bank fetch failed: {e}") from e

    # wbgapi returns columns as indicator codes, index as year
    df = df.rename(columns={
        wb_cfg["indicators"]["gdp"]:             "gdp_lka",
        wb_cfg["indicators"]["population"]:      "population_lka",
        wb_cfg["indicators"]["lending_rate"]:    "lending_rate",
        wb_cfg["indicators"]["govt_consumption"]:"govt_consumption",
        "time":                                  "year",
        "economy":                               "economy",
    })

    # Drop economy column (only LKA)
    df = df.drop(columns=[c for c in ["economy"] if c in df.columns], errors="ignore")

    # Parse year
    if "year" in df.columns:
        df["year"] = df["year"].astype(str).str.extract(r"(\d{4})").astype(int)
    else:
        df = df.reset_index()
        df["year"] = df["time"].astype(str).str.extract(r"(\d{4})").astype(int)
        df = df.drop(columns=["time"], errors="ignore")

    df = df.sort_values("year").reset_index(drop=True)

    # Forward-fill missing years (World Bank publication lag)
    for col in ["gdp_lka", "population_lka", "lending_rate", "govt_consumption"]:
        if col in df.columns:
            df[col] = df[col].fillna(method="ffill")

    # Expand annual → monthly via linear interpolation
    months = pd.date_range(
        start=f"{df['year'].min()}-01-01",
        end=f"{df['year'].max()}-12-01",
        freq="MS",
    )
    monthly = pd.DataFrame({"month": months})
    monthly["year"] = monthly["month"].dt.year

    monthly = monthly.merge(df, on="year", how="left")
    for col in ["gdp_lka", "population_lka", "lending_rate", "govt_consumption"]:
        if col in monthly.columns:
            monthly[col] = monthly[col].interpolate(method="linear")

    # Monthly → weekly (forward fill)
    weeks = pd.date_range(start="2010-01-04", end="2030-12-29", freq="W-MON")
    weekly = pd.DataFrame({"week_start": weeks})
    weekly["month_key"] = weekly["week_start"].dt.to_period("M").dt.to_timestamp()
    monthly = monthly.rename(columns={"month": "month_key"})
    weekly = weekly.merge(monthly.drop(columns=["year"]), on="month_key", how="left")
    for col in ["gdp_lka", "population_lka", "lending_rate", "govt_consumption"]:
        if col in weekly.columns:
            weekly[col] = weekly[col].fillna(method="ffill")
    weekly = weekly.drop(columns=["month_key"])

    os.makedirs(cfg["paths"]["raw_economic"], exist_ok=True)
    weekly.to_csv(out_path, index=False)
    logger.info("[ECONOMIC] World Bank weekly data saved: %d rows -> %s", len(weekly), out_path)
    return weekly


# ── Pink Sheet (Metals index as clinker proxy) ────────────────

def fetch_pink_sheet(cfg: dict) -> pd.DataFrame:
    """Download World Bank Pink Sheet and extract Metals & Minerals index."""
    out_path = os.path.join(cfg["paths"]["raw_economic"], "pink_sheet_metals.csv")
    if os.path.exists(out_path):
        logger.info("[ECONOMIC] Pink Sheet already exists — skipping")
        return pd.read_csv(out_path)

    url = cfg["worldbank"]["pink_sheet_url"]
    logger.info("[ECONOMIC] Downloading Pink Sheet from %s", url)
    try:
        r = requests.get(url, timeout=120)
        r.raise_for_status()
    except Exception as e:
        logger.warning("[ECONOMIC] Pink Sheet download failed (non-fatal): %s", e)
        return pd.DataFrame(columns=["month", "metals_index"])

    xl_path = out_path.replace(".csv", ".xlsx")
    with open(xl_path, "wb") as f:
        f.write(r.content)

    try:
        xls = pd.ExcelFile(xl_path)
        sheet = next((s for s in xls.sheet_names if "month" in s.lower() or "price" in s.lower()), xls.sheet_names[0])
        raw = pd.read_excel(xl_path, sheet_name=sheet, header=None)

        # Find the Metals & Minerals row
        metals_row = None
        for i, row in raw.iterrows():
            if any("metal" in str(v).lower() for v in row.values):
                metals_row = i
                break

        if metals_row is None:
            logger.warning("[ECONOMIC] Could not find Metals & Minerals row in Pink Sheet")
            return pd.DataFrame(columns=["month", "metals_index"])

        # Header row is typically 4 rows before data
        header_row = raw.iloc[metals_row - 1]
        values = raw.iloc[metals_row]

        result = pd.DataFrame({
            "month": header_row.values[1:],
            "metals_index": pd.to_numeric(values.values[1:], errors="coerce"),
        }).dropna()
        result["month"] = pd.to_datetime(result["month"], errors="coerce")
        result = result.dropna(subset=["month"]).sort_values("month")
        result.to_csv(out_path, index=False)
        logger.info("[ECONOMIC] Pink Sheet saved: %d rows", len(result))
        return result
    except Exception as e:
        logger.warning("[ECONOMIC] Pink Sheet parse failed (non-fatal): %s", e)
        return pd.DataFrame(columns=["month", "metals_index"])


# ── CBSL PMI ──────────────────────────────────────────────────

def load_cbsl_pmi(cfg: dict) -> pd.DataFrame:
    """
    Load CBSL Construction PMI from data/raw/economic/cbsl_pmi_construction.csv if present.
    Falls back to backward-fill from mean of first 6 available readings for 2010-2017.
    Returns weekly DataFrame with columns [week_start, cbsl_pmi_construction].
    """
    pmi_path = os.path.join(cfg["paths"]["raw_economic"], "cbsl_pmi_construction.csv")

    if not os.path.exists(pmi_path):
        logger.warning("[ECONOMIC] CBSL PMI file not found at %s — dropping PMI column", pmi_path)
        return pd.DataFrame(columns=["week_start", "cbsl_pmi_construction"])

    try:
        pmi = pd.read_csv(pmi_path)
        pmi.columns = [c.strip().lower().replace(" ", "_") for c in pmi.columns]
        pmi = pmi.rename(columns={pmi.columns[0]: "month", pmi.columns[1]: "cbsl_pmi_construction"})
        pmi["month"] = pd.to_datetime(pmi["month"], infer_datetime_format=True)
        pmi = pmi.sort_values("month").dropna(subset=["cbsl_pmi_construction"])

        # Backward-fill 2010-2017 gap from mean of first 6 available readings
        backfill_value = float(pmi["cbsl_pmi_construction"].head(6).mean())
        logger.info("[ECONOMIC] CBSL PMI backfill value (mean first 6): %.2f", backfill_value)

        # Build weekly series
        weeks = pd.date_range(start="2010-01-04", end="2030-12-29", freq="W-MON")
        weekly = pd.DataFrame({"week_start": weeks})
        weekly["month_key"] = weekly["week_start"].dt.to_period("M").dt.to_timestamp()
        pmi_m = pmi.rename(columns={"month": "month_key"})
        weekly = weekly.merge(pmi_m, on="month_key", how="left")
        weekly["cbsl_pmi_construction"] = weekly["cbsl_pmi_construction"].fillna(backfill_value)
        return weekly[["week_start", "cbsl_pmi_construction"]]
    except Exception as e:
        logger.warning("[ECONOMIC] CBSL PMI parse failed (non-fatal): %s — dropping PMI", e)
        return pd.DataFrame(columns=["week_start", "cbsl_pmi_construction"])
