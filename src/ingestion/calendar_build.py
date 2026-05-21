import logging
import os
from datetime import date

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _vesak_week(year: int) -> date:
    """Approximate Vesak Poya: first full moon in May. Returns a Monday."""
    # Full moon cycle ≈ 29.53 days. Use a lookup for accuracy.
    # Known Vesak dates (day of Poya):
    vesak_dates = {
        2010: date(2010, 5, 28), 2011: date(2011, 5, 17), 2012: date(2012, 5, 6),
        2013: date(2013, 5, 25), 2014: date(2014, 5, 14), 2015: date(2015, 5, 4),
        2016: date(2016, 5, 22), 2017: date(2017, 5, 11), 2018: date(2018, 5, 30),
        2019: date(2019, 5, 19), 2020: date(2020, 5, 7),  2021: date(2021, 5, 26),
        2022: date(2022, 5, 15), 2023: date(2023, 5, 5),  2024: date(2024, 5, 23),
        2025: date(2025, 5, 13), 2026: date(2026, 5, 2),  2027: date(2027, 5, 21),
        2028: date(2028, 5, 10), 2029: date(2029, 5, 29), 2030: date(2030, 5, 18),
    }
    if year in vesak_dates:
        d = vesak_dates[year]
    else:
        # Fallback: second Monday of May
        d = date(year, 5, 14)
    # Return Monday of that week
    return d - pd.Timedelta(days=d.weekday())


def build_calendar(cfg: dict) -> pd.DataFrame:
    """Build Sri Lanka weekly calendar table (2010–2030)."""
    out_path = os.path.join(cfg["paths"]["raw_calendar"], "lka_calendar.csv")
    os.makedirs(cfg["paths"]["raw_calendar"], exist_ok=True)

    if os.path.exists(out_path):
        logger.info("[CALENDAR] Already exists — skipping: %s", out_path)
        return pd.read_csv(out_path, parse_dates=["week_start"])

    weeks = pd.date_range(start="2010-01-04", end="2030-12-29", freq="W-MON")
    df = pd.DataFrame({"week_start": weeks})
    df["week_of_year"] = df["week_start"].dt.isocalendar().week.astype(int)
    df["month"] = df["week_start"].dt.month

    # Monsoon flags
    def _sw_monsoon(d: pd.Timestamp) -> int:
        md = d.month * 100 + d.day
        return int(515 <= md <= 915)

    def _ne_monsoon(d: pd.Timestamp) -> int:
        md = d.month * 100 + d.day
        return int(md >= 1015 or md <= 115)

    def _dry_season(d: pd.Timestamp) -> int:
        md = d.month * 100 + d.day
        return int((200 <= md <= 514) or (916 <= md <= 1014))

    df["is_sw_monsoon"] = df["week_start"].apply(_sw_monsoon)
    df["is_ne_monsoon"] = df["week_start"].apply(_ne_monsoon)
    df["is_dry_season"] = df["week_start"].apply(_dry_season)

    # Sinhala/Tamil New Year: week containing April 13–14
    def _sinhala_new_year(d: pd.Timestamp) -> int:
        week_end = d + pd.Timedelta(days=6)
        for day in pd.date_range(d, week_end):
            if day.month == 4 and day.day in (13, 14):
                return 1
        return 0

    df["is_sinhala_tamil_new_year"] = df["week_start"].apply(_sinhala_new_year)

    # Vesak Poya
    vesak_mondays = {_vesak_week(y) for y in range(2010, 2031)}
    df["is_vesak"] = df["week_start"].apply(
        lambda d: int(d.date() in vesak_mondays)
    )

    # Christmas week: week containing Dec 25
    def _christmas(d: pd.Timestamp) -> int:
        week_end = d + pd.Timedelta(days=6)
        for day in pd.date_range(d, week_end):
            if day.month == 12 and day.day == 25:
                return 1
        return 0

    df["is_christmas_week"] = df["week_start"].apply(_christmas)

    # Year-end quarter: last 3 weeks of March (fiscal Q4 rush)
    def _year_end_q(d: pd.Timestamp) -> int:
        if d.month == 3 and d.day >= 10:
            return 1
        return 0

    df["is_year_end_quarter"] = df["week_start"].apply(_year_end_q)

    # Post-holiday lags
    holiday_flags = (
        df["is_sinhala_tamil_new_year"]
        | df["is_vesak"]
        | df["is_christmas_week"]
    )
    df["post_holiday_lag_1"] = holiday_flags.shift(1).fillna(0).astype(int)
    df["post_holiday_lag_2"] = holiday_flags.shift(2).fillna(0).astype(int)

    df.to_csv(out_path, index=False)
    logger.info("[CALENDAR] Calendar built: %d rows -> %s", len(df), out_path)
    return df
