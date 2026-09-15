"""Calendrier : jours fériés, ponts, vacances scolaires (nombre de zones A/B/C en congés)."""

from __future__ import annotations

import logging
from datetime import date, timedelta

import holidays
import pandas as pd

from wattcast.config import CountryConfig
from wattcast.ingest._http import get, write_parquet

log = logging.getLogger(__name__)

SCHOOL_ZONES = {"Zone A", "Zone B", "Zone C"}


def _school_holidays(cfg: CountryConfig) -> pd.DataFrame:
    rows = get(cfg.sources["school_holidays"]["url"]).json()
    df = pd.DataFrame(rows)
    df = df[df["zones"].isin(SCHOOL_ZONES) & df["population"].isin(["-", "Élèves"])]
    tz = cfg.timezone
    # start_date = minuit local du 1er jour de congés ; end_date = minuit local du jour de reprise.
    df["start"] = pd.to_datetime(df["start_date"], utc=True).dt.tz_convert(tz).dt.date
    df["end"] = pd.to_datetime(df["end_date"], utc=True).dt.tz_convert(tz).dt.date
    return df.drop_duplicates(["zones", "start", "end"])[["zones", "description", "start", "end"]]


def build_calendar(cfg: CountryConfig) -> pd.DataFrame:
    start = date.fromisoformat(cfg.raw["history_start"])
    end = date(date.today().year + 1, 12, 31)
    days = pd.date_range(start, end, freq="D").date

    fr = holidays.country_holidays(cfg.raw["holidays_code"], years=range(start.year, end.year + 1))
    cal = pd.DataFrame({"day": days})
    cal["holiday_name"] = [fr.get(d) for d in days]
    cal["is_holiday"] = cal["holiday_name"].notna()

    # Pont : jour ouvré coincé entre un férié et un week-end (ex. vendredi après l'Ascension).
    def off(d: date) -> bool:
        return d in fr or d.weekday() >= 5

    def is_bridge(d: date) -> bool:
        before, after = d - timedelta(1), d + timedelta(1)
        return not off(d) and off(before) and off(after) and (before in fr or after in fr)

    cal["is_bridge"] = [is_bridge(d) for d in days]

    school = _school_holidays(cfg)
    zones_off = pd.Series(0, index=pd.Index(days))
    for row in school.itertuples():
        mask = (zones_off.index >= row.start) & (zones_off.index < row.end)
        zones_off[mask] += 1
    cal["school_zones_off"] = zones_off.clip(upper=3).to_numpy()

    write_parquet(cal, cfg.paths.raw / "calendar.parquet")
    log.info(
        "calendrier : %d jours, %d fériés, %d ponts",
        len(cal),
        cal["is_holiday"].sum(),
        cal["is_bridge"].sum(),
    )
    return cal
