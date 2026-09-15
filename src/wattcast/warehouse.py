"""Accès en lecture à l'entrepôt DuckDB construit par dbt."""

from __future__ import annotations

import duckdb
import pandas as pd

from wattcast.config import CountryConfig


def load_mart(cfg: CountryConfig) -> pd.DataFrame:
    with duckdb.connect(str(cfg.paths.warehouse), read_only=True) as con:
        df = con.execute("select * from mart_halfhourly order by ts_utc").df()
    # DuckDB renvoie le local sans fuseau (voulu) ; l'UTC reste explicite.
    df["ts_utc"] = pd.to_datetime(df["ts_utc"], utc=True)
    return df


def training_mask(frame: pd.DataFrame, cfg: CountryConfig) -> pd.Series:
    """Retire les périodes déclarées anormales dans la config (ex. confinement 2020)."""
    keep = pd.Series(True, index=frame.index)
    for period in cfg.raw.get("exclude_from_training", []):
        keep &= ~frame["day"].between(pd.Timestamp(period["start"]), pd.Timestamp(period["end"]))
    return keep
