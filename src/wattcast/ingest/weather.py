"""Ingestion météo Open-Meteo, en trois « vues » qui ne doivent jamais être mélangées :

- observed     : météo réalisée (réanalyse). Sert à l'entraînement et à l'analyse.
- forecast_d1  : météo telle qu'elle était PRÉVUE AVANT LA COUPURE de la veille à midi
                 (API Previous Runs : `*_previous_day1` le matin, `*_previous_day2`
                 l'après-midi), complétée par l'API Historical Forecast là où il manque des
                 valeurs. Sert au backtest : c'est ce qu'on aurait eu sous les yeux.
- forecast_live: instantané de la prévision du jour, pris au moment où l'on prédit.
                 Conservé tel quel pour que le suivi live soit auditable.
"""

from __future__ import annotations

import logging
import time
from datetime import date

import pandas as pd

from wattcast.config import CountryConfig
from wattcast.ingest._http import get, write_parquet

log = logging.getLogger(__name__)

ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"
PREVIOUS_RUNS = "https://previous-runs-api.open-meteo.com/v1/forecast"
HISTORICAL_FORECAST = "https://historical-forecast-api.open-meteo.com/v1/forecast"
FORECAST = "https://api.open-meteo.com/v1/forecast"

FORECAST_D1_START = 2021  # début de couverture de l'API Previous Runs (température)
PAUSE_S = 3.0


def _fetch(url: str, cfg: CountryConfig, hourly: list[str], **params: str | int) -> pd.DataFrame:
    """Un appel multi-points ; renvoie un format long (time, point, variables...)."""
    points = cfg.sources["weather"]["points"]
    query = {
        "latitude": ",".join(str(p["lat"]) for p in points),
        "longitude": ",".join(str(p["lon"]) for p in points),
        "hourly": ",".join(hourly),
        "timezone": "GMT",
        **params,
    }
    for attempt in range(1, 4):
        try:
            payload = get(url, query).json()
            time.sleep(PAUSE_S)  # rester sous la limite de débit gratuite d'Open-Meteo
            break
        except ValueError:  # réponse tronquée en cours de transfert : on redemande
            log.warning("JSON illisible sur %s (essai %d/3)", url, attempt)
            time.sleep(5 * attempt)
    else:
        raise RuntimeError(f"Réponse illisible après 3 essais : {url}")
    if isinstance(payload, dict):  # un seul point → l'API ne renvoie pas de liste
        payload = [payload]
    frames = []
    for point, item in zip(points, payload, strict=True):
        df = pd.DataFrame(item["hourly"])
        df["time"] = pd.to_datetime(df["time"], utc=True)
        df.insert(1, "point", point["name"])
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def _years(start: int, end: int) -> list[int]:
    return list(range(start, end + 1))


def _fetch_year(url: str, cfg: CountryConfig, hourly: list[str], year: int) -> pd.DataFrame:
    """Une année en deux semestres : les réponses d'un an entier arrivent parfois tronquées."""
    today = date.today()
    halves = [(date(year, 1, 1), date(year, 6, 30)), (date(year, 7, 1), date(year, 12, 31))]
    frames = [
        _fetch(url, cfg, hourly, start_date=start.isoformat(), end_date=min(end, today).isoformat())
        for start, end in halves
        if start <= today
    ]
    return pd.concat(frames, ignore_index=True)


def write_points(cfg: CountryConfig) -> None:
    """Les poids viennent de la config ; dbt les lit ici pour agréger la météo nationale."""
    write_parquet(
        pd.DataFrame(cfg.sources["weather"]["points"]), cfg.paths.raw / "weather" / "points.parquet"
    )


def ingest_observed(cfg: CountryConfig, *, force: bool = False) -> int:
    write_points(cfg)
    variables = cfg.sources["weather"]["variables"]
    today = date.today()
    n = 0
    for year in _years(int(cfg.raw["history_start"][:4]), today.year):
        path = cfg.paths.raw / "weather" / "observed" / f"{year}.parquet"
        # Les années closes ne changent plus ; l'année en cours (et la précédente en janvier) si.
        closed = year < today.year - (1 if today.month == 1 else 0)
        if path.exists() and closed and not force:
            continue
        df = _fetch_year(ARCHIVE, cfg, variables, year)
        write_parquet(df, path)
        n += len(df)
        log.info("météo observée %d : %d lignes", year, len(df))
    return n


def ingest_forecast_d1(cfg: CountryConfig, *, force: bool = False) -> int:
    all_variables = cfg.sources["weather"]["variables"]
    variables = ["temperature_2m"]  # seule variable avec une archive J-1 fiable depuis 2021
    today = date.today()
    n = 0
    for year in _years(FORECAST_D1_START, today.year):
        path = cfg.paths.raw / "weather" / "forecast_d1" / f"{year}.parquet"
        closed = year < today.year - (1 if today.month == 1 else 0)
        if path.exists() and closed and not force:
            continue
        lagged = [f"{v}_previous_day{k}" for v in variables for k in (1, 2)]
        prev = _fetch_year(PREVIOUS_RUNS, cfg, lagged, year)
        # `previous_day1` d'une heure h = prévision émise la veille à l'heure h. Pour les heures
        # de l'après-midi, elle est émise APRÈS la coupure de midi : on prend alors `previous_day2`.
        morning = prev["time"].dt.tz_convert(cfg.timezone).dt.hour < 12
        for v in variables:
            prev[v] = prev[f"{v}_previous_day1"].where(morning, prev[f"{v}_previous_day2"])
        prev = prev.drop(columns=lagged)
        hist = _fetch_year(HISTORICAL_FORECAST, cfg, variables, year)

        merged = prev.merge(hist, on=["time", "point"], how="outer", suffixes=("", "_hist"))
        # Seule la température sert au modèle : c'est elle qui décide si une ligne est « comblée ».
        # Les autres variables n'ont pas d'archive J-1 avant 2024 ; on les garde brutes, sans les combler.
        merged["filled_from_historical"] = merged["temperature_2m"].isna()
        merged["temperature_2m"] = merged["temperature_2m"].fillna(merged["temperature_2m_hist"])
        merged = merged.drop(columns=[f"{v}_hist" for v in variables])
        for v in all_variables:  # même schéma que la vue observée
            merged[v] = merged[v] if v in merged else float("nan")
        write_parquet(merged, path)
        n += len(merged)
        log.info(
            "météo prévue J-1 %d : %d lignes (%d comblées par Historical Forecast)",
            year,
            len(merged),
            int(merged["filled_from_historical"].sum()),
        )
    return n


def snapshot_live_forecast(cfg: CountryConfig, issued: pd.Timestamp) -> pd.DataFrame:
    """Instantané de la prévision courante (hier, aujourd'hui et 2 jours devant)."""
    variables = cfg.sources["weather"]["variables"]
    df = _fetch(FORECAST, cfg, variables, past_days=2, forecast_days=3)
    df["issued_at"] = issued
    path = cfg.paths.raw / "weather" / "forecast_live" / f"{issued:%Y-%m-%dT%H%M}.parquet"
    write_parquet(df, path)
    log.info("instantané météo live %s : %d lignes", issued, len(df))
    return df
