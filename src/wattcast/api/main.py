"""API de lecture : sert les prévisions émises et les scores, sans recalcul à la volée.

Le calcul se fait dans le pipeline quotidien ; l'API ne fait que lire les sorties publiées.
Elle reste donc rapide, sans état, et ne peut pas servir une prévision « refaite après coup ».
"""

from __future__ import annotations

import json
from datetime import date
from typing import Annotated

import pandas as pd
from fastapi import FastAPI, HTTPException, Path, Query
from pydantic import BaseModel, Field

from wattcast.config import CONFIG_DIR, load_config

app = FastAPI(
    title="WattCast",
    version="1.0.0",
    description="Prévision J+1 de la consommation électrique, comparée à la prévision officielle de RTE.",
)

COUNTRIES = sorted(p.stem for p in CONFIG_DIR.glob("*.yaml"))
Country = Annotated[str, Path(description="Pays configuré", examples=["france"])]


class Slot(BaseModel):
    ts_utc: str
    pred_mw: float
    lo_mw: float = Field(description="Borne basse de l'intervalle à 80 %")
    hi_mw: float
    rte_d1_mw: float | None
    actual_mw: float | None


class Forecast(BaseModel):
    country: str
    target_day: date
    issued_at: str
    model_version: str
    slots: list[Slot]


def _cfg(country: str):
    if country not in COUNTRIES:
        raise HTTPException(404, f"Pays inconnu : {country}. Disponibles : {COUNTRIES}")
    return load_config(country)


def _read_json(path) -> dict | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _nan_to_none(x: float) -> float | None:
    return None if pd.isna(x) else float(x)


@app.get("/health")
def health() -> dict:
    status = {}
    for country in COUNTRIES:
        cfg = load_config(country)
        meta = _read_json(cfg.paths.models / "champion" / "meta.json")
        log = cfg.paths.outputs / "live_predictions.parquet"
        last = pd.read_parquet(log, columns=["target_day"])["target_day"].max() if log.exists() else None
        status[country] = {
            "model_version": meta and meta["version"],
            "last_forecast_day": None if last is None else str(last.date()),
        }
    ok = all(s["model_version"] for s in status.values())
    return {"status": "ok" if ok else "degraded", "countries": status}


@app.get("/forecast/{country}", response_model=Forecast)
def forecast(
    country: Country,
    day: Annotated[date | None, Query(description="Jour cible, défaut : le dernier émis")] = None,
):
    cfg = _cfg(country)
    path = cfg.paths.outputs / "live_predictions.parquet"
    if not path.exists():
        raise HTTPException(404, "Aucune prévision émise pour l'instant")
    log = pd.read_parquet(path)
    target = pd.Timestamp(day) if day else log["target_day"].max()
    rows = log[log["target_day"] == target].sort_values("ts_utc")
    if rows.empty:
        raise HTTPException(404, f"Pas de prévision pour le {target.date()}")
    return Forecast(
        country=country,
        target_day=target.date(),
        issued_at=str(rows["issued_at"].iloc[0]),
        model_version=str(rows["model_version"].iloc[0]),
        slots=[
            Slot(
                ts_utc=r.ts_utc.isoformat(),
                pred_mw=round(r.pred, 1),
                lo_mw=round(r.lo, 1),
                hi_mw=round(r.hi, 1),
                rte_d1_mw=_nan_to_none(r.rte_d1),
                actual_mw=_nan_to_none(r.actual),
            )
            for r in rows.itertuples()
        ],
    )


@app.get("/metrics/{country}")
def metrics(country: Country) -> dict:
    """Scores publiés : backtest (historique rejoué) et live (depuis la mise en production)."""
    out = _cfg(country).paths.outputs
    return {
        "backtest": _read_json(out / "backtest_metrics.json"),
        "live": _read_json(out / "live_metrics.json"),
        "drift": _read_json(out / "drift.json"),
    }
