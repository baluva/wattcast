"""Prévision live et suivi honnête.

- `predict_tomorrow` : lancé la veille vers midi. La prévision est ajoutée au journal
  `live_predictions.parquet` et n'est JAMAIS réécrite : relancer la commande ne permet pas
  de « corriger » une prévision après coup.
- `score_live` : complète le journal avec la conso réalisée et la prévision RTE, puis
  calcule les scores journaliers.
"""

from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd

from wattcast.config import CountryConfig
from wattcast.evaluate.metrics import COVERAGE, daily_errors, online_bias_correction
from wattcast.features import build_frame
from wattcast.ingest.weather import snapshot_live_forecast
from wattcast.models.registry import load_champion
from wattcast.warehouse import load_mart

log = logging.getLogger(__name__)

WEATHER_COLS = {
    "temperature_2m": "temp_fc_c",
    "cloud_cover": "cloud_fc_pct",
    "shortwave_radiation": "radiation_fc_wm2",
    "wind_speed_10m": "wind_fc_kmh",
}


def _national_halfhourly(snapshot: pd.DataFrame, cfg: CountryConfig) -> pd.DataFrame:
    weights = {p["name"]: p["weight"] for p in cfg.sources["weather"]["points"]}
    snap = snapshot.assign(w=snapshot["point"].map(weights))
    hourly = snap.groupby("time").apply(
        lambda g: pd.Series({col: np.average(g[src], weights=g["w"]) for src, col in WEATHER_COLS.items()}),
        include_groups=False,
    )
    # Heure → demi-heure par interpolation linéaire, comme dans dbt.
    return hourly.resample("30min").mean().interpolate(limit=1)


def _calibration(cfg: CountryConfig, target_day: pd.Timestamp) -> tuple[float, float]:
    """Biais récent et demi-largeur d'intervalle, avec les erreurs connues à la coupure.

    Même règle que dans le backtest : biais = médiane des erreurs relatives brutes des jours
    D-29 à D-2 ; intervalle = quantile 80 % des erreurs de la prévision calibrée, D-57 à D-2.
    Les erreurs live priment ; le backtest comble les jours où il n'y en a pas encore.
    """
    bt = pd.read_parquet(
        cfg.paths.outputs / "backtest.parquet", columns=["day", "actual", "model", "model_cal"]
    )
    err = bt.rename(columns={"model": "pred_raw", "model_cal": "cal"})
    live_path = cfg.paths.outputs / "live_predictions.parquet"
    if live_path.exists():
        live = pd.read_parquet(live_path).dropna(subset=["actual"])
        live = live.rename(columns={"target_day": "day", "pred": "cal"})[["day", "actual", "pred_raw", "cal"]]
        err = pd.concat([live, err[~err["day"].isin(live["day"])]], ignore_index=True)
    known = err["day"] <= target_day - pd.Timedelta(days=2)

    recent_bias = known & (err["day"] >= target_day - pd.Timedelta(days=29))
    bias = float(((err["pred_raw"] - err["actual"]) / err["actual"])[recent_bias].median())
    recent_q = known & (err["day"] >= target_day - pd.Timedelta(days=57))
    half = float(
        np.quantile(((err["cal"] - err["actual"]).abs() / err["actual"])[recent_q].dropna(), COVERAGE)
    )
    return (0.0 if np.isnan(bias) else bias), half


def predict_tomorrow(cfg: CountryConfig, now: pd.Timestamp | None = None) -> pd.DataFrame | None:
    now = now or pd.Timestamp.now(tz=cfg.timezone)
    target_day = (now.normalize() + pd.Timedelta(days=1)).tz_localize(None)
    log_path = cfg.paths.outputs / "live_predictions.parquet"
    journal = pd.read_parquet(log_path) if log_path.exists() else None
    if journal is not None and (journal["target_day"] == target_day).any():
        log.info("prévision du %s déjà émise : on ne la réécrit pas", target_day.date())
        return None

    mart = load_mart(cfg)
    weather = _national_halfhourly(snapshot_live_forecast(cfg, now.tz_convert("UTC")), cfg)
    mart = mart.set_index("ts_utc")
    overlap = mart.index.intersection(weather.index)
    for col in WEATHER_COLS.values():
        # La prévision du jour remplace la météo « prévue la veille » pour J-1 et J.
        mart.loc[overlap, col] = weather.loc[overlap, col]
    mart = mart.reset_index()

    frame = build_frame(mart, "forecast")
    rows = frame[frame["day"] == target_day]
    if len(rows) < 46:
        raise RuntimeError(f"Grille incomplète pour le {target_day.date()} ({len(rows)} créneaux)")
    missing = rows[["level", "lag_d2_r", "morning_d1_r", "temp"]].isna().mean()
    if missing.any():
        log.warning("variables manquantes (part de NaN) : %s", missing[missing > 0].to_dict())

    model, meta = load_champion(cfg)
    pred_raw = model.predict_mw(rows)
    bias, half = _calibration(cfg, target_day)
    pred = pred_raw / (1 + bias)
    out = pd.DataFrame(
        {
            "issued_at": now.tz_convert("UTC"),
            "target_day": target_day,
            "ts_utc": rows["ts_utc"].to_numpy(),
            "slot": rows["slot"].to_numpy(),
            "pred_raw": pred_raw,
            "bias_applied": bias,
            "pred": pred,
            "lo": pred * (1 - half),
            "hi": pred * (1 + half),
            "model_version": meta["version"],
            "actual": np.nan,
            "rte_d1": rows["rte_forecast_d1_mw"].to_numpy(),
        }
    )
    journal = out if journal is None else pd.concat([journal, out], ignore_index=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    journal.to_parquet(log_path, index=False)
    log.info(
        "prévision du %s émise (modèle %s) : pointe %.0f MW à %s",
        target_day.date(),
        meta["version"],
        pred.max(),
        rows["ts_local"].iloc[int(pred.argmax())],
    )
    return out


def score_live(cfg: CountryConfig) -> pd.DataFrame | None:
    log_path = cfg.paths.outputs / "live_predictions.parquet"
    if not log_path.exists():
        log.info("aucune prévision live à noter")
        return None
    journal = pd.read_parquet(log_path)
    if "actual_source" not in journal:
        journal["actual_source"] = pd.Series(pd.NA, index=journal.index, dtype="string")
    mart = load_mart(cfg).set_index("ts_utc")[["consumption_mw", "rte_forecast_d1_mw", "consumption_source"]]
    found = mart.reindex(journal["ts_utc"]).set_axis(journal.index)
    # La conso réelle est FIGÉE à la première notation (définition temps réel, celle que vise RTE) :
    # la version consolidée publiée des mois plus tard ne vient pas réécrire le score.
    todo = journal["actual"].isna() & found["consumption_mw"].notna()
    journal.loc[todo, "actual"] = found.loc[todo, "consumption_mw"]
    journal.loc[todo, "actual_source"] = found.loc[todo, "consumption_source"]
    journal["rte_d1"] = journal["rte_d1"].fillna(found["rte_forecast_d1_mw"])
    journal.to_parquet(log_path, index=False)

    scored = journal.rename(columns={"target_day": "day"})
    scored["rte_cal"] = online_bias_correction(scored, "rte_d1")
    daily = daily_errors(scored.rename(columns={"pred": "model"}), ["model", "rte_cal", "rte_d1"])
    if daily.empty:
        log.info("aucun jour complet à noter pour l'instant")
        return daily
    inside = (scored["actual"] >= scored["lo"]) & (scored["actual"] <= scored["hi"])
    daily["coverage"] = inside[scored["actual"].notna()].groupby(scored["day"]).mean()
    daily["model_beats_rte"] = daily["model"] < daily["rte_cal"]
    daily = daily.reset_index().rename(
        columns={"model": "mape_model", "rte_cal": "mape_rte", "rte_d1": "mape_rte_raw"}
    )
    daily.to_parquet(cfg.paths.outputs / "live_scores.parquet", index=False)

    summary = {
        "days": int(len(daily)),
        "mape_model": round(float(daily["mape_model"].mean()), 3),
        "mape_rte": round(float(daily["mape_rte"].mean()), 3),
        "mape_rte_raw": round(float(daily["mape_rte_raw"].mean()), 3),
        "win_rate_vs_rte": round(float(daily["model_beats_rte"].mean()), 3),
        "first_day": str(daily["day"].min().date()),
        "last_day": str(daily["day"].max().date()),
    }
    (cfg.paths.outputs / "live_metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log.info("suivi live : %s", summary)
    return daily
