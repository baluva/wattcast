"""Backtest walk-forward : on rejoue l'histoire mois par mois, sans jamais regarder le futur.

Pour chaque mois M :
  1. entraînement sur tout ce qui est connu la veille du 1er jour de M (météo observée) ;
  2. prévision de chaque jour de M avec la météo PRÉVUE LA VEILLE, comme en conditions réelles ;
  3. comparaison avec la prévision J-1 publiée par RTE pour les mêmes demi-heures.

Une variante « météo parfaite » (météo observée en test) est calculée à part : elle n'est
PAS comparable à RTE, mais elle chiffre ce que coûtent les erreurs de prévision météo.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime

import mlflow
import pandas as pd

from wattcast.config import CountryConfig
from wattcast.evaluate import metrics as m
from wattcast.features import baselines, build_frame
from wattcast.models.lgbm import NUM_ROUNDS, PARAMS, Forecaster
from wattcast.tracking import setup_mlflow
from wattcast.warehouse import load_mart, training_mask

log = logging.getLogger(__name__)

WARMUP_MONTHS = 6  # mois prédits avant backtest_start pour amorcer l'intervalle conformal
# Le duel affiché : modèle calibré contre RTE recalibré. Les versions brutes restent publiées.
MODELS = ["model_cal", "rte_cal", "model", "rte_d1", "naive_d7", "seasonal_d7_d14"]


def run_backtest(cfg: CountryConfig) -> dict:
    mart = load_mart(cfg)
    obs = build_frame(mart, "observed")
    fc = build_frame(mart, "forecast")
    keep = training_mask(obs, cfg)

    last_full_day = obs.loc[obs["consumption_mw"].notna(), "day"].max() - pd.Timedelta(days=1)
    start = pd.Timestamp(cfg.raw["backtest_start"]) - pd.DateOffset(months=WARMUP_MONTHS)
    months = pd.date_range(start, last_full_day, freq="MS")

    parts = []
    model = Forecaster()
    for month in months:
        train = obs[keep & (obs["day"] <= month - pd.Timedelta(days=2))]
        test_idx = fc.index[
            (fc["day"] >= month)
            & (fc["day"] < month + pd.DateOffset(months=1))
            & (fc["day"] <= last_full_day)
        ]
        model = Forecaster().fit(train)
        test = fc.loc[test_idx]
        part = pd.DataFrame(
            {
                "ts_utc": test["ts_utc"],
                "day": test["day"],
                "slot": test["slot"],
                "actual": test["consumption_mw"],
                "rte_d1": test["rte_forecast_d1_mw"],
                "model": model.predict_mw(test),
                "model_perfect_weather": model.predict_mw(obs.loc[test_idx]),
                "fold": month.strftime("%Y-%m"),
            }
        ).join(baselines(test))
        parts.append(part)
        log.info(
            "pli %s : %d lignes d'entraînement, MAPE brut %.2f %% (RTE brut %.2f %%)",
            month.strftime("%Y-%m"),
            len(train),
            m.mape(part["actual"], part["model"]),
            m.mape(part["actual"], part["rte_d1"]),
        )

    bt = pd.concat(parts, ignore_index=True)
    # Même recalibrage pour les deux concurrents, avec les seules erreurs connues à la coupure.
    bt["model_cal"] = m.online_bias_correction(bt, "model")
    bt["rte_cal"] = m.online_bias_correction(bt, "rte_d1")
    bt = bt.join(m.online_conformal(bt, "model_cal"))
    bt = bt[bt["day"] >= pd.Timestamp(cfg.raw["backtest_start"])].reset_index(drop=True)
    # Comparaison équitable : uniquement les demi-heures où RTE a publié une prévision.
    bt = bt[bt["rte_d1"].notna() & bt["actual"].notna()].reset_index(drop=True)

    out = cfg.paths.outputs
    out.mkdir(parents=True, exist_ok=True)
    bt.to_parquet(out / "backtest.parquet", index=False)
    (out / "feature_importance.json").write_text(
        json.dumps(model.feature_importance().round(4).to_dict(), indent=2), encoding="utf-8"
    )
    results = publish_scores(cfg)

    setup_mlflow(cfg)
    with mlflow.start_run(run_name=f"backtest-{cfg.country}"):
        mlflow.log_params({**PARAMS, "num_rounds": NUM_ROUNDS, "backtest_start": cfg.raw["backtest_start"]})
        mlflow.log_metrics({k: v for k, v in results["overall"].items() if isinstance(v, float)})
        mlflow.log_artifact(str(out / "backtest_metrics.json"))
    log.info("résumé : %s", json.dumps(results["overall"], indent=2))
    return results


def publish_scores(cfg: CountryConfig) -> dict:
    """Calcule les scores publiés à partir des prévisions du backtest.

    Les jours où la température « prévue la veille » a dû être comblée par une prévision plus
    fraîche (trous de l'archive Open-Meteo) sont exclus du score, pour les deux concurrents :
    le modèle y aurait un avantage que RTE n'avait pas.
    """
    out = cfg.paths.outputs
    bt = pd.read_parquet(out / "backtest.parquet")
    flags = load_mart(cfg)[["ts_utc", "temp_fc_filled"]]
    filled = bt[["ts_utc", "day"]].merge(flags, on="ts_utc", how="left")["temp_fc_filled"].fillna(False)
    tainted_days = set(bt.loc[filled.to_numpy(), "day"])
    scored = bt[~bt["day"].isin(tainted_days)].reset_index(drop=True)

    results = summarize(scored)
    results["overall"]["days_excluded_weather_gap"] = len(tainted_days)
    results["period"] = [str(scored["day"].min().date()), str(scored["day"].max().date())]
    results["half_hours"] = len(scored)
    fi = out / "feature_importance.json"
    results["feature_importance"] = json.loads(fi.read_text(encoding="utf-8")) if fi.exists() else {}
    results["generated_at"] = datetime.now().isoformat(timespec="seconds")

    scored.to_parquet(out / "backtest_scored.parquet", index=False)
    (out / "backtest_metrics.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    log.info("scores publiés (%d jours exclus) : %s", len(tainted_days), json.dumps(results["overall"]))
    return results


def summarize(bt: pd.DataFrame) -> dict:
    cols = [*MODELS, "model_perfect_weather"]
    daily = m.daily_errors(bt, cols)
    overall: dict = {}
    for c in cols:
        overall[f"mape_{c}"] = round(m.mape(bt["actual"], bt[c]), 3)
        overall[f"rmse_{c}"] = round(m.rmse(bt["actual"], bt[c]), 1)
    overall["days"] = int(len(daily))
    overall["days_model_beats_rte"] = int((daily["model_cal"] < daily["rte_cal"]).sum())
    overall["win_rate_vs_rte"] = round(float((daily["model_cal"] < daily["rte_cal"]).mean()), 3)
    overall["win_rate_vs_rte_raw"] = round(float((daily["model"] < daily["rte_d1"]).mean()), 3)
    overall["bias_rte_raw_pct"] = round(float(((bt["rte_d1"] - bt["actual"]) / bt["actual"]).mean() * 100), 2)
    inside = bt["lo"].notna() & (bt["actual"] >= bt["lo"]) & (bt["actual"] <= bt["hi"])
    overall["interval_coverage_80"] = round(float(inside[bt["lo"].notna()].mean()), 3)

    by_year = {}
    for year, g in bt.groupby(bt["day"].dt.year):
        d = daily[daily.index.year == year]
        by_year[int(year)] = {
            "mape_model": round(m.mape(g["actual"], g["model_cal"]), 3),
            "mape_rte": round(m.mape(g["actual"], g["rte_cal"]), 3),
            "mape_rte_raw": round(m.mape(g["actual"], g["rte_d1"]), 3),
            "mape_naive_d7": round(m.mape(g["actual"], g["naive_d7"]), 3),
            "win_rate_vs_rte": round(float((d["model_cal"] < d["rte_cal"]).mean()), 3),
        }
    return {"overall": overall, "by_year": by_year}
