"""Registre minimal champion / challenger, sur disque.

Règle de promotion (volontairement simple et défendable) :
  - holdout = les 8 dernières semaines complètes ;
  - le challenger est entraîné sur tout ce qui précède, puis prédit le holdout avec la
    météo prévue la veille (conditions réelles) ;
  - il remplace le champion seulement s'il fait mieux que le champion SUR CE MÊME holdout.
    Le champion est jugé sur ses vraies prévisions live si on en a assez, sinon sur une
    réplique réentraînée avec la même coupure (sa version en production a vu le holdout).
  - une fois promu, le challenger est réentraîné sur toutes les données disponibles.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime

import mlflow
import pandas as pd

from wattcast.config import CountryConfig
from wattcast.evaluate.metrics import mape
from wattcast.features import build_frame
from wattcast.models.lgbm import NUM_ROUNDS, PARAMS, Forecaster
from wattcast.tracking import setup_mlflow
from wattcast.warehouse import load_mart, training_mask

log = logging.getLogger(__name__)

HOLDOUT_DAYS = 56
MIN_LIVE_DAYS = 28


def champion_dir(cfg: CountryConfig):
    return cfg.paths.models / "champion"


def load_champion(cfg: CountryConfig) -> tuple[Forecaster, dict]:
    d = champion_dir(cfg)
    meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    return Forecaster.load(d / "model.txt"), meta


def _champion_live_mape(cfg: CountryConfig, days: pd.DatetimeIndex) -> float | None:
    path = cfg.paths.outputs / "live_predictions.parquet"
    if not path.exists():
        return None
    live = pd.read_parquet(path)
    live = (
        live[live["target_day"].isin(days) & live["actual"].notna()] if "actual" in live else live.iloc[0:0]
    )
    if live["target_day"].nunique() < MIN_LIVE_DAYS:
        return None
    return mape(live["actual"], live["pred"])


def train_and_maybe_promote(cfg: CountryConfig, *, force_promote: bool = False) -> dict:
    mart = load_mart(cfg)
    obs = build_frame(mart, "observed")
    fc = build_frame(mart, "forecast")
    keep = training_mask(obs, cfg)

    last_full_day = obs.loc[obs["consumption_mw"].notna(), "day"].max() - pd.Timedelta(days=1)
    holdout_start = last_full_day - pd.Timedelta(days=HOLDOUT_DAYS - 1)
    holdout_days = pd.date_range(holdout_start, last_full_day, freq="D")
    in_holdout = fc["day"].between(holdout_start, last_full_day) & fc["consumption_mw"].notna()
    pre_holdout = keep & (obs["day"] <= holdout_start - pd.Timedelta(days=2))

    challenger = Forecaster().fit(obs[pre_holdout])
    hold = fc[in_holdout]
    challenger_mape = mape(hold["consumption_mw"], pd.Series(challenger.predict_mw(hold), index=hold.index))
    rte_mape = mape(hold["consumption_mw"], hold["rte_forecast_d1_mw"])

    has_champion = (champion_dir(cfg) / "model.txt").exists()
    champion_mape, champion_basis = None, None
    if has_champion:
        _, meta = load_champion(cfg)
        champion_mape = _champion_live_mape(cfg, holdout_days)
        champion_basis = "live"
        if champion_mape is None:
            # Réplique : mêmes hyperparamètres que le champion, même coupure que le challenger.
            replica = Forecaster(params=meta["params"], num_rounds=meta["num_rounds"]).fit(obs[pre_holdout])
            champion_mape = mape(
                hold["consumption_mw"], pd.Series(replica.predict_mw(hold), index=hold.index)
            )
            champion_basis = "replica"

    promote = force_promote or not has_champion or challenger_mape < champion_mape
    decision = {
        "decided_at": datetime.now().isoformat(timespec="seconds"),
        "holdout": [holdout_start.date().isoformat(), last_full_day.date().isoformat()],
        "challenger_mape": round(challenger_mape, 3),
        "champion_mape": None if champion_mape is None else round(champion_mape, 3),
        "champion_basis": champion_basis,
        "rte_mape": round(rte_mape, 3),
        "promoted": bool(promote),
    }
    log.info("décision : %s", decision)

    if promote:
        final = Forecaster().fit(obs[keep & (obs["day"] <= last_full_day)])
        d = champion_dir(cfg)
        final.save(d / "model.txt")
        meta = {
            "version": datetime.now().strftime("%Y%m%d-%H%M"),
            "trained_until": last_full_day.date().isoformat(),
            "params": PARAMS,
            "num_rounds": NUM_ROUNDS,
            "holdout_mape": decision["challenger_mape"],
            "holdout_rte_mape": decision["rte_mape"],
        }
        (d / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        decision["version"] = meta["version"]

    history = cfg.paths.outputs / "promotions.jsonl"
    history.parent.mkdir(parents=True, exist_ok=True)
    with history.open("a", encoding="utf-8") as f:
        f.write(json.dumps(decision) + "\n")

    setup_mlflow(cfg)
    with mlflow.start_run(run_name=f"train-{cfg.country}"):
        mlflow.log_params({**PARAMS, "num_rounds": NUM_ROUNDS, "holdout_days": HOLDOUT_DAYS})
        mlflow.log_metrics(
            {
                k: float(v)
                for k, v in decision.items()
                if isinstance(v, (int, float)) and not isinstance(v, bool)
            }
        )
        mlflow.set_tag("promoted", str(promote))
    return decision
