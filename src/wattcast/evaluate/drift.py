"""Surveillance du drift, pensée pour une série saisonnière.

Comparer septembre à juillet signalerait un « drift » de température à coup sûr : c'est
juste la saison. La référence est donc la MÊME fenêtre de 28 jours un an plus tôt.

On travaille à la maille journalière : avec 1 300 demi-heures par fenêtre, un test de
Kolmogorov-Smirnov détecte des écarts minuscules et sonne en permanence.

Deux signaux, combinés dans `retrain_recommended` :
  1. drift des données (Evidently, K-S) sur la cible normalisée et les variables clés ;
  2. dégradation de performance : MAPE live récent vs MAPE du backtest à la même saison.
"""

from __future__ import annotations

import json
import logging
import warnings

import pandas as pd

from wattcast.config import CountryConfig
from wattcast.features import build_frame
from wattcast.warehouse import load_mart

log = logging.getLogger(__name__)

WINDOW_DAYS = 28
DRIFT_SHARE_ALERT = 0.5
PERF_DEGRADATION_ALERT = 1.25
MONITORED = ["y_ratio", "temp_daymean", "lag_d7_r", "morning_trend", "hdd", "cdd"]


def _daily(frame: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    rows = frame[frame["day"].between(start, end)]
    return rows.groupby("day")[MONITORED].mean().dropna()


def _window_mape(df: pd.DataFrame, day_col: str, pred_col: str, start, end) -> float | None:
    w = df[df[day_col].between(start, end) & df["actual"].notna()]
    if w[day_col].nunique() < WINDOW_DAYS // 2:
        return None
    return float(((w[pred_col] - w["actual"]).abs() / w["actual"]).mean() * 100)


def drift_report(cfg: CountryConfig) -> dict:
    from evidently import Report
    from evidently.presets import DataDriftPreset

    frame = build_frame(load_mart(cfg), "observed")
    last_full_day = frame.loc[frame["consumption_mw"].notna(), "day"].max() - pd.Timedelta(days=1)
    cur_start = last_full_day - pd.Timedelta(days=WINDOW_DAYS - 1)
    ref_start, ref_end = cur_start - pd.DateOffset(years=1), last_full_day - pd.DateOffset(years=1)

    current, reference = _daily(frame, cur_start, last_full_day), _daily(frame, ref_start, ref_end)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        snapshot = Report([DataDriftPreset()]).run(current, reference)
    out = cfg.paths.outputs
    out.mkdir(parents=True, exist_ok=True)
    snapshot.save_html(str(out / "drift_report.html"))

    metrics = snapshot.dict()["metrics"]
    share = next(
        mt["value"]["share"] for mt in metrics if mt["metric_name"].startswith("DriftedColumnsCount")
    )
    p_values = {
        mt["config"]["column"]: round(float(mt["value"]), 4)
        for mt in metrics
        if mt["metric_name"].startswith("ValueDrift")
    }

    # Performance : live récent vs backtest à la même saison l'an dernier.
    mape_ref = mape_cur = None
    bt_path, live_path = out / "backtest.parquet", out / "live_predictions.parquet"
    if bt_path.exists():
        bt = pd.read_parquet(bt_path, columns=["day", "model", "actual"])
        mape_ref = _window_mape(bt, "day", "model", ref_start, ref_end)
    if live_path.exists():
        live = pd.read_parquet(live_path)
        mape_cur = _window_mape(live, "target_day", "pred", cur_start, last_full_day)

    degraded = mape_ref is not None and mape_cur is not None and mape_cur > PERF_DEGRADATION_ALERT * mape_ref
    result = {
        "current_window": [str(cur_start.date()), str(last_full_day.date())],
        "reference_window": [str(ref_start.date()), str(ref_end.date())],
        "drift_share": round(float(share), 3),
        "p_values": p_values,
        "mape_live_current": None if mape_cur is None else round(mape_cur, 3),
        "mape_backtest_same_season": None if mape_ref is None else round(mape_ref, 3),
        "performance_degraded": bool(degraded),
        "retrain_recommended": bool(share >= DRIFT_SHARE_ALERT or degraded),
    }
    (out / "drift.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    log.info("drift : %s", result)
    return result
