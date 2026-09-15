"""Métriques et intervalles de prédiction."""

from __future__ import annotations

import numpy as np
import pandas as pd

COVERAGE = 0.80  # intervalle à 80 % : 1 demi-heure sur 5 doit tomber dehors, pas plus


def mape(actual: pd.Series, pred: pd.Series) -> float:
    ok = actual.notna() & pred.notna()
    return float((np.abs(pred[ok] - actual[ok]) / actual[ok]).mean() * 100)


def rmse(actual: pd.Series, pred: pd.Series) -> float:
    ok = actual.notna() & pred.notna()
    return float(np.sqrt(((pred[ok] - actual[ok]) ** 2).mean()))


def daily_errors(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """MAPE journalier par modèle (jours complets uniquement)."""
    rows = df.dropna(subset=["actual", *columns])
    rel = rows[columns].sub(rows["actual"], axis=0).abs().div(rows["actual"], axis=0) * 100
    rel["day"] = rows["day"]
    out = rel.groupby("day").agg(["mean", "size"])
    full = out[(columns[0], "size")] >= 46  # 46 à 50 créneaux selon les changements d'heure
    return out.loc[full, [(c, "mean") for c in columns]].droplevel(1, axis=1)


def _rolling_past(df: pd.DataFrame, values: pd.Series, window_days: int, stat) -> pd.Series:
    """Pour chaque jour D, `stat` des valeurs des jours D-window-1 … D-2.

    D-2 est le dernier jour entièrement connu à la coupure de D-1 midi : aucune valeur
    du jour prédit, ni de la veille, n'entre dans le calcul. Il faut au moins 14 jours d'historique.
    """
    per_day = {d: v.dropna().to_numpy() for d, v in values.groupby(df["day"])}
    days = sorted(per_day)
    out = {}
    for d in days:
        past = [
            per_day[p]
            for p in days
            if d - pd.Timedelta(days=window_days + 1) <= p <= d - pd.Timedelta(days=2)
        ]
        out[d] = stat(np.concatenate(past)) if len(past) >= 14 else np.nan
    return df["day"].map(out)


def online_bias_correction(df: pd.DataFrame, pred_col: str, window_days: int = 28) -> pd.Series:
    """Recalibrage en ligne : on retire à la prévision son biais relatif médian récent.

    Utile quand la « vérité » n'a pas la même définition que la cible du prévisionniste :
    RTE prévoit la conso au sens temps réel, alors que la conso consolidée publiée des mois
    plus tard est révisée de +2 à +3 %. Appliqué de la même façon à tous les concurrents.
    """
    rel = (df[pred_col] - df["actual"]) / df["actual"]
    bias = _rolling_past(df, rel, window_days, np.median).fillna(0.0)
    return df[pred_col] / (1 + bias)


def online_conformal(df: pd.DataFrame, pred_col: str, window_days: int = 56) -> pd.DataFrame:
    """Intervalle conformal « en ligne », sans fuite de données.

    Pour le jour D, la demi-largeur est le quantile 80 % des erreurs relatives absolues
    des `window_days` jours précédents (jusqu'à D-2), donc des erreurs réellement hors
    échantillon et connues à la date de la prévision. Aucune hypothèse de loi sur les résidus.
    """
    rel_err = (df[pred_col] - df["actual"]).abs() / df["actual"]
    half = _rolling_past(df, rel_err, window_days, lambda a: np.quantile(a, COVERAGE))
    return pd.DataFrame({"lo": df[pred_col] * (1 - half), "hi": df[pred_col] * (1 + half)}, index=df.index)
