"""Variables de prévision, construites selon le protocole « la veille à midi ».

Règle d'or : pour prédire le jour D, une variable ne peut utiliser que ce qui est
connu à D-1 12:00 heure locale. Concrètement :
- conso : jours D-2 et avant en entier, plus la matinée de D-1 (avant 11:30) ;
- météo de D et de D-1 : PRÉVUE (on ne connaît pas encore le temps qu'il fera) ;
- météo de D-2 et avant : observée.

Le niveau de consommation dérive dans le temps (sobriété 2022-2023, électrification...).
Un modèle à arbres n'extrapole pas : on lui fait prédire un RATIO par rapport au niveau
récent (moyenne des 7 derniers jours connus), puis on remultiplie.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

WeatherView = Literal["observed", "forecast"]

MORNING_SLOTS = 23  # jusqu'à 11:00 inclus : ce qui est publié avant la coupure de midi
MORNING_START = 12  # à partir de 06:00 : la matinée active, loin des changements d'heure
DST_SLOTS = (4, 5)  # 02:00 et 02:30, heure locale

FEATURES = [
    "slot",
    "dow",
    "month",
    "doy_sin",
    "doy_cos",
    "is_weekend",
    "is_holiday",
    "is_bridge",
    "school_zones_off",
    "holiday_eve",
    "holiday_after",
    "xmas_period",
    "lag_d2_r",
    "lag_d7_r",
    "lag_d14_r",
    "daymean_d2_r",
    "morning_d1_r",
    "morning_trend",
    "temp",
    "temp_daymean",
    "temp_daymin",
    "temp_daymax",
    "hdd",
    "cdd",
    "temp_d1_mean",
    "temp_d2_mean",
    "temp_minus_d7",
    "temp_minus_level",
    # Pas de nébulosité, rayonnement ni vent : leur prévision « faite la veille » n'est archivée
    # qu'à partir de 2024. Les combler avec des prévisions plus fraîches serait une fuite.
]
TARGET = "y_ratio"


def _slot(ts_local: pd.Series) -> pd.Series:
    return ts_local.dt.hour * 2 + ts_local.dt.minute // 30


def build_frame(mart: pd.DataFrame, weather: WeatherView) -> pd.DataFrame:
    """Une ligne par demi-heure de l'historique, avec variables et cible.

    `weather` choisit la météo du jour prédit : observée (entraînement) ou prévue la veille
    (backtest et live). Les jours passés utilisent toujours l'observé, à défaut le prévu.
    """
    df = mart.copy()
    df["ts_local"] = pd.to_datetime(df["ts_local"])
    df["day"] = pd.to_datetime(df["day_local"])
    df["slot"] = _slot(df["ts_local"])

    # --- Tables par (jour, créneau) et par jour, pour les décalages -------------------------
    # groupby (jour, créneau) : lors du passage à l'heure d'hiver, un créneau apparaît deux fois.
    by_slot = df.groupby(["day", "slot"])["consumption_mw"].mean()
    # Moyennes journalières hors 02:00-03:00, la tranche que les changements d'heure suppriment
    # ou doublent : sinon un jour de 23 h (sans son creux de nuit) paraît plus « chargé ».
    stable = by_slot[~by_slot.index.get_level_values("slot").isin(DST_SLOTS)]
    daily = stable.groupby(level="day").mean().to_frame("cons_mean")
    slots = stable.index.get_level_values("slot")
    morning = stable[(slots >= MORNING_START) & (slots < MORNING_SLOTS)].groupby(level="day").mean()

    past_temp = df["temp_obs_c"].fillna(df["temp_fc_c"])
    temp_past_daily = past_temp.groupby(df["day"]).mean()
    temp_past_slot = past_temp.groupby([df["day"], df["slot"]]).mean()

    def shift_days(n: int) -> pd.Series:
        return df["day"] - pd.Timedelta(days=n)

    def lookup(table: pd.Series, n: int, with_slot: bool) -> np.ndarray:
        keys = (
            pd.MultiIndex.from_arrays([shift_days(n), df["slot"]]) if with_slot else pd.Index(shift_days(n))
        )
        return table.reindex(keys).to_numpy()

    # Niveau = moyenne des jours D-8 à D-2 (les 7 derniers jours complets connus à la coupure).
    # shift(2) suppose un index quotidien continu : on le garantit.
    full_days = pd.date_range(daily.index.min(), daily.index.max(), freq="D")
    level = daily["cons_mean"].reindex(full_days).shift(2).rolling(7, min_periods=5).mean()
    df["level"] = level.reindex(df["day"]).to_numpy()

    df["lag_d2_r"] = lookup(by_slot, 2, True) / df["level"]
    df["lag_d7_r"] = lookup(by_slot, 7, True) / df["level"]
    df["lag_d14_r"] = lookup(by_slot, 14, True) / df["level"]
    df["daymean_d2_r"] = lookup(daily["cons_mean"], 2, False) / df["level"]
    df["morning_d1_r"] = lookup(morning, 1, False) / df["level"]
    df["morning_trend"] = lookup(morning, 1, False) / lookup(morning, 8, False)

    # --- Calendrier ---------------------------------------------------------------------------
    df["dow"] = df["day"].dt.dayofweek
    df["month"] = df["day"].dt.month
    doy = df["day"].dt.dayofyear
    df["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    df["is_weekend"] = (df["dow"] >= 5).astype(int)
    holidays = set(df.loc[df["is_holiday"], "day"])
    df["holiday_eve"] = (df["day"] + pd.Timedelta(days=1)).isin(holidays).astype(int)
    df["holiday_after"] = (df["day"] - pd.Timedelta(days=1)).isin(holidays).astype(int)
    md = df["day"].dt.month * 100 + df["day"].dt.day
    df["xmas_period"] = ((md >= 1224) | (md <= 101)).astype(int)
    for col in ["is_holiday", "is_bridge"]:
        df[col] = df[col].astype(int)

    # --- Météo --------------------------------------------------------------------------------
    src = "obs" if weather == "observed" else "fc"
    df["temp"] = df[f"temp_{src}_c"]
    day_temp = df.groupby("day")["temp"].agg(["mean", "min", "max"])
    df["temp_daymean"] = day_temp["mean"].reindex(df["day"]).to_numpy()
    df["temp_daymin"] = day_temp["min"].reindex(df["day"]).to_numpy()
    df["temp_daymax"] = day_temp["max"].reindex(df["day"]).to_numpy()
    df["hdd"] = (15 - df["temp"]).clip(lower=0)  # besoin de chauffage
    df["cdd"] = (df["temp"] - 20).clip(lower=0)  # besoin de climatisation

    # D-1 : l'après-midi n'est pas encore passé à la coupure → même vue que le jour prédit.
    temp_d1_daily = df.groupby("day")["temp"].mean() if weather == "forecast" else temp_past_daily
    df["temp_d1_mean"] = temp_d1_daily.reindex(shift_days(1)).to_numpy()
    df["temp_d2_mean"] = temp_past_daily.reindex(shift_days(2)).to_numpy()
    df["temp_minus_d7"] = (
        df["temp"] - temp_past_slot.reindex(pd.MultiIndex.from_arrays([shift_days(7), df["slot"]])).to_numpy()
    )
    # Écart au climat de la semaine qui sert de niveau (D-8 à D-2) : c'est ce qui manquait au
    # modèle pour réagir aux coups de froid, où la conso bondit de 30 à 40 % en quelques jours.
    temp_level = temp_past_daily.reindex(full_days).shift(2).rolling(7, min_periods=5).mean()
    df["temp_minus_level"] = df["temp_daymean"] - temp_level.reindex(df["day"]).to_numpy()

    df[TARGET] = df["consumption_mw"] / df["level"]
    return df


def to_mw(ratio: np.ndarray | pd.Series, frame: pd.DataFrame) -> np.ndarray:
    return np.asarray(ratio) * frame["level"].to_numpy()


def baselines(frame: pd.DataFrame) -> pd.DataFrame:
    """Références naïves : ce que tout modèle doit battre avant de se comparer à RTE."""
    lag7 = frame["lag_d7_r"] * frame["level"]
    lag14 = frame["lag_d14_r"] * frame["level"]
    return pd.DataFrame(
        {"naive_d7": lag7, "seasonal_d7_d14": (lag7 + lag14) / 2},
        index=frame.index,
    )
