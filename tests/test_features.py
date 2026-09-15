from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from wattcast.features import FEATURES, MORNING_SLOTS, TARGET, build_frame

TARGET_DAY = pd.Timestamp("2025-11-05")


@pytest.mark.parametrize("weather", ["observed", "forecast"])
def test_no_leakage_from_after_cutoff(mart, weather):
    """Le test le plus important du projet.

    On sabote tout ce qui n'est pas connu à D-1 midi (après-midi de D-1, jour D, futur) :
    aucune variable du jour D ne doit bouger. Seule la cible, et la météo du jour D dans la
    vue choisie (une prévision légitime), ont le droit de changer.
    """
    cutoff_local = TARGET_DAY - pd.Timedelta(days=1) + pd.Timedelta(hours=MORNING_SLOTS / 2)
    after = mart["ts_local"] >= cutoff_local

    sabotaged = mart.copy()
    sabotaged.loc[after, "consumption_mw"] *= 3
    sabotaged.loc[after, "rte_forecast_d1_mw"] = -1
    sabotaged.loc[after, "temp_obs_c"] += 20  # l'observé du futur est inconnu à la coupure

    clean = build_frame(mart, weather)
    dirty = build_frame(sabotaged, weather)
    day = clean["day"] == TARGET_DAY
    allowed_to_change = {TARGET}
    if weather == "observed":
        # En entraînement, le jour D utilise la météo observée : c'est voulu (et testé à part).
        allowed_to_change |= {
            "temp",
            "temp_daymean",
            "temp_daymin",
            "temp_daymax",
            "hdd",
            "cdd",
            "temp_minus_d7",
            "temp_d1_mean",
            "temp_minus_level",
        }
    checked = [f for f in FEATURES if f not in allowed_to_change]
    pd.testing.assert_frame_equal(clean.loc[day, checked], dirty.loc[day, checked])


def test_forecast_view_uses_forecast_weather(mart):
    fc = build_frame(mart, "forecast")
    obs = build_frame(mart, "observed")
    assert np.allclose(fc["temp"], mart["temp_fc_c"])
    assert np.allclose(obs["temp"], mart["temp_obs_c"])


def test_dst_day_keeps_all_slots(mart):
    """Le 26/10/2025 compte 50 demi-heures (heure d'hiver) : aucune ne doit disparaître."""
    frame = build_frame(mart, "forecast")
    assert (frame["day"] == "2025-10-26").sum() == 50
    assert frame["slot"].between(0, 47).all()


@pytest.mark.parametrize(
    ("fixture", "dst_window"),
    [("spring_mart", ("2025-03-29", "2025-04-08")), ("mart", ("2025-10-25", "2025-11-05"))],
)
def test_daily_aggregates_ignore_dst(request, fixture, dst_window):
    """Profil identique chaque jour : autour du changement d'heure, rien ne doit bouger.

    Bug réel trouvé par l'analyse d'erreurs : le lendemain du passage à l'heure d'été, la
    « matinée de la veille » (jour de 23 h, sans son creux de 2 h) paraissait plus chargée,
    et le modèle surestimait la conso de 8 %.
    """
    flat = request.getfixturevalue(fixture).copy()
    hour = flat["ts_local"].dt.hour + flat["ts_local"].dt.minute / 60
    flat["consumption_mw"] = 50_000 + 8_000 * np.sin((hour - 6) / 24 * 2 * np.pi)
    frame = build_frame(flat, "observed")
    around_dst = frame["day"].between(*dst_window)
    for col in ["morning_trend", "daymean_d2_r"]:
        values = frame.loc[around_dst, col].dropna()
        assert len(values) > 0
        np.testing.assert_allclose(values, values.iloc[0], rtol=1e-9, err_msg=col)


def test_ratio_target_roundtrips_to_mw(mart):
    frame = build_frame(mart, "observed").dropna(subset=["level"])
    assert np.allclose(frame[TARGET] * frame["level"], frame["consumption_mw"])


def test_calendar_flags(mart):
    frame = build_frame(mart, "observed")
    by_day = frame.groupby("day")[["is_holiday", "holiday_eve", "holiday_after", "is_bridge"]].max()
    assert by_day.loc["2025-11-11", "is_holiday"] == 1
    assert by_day.loc["2025-11-10", "holiday_eve"] == 1
    assert by_day.loc["2025-11-12", "holiday_after"] == 1
