from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def make_mart(start: str, end: str) -> pd.DataFrame:
    """Faux mart demi-horaire : profil journalier + creux du week-end + bruit."""
    ts_utc = pd.date_range(start, f"{end} 23:30", freq="30min", tz="UTC")
    ts_local = ts_utc.tz_convert("Europe/Paris")
    rng = np.random.default_rng(0)
    hour = ts_local.hour + ts_local.minute / 60
    base = 50_000 + 8_000 * np.sin((hour - 6) / 24 * 2 * np.pi) - 5_000 * (ts_local.dayofweek >= 5)
    temp = 15 + 5 * np.sin((hour - 9) / 24 * 2 * np.pi) + rng.normal(0, 1, len(ts_utc))
    return pd.DataFrame(
        {
            "ts_utc": ts_utc,
            "ts_local": ts_local.tz_localize(None),
            "day_local": ts_local.tz_localize(None).normalize(),
            "consumption_mw": base + rng.normal(0, 500, len(ts_utc)),
            "rte_forecast_d1_mw": base,
            "temp_obs_c": temp,
            "cloud_obs_pct": 50.0,
            "radiation_obs_wm2": 100.0,
            "wind_obs_kmh": 10.0,
            "temp_fc_c": temp + 0.5,
            "cloud_fc_pct": 55.0,
            "radiation_fc_wm2": 90.0,
            "wind_fc_kmh": 12.0,
            "is_holiday": ts_local.strftime("%m-%d") == "11-11",
            "is_bridge": ts_local.strftime("%m-%d") == "11-10",
            "school_zones_off": 0,
        }
    )


@pytest.fixture
def mart() -> pd.DataFrame:
    """60 jours avec le passage à l'heure d'hiver 2025 (26 octobre) et le 11 novembre."""
    return make_mart("2025-09-20", "2025-11-18")


@pytest.fixture
def spring_mart() -> pd.DataFrame:
    """Passage à l'heure d'été 2025 (30 mars) : un jour de 23 heures."""
    return make_mart("2025-03-01", "2025-04-20")
