from __future__ import annotations

import numpy as np
import pandas as pd

from wattcast.evaluate.metrics import daily_errors, mape, online_bias_correction, online_conformal


def _series(n_days: int, pred: float) -> pd.DataFrame:
    days = pd.date_range("2025-01-01", periods=n_days, freq="D").repeat(48)
    return pd.DataFrame({"day": days, "actual": 100.0, "pred": pred})


def test_bias_correction_removes_constant_bias():
    """RTE sous-estime de 2,5 % en permanence : après recalibrage, l'erreur disparaît."""
    df = _series(40, 97.5)
    corrected = online_bias_correction(df, "pred")
    late = df["day"] >= "2025-01-20"
    assert np.allclose(corrected[late], 100.0)
    # Les 15 premiers jours n'ont pas assez d'historique : prévision laissée telle quelle.
    assert np.allclose(corrected[df["day"] <= "2025-01-15"], 97.5)


def test_bias_correction_ignores_target_day_and_eve():
    df = _series(40, 97.5)
    base = online_bias_correction(df, "pred")
    shocked = df.copy()
    shocked.loc[shocked["day"] >= "2025-02-08", "pred"] = 50.0  # J-1 et J faussés
    after = online_bias_correction(shocked, "pred")
    d = df["day"] == "2025-02-09"
    np.testing.assert_allclose(base[d] / df.loc[d, "pred"], after[d] / shocked.loc[d, "pred"])


def test_mape_ignores_missing():
    actual = pd.Series([100.0, 200.0, np.nan])
    pred = pd.Series([110.0, 180.0, 5.0])
    assert mape(actual, pred) == 10.0


def test_daily_errors_drops_incomplete_days():
    days = pd.to_datetime(["2025-01-01"] * 48 + ["2025-01-02"] * 10)
    df = pd.DataFrame({"day": days, "actual": 100.0, "model": 110.0, "rte_d1": 95.0})
    out = daily_errors(df, ["model", "rte_d1"])
    assert list(out.index) == [pd.Timestamp("2025-01-01")]
    assert out.loc["2025-01-01", "model"] == 10.0


def test_online_conformal_uses_only_past_errors():
    """L'intervalle du jour D ne doit pas dépendre de l'erreur du jour D ni de D-1."""
    days = pd.date_range("2025-01-01", periods=40, freq="D").repeat(48)
    rng = np.random.default_rng(1)
    df = pd.DataFrame({"day": days, "actual": 100.0, "model": 100 + rng.normal(0, 2, len(days))})
    base = online_conformal(df, "model")

    shocked = df.copy()
    last_two = shocked["day"] >= "2025-02-08"
    shocked.loc[last_two, "model"] = 1_000.0  # erreurs énormes sur D-1 et D
    after = online_conformal(shocked, "model")

    d = shocked["day"] == "2025-02-09"
    half_base = (base.loc[d, "hi"] / df.loc[d, "model"] - 1).round(9)
    half_after = (after.loc[d, "hi"] / shocked.loc[d, "model"] - 1).round(9)
    pd.testing.assert_series_equal(half_base, half_after)
