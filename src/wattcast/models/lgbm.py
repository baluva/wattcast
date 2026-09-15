"""Modèle LightGBM sur le ratio conso / niveau récent."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from wattcast.features import FEATURES, TARGET, to_mw

PARAMS: dict = {
    "objective": "l2",
    "learning_rate": 0.04,
    "num_leaves": 63,
    "min_data_in_leaf": 60,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "verbosity": -1,
    "seed": 42,
}
NUM_ROUNDS = 900


@dataclass
class Forecaster:
    params: dict = field(default_factory=lambda: dict(PARAMS))
    num_rounds: int = NUM_ROUNDS
    booster: lgb.Booster | None = None

    def fit(self, frame: pd.DataFrame) -> Forecaster:
        train = frame.dropna(subset=[TARGET, "lag_d7_r", "level"])
        data = lgb.Dataset(train[FEATURES], train[TARGET], free_raw_data=True)
        self.booster = lgb.train(self.params, data, num_boost_round=self.num_rounds)
        return self

    def predict_mw(self, frame: pd.DataFrame) -> np.ndarray:
        if self.booster is None:
            raise RuntimeError("Modèle non entraîné")
        return to_mw(self.booster.predict(frame[FEATURES]), frame)

    def feature_importance(self) -> pd.Series:
        assert self.booster is not None
        gain = self.booster.feature_importance(importance_type="gain")
        return pd.Series(gain, index=FEATURES).sort_values(ascending=False) / gain.sum()

    def save(self, path: Path) -> None:
        assert self.booster is not None
        path.parent.mkdir(parents=True, exist_ok=True)
        self.booster.save_model(str(path))

    @classmethod
    def load(cls, path: Path) -> Forecaster:
        return cls(booster=lgb.Booster(model_file=str(path)))
