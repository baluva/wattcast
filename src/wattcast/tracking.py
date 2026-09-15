"""MLflow en local (SQLite) : aucun serveur à faire tourner, `just mlflow` pour l'UI."""

from __future__ import annotations

import mlflow

from wattcast.config import CountryConfig


def setup_mlflow(cfg: CountryConfig) -> None:
    db = (cfg.paths.data / "mlflow.db").as_posix()
    mlflow.set_tracking_uri(f"sqlite:///{db}")
    mlflow.set_experiment(f"wattcast-{cfg.country}")
