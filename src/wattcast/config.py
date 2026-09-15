"""Chargement de la configuration pays et chemins du projet."""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(os.environ.get("WATTCAST_ROOT", Path(__file__).resolve().parents[2]))
CONFIG_DIR = ROOT / "configs"


@dataclass(frozen=True)
class CountryConfig:
    raw: dict[str, Any]

    @property
    def country(self) -> str:
        return self.raw["country"]

    @property
    def timezone(self) -> str:
        return self.raw["timezone"]

    @property
    def freq_minutes(self) -> int:
        return int(self.raw["freq_minutes"])

    @property
    def steps_per_day(self) -> int:
        return 24 * 60 // self.freq_minutes

    @property
    def sources(self) -> dict[str, Any]:
        return self.raw["sources"]

    @cached_property
    def paths(self) -> Paths:
        return Paths(ROOT / "data", self.country)


@dataclass(frozen=True)
class Paths:
    data: Path
    country: str

    @property
    def raw(self) -> Path:
        return self.data / "raw" / self.country

    @property
    def warehouse(self) -> Path:
        return self.data / "warehouse.duckdb"

    @property
    def models(self) -> Path:
        return self.data / "models" / self.country

    @property
    def outputs(self) -> Path:
        """Ce qui est publié : prévisions live, backtest, scores, drift."""
        return self.data / "outputs" / self.country


def load_config(country: str = "france") -> CountryConfig:
    with open(CONFIG_DIR / f"{country}.yaml", encoding="utf-8") as f:
        return CountryConfig(yaml.safe_load(f))
