"""Ingestion eCO2mix (RTE, via la plateforme ODRÉ).

Deux jeux complémentaires :
- consolidé : 2012 → il y a quelques mois, valeurs définitives, conso à la demi-heure ;
- temps réel : depuis la fin du consolidé, réécrit par RTE au fil de l'eau.

Chaque jeu est réécrit en entier à chaque exécution (idempotent). La priorité
consolidé > temps réel est appliquée dans dbt, pas ici : la couche brute reste brute.
"""

from __future__ import annotations

import io
import logging

import pandas as pd

from wattcast.config import CountryConfig
from wattcast.ingest._http import get, write_parquet

log = logging.getLogger(__name__)


def _max_timestamp(base_url: str, dataset: str) -> str | None:
    resp = get(f"{base_url}/{dataset}/records", {"select": "max(date_heure) as m"})
    results = resp.json()["results"]
    return results[0]["m"] if results else None


def _export(base_url: str, dataset: str, fields: list[str]) -> pd.DataFrame:
    resp = get(f"{base_url}/{dataset}/exports/parquet", {"select": ",".join(fields)})
    df = pd.read_parquet(io.BytesIO(resp.content))
    df["date_heure"] = pd.to_datetime(df["date_heure"], utc=True)
    for col in fields[1:]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.sort_values("date_heure").reset_index(drop=True)


def ingest_consumption(cfg: CountryConfig, *, force: bool = False) -> dict[str, int]:
    src = cfg.sources["consumption"]
    out_dir = cfg.paths.raw / "consumption"
    stats: dict[str, int] = {}

    for kind in ("consolidated", "realtime"):
        dataset = src[kind]
        path = out_dir / f"{kind}.parquet"
        if kind == "consolidated" and path.exists() and not force:
            # Le consolidé ne bouge que quelques fois par an : on évite 30 Mo de téléchargement.
            remote_max = pd.Timestamp(_max_timestamp(src["base_url"], dataset))
            local_max = pd.read_parquet(path, columns=["date_heure"])["date_heure"].max()
            if remote_max <= local_max:
                log.info("%s déjà à jour (%s)", dataset, local_max)
                stats[kind] = 0
                continue
        df = _export(src["base_url"], dataset, src["fields"])
        write_parquet(df, path)
        stats[kind] = len(df)
        log.info("%s : %d lignes, %s → %s", dataset, len(df), df["date_heure"].min(), df["date_heure"].max())
    return stats
