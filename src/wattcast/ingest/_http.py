"""Client HTTP commun : retries avec backoff, écriture atomique des fichiers."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import httpx
import pandas as pd

log = logging.getLogger(__name__)

RETRY_STATUS = {429, 500, 502, 503, 504}


def get(
    url: str, params: dict[str, Any] | None = None, *, retries: int = 5, timeout: float = 300
) -> httpx.Response:
    """GET avec backoff exponentiel. Lève une erreur claire si l'API ne répond jamais."""
    delay = 2.0
    for attempt in range(1, retries + 1):
        try:
            resp = httpx.get(url, params=params, timeout=timeout, follow_redirects=True)
            if resp.status_code not in RETRY_STATUS:
                resp.raise_for_status()
                return resp
            log.warning("HTTP %s sur %s (essai %d/%d)", resp.status_code, url, attempt, retries)
        except (httpx.TransportError, httpx.TimeoutException) as exc:
            log.warning("%s sur %s (essai %d/%d)", type(exc).__name__, url, attempt, retries)
        if attempt < retries:
            time.sleep(delay)
            delay = min(delay * 2, 60)
    raise RuntimeError(f"Échec après {retries} essais : {url}")


def write_parquet(df: pd.DataFrame, path: Path) -> None:
    """Écrit dans un fichier temporaire puis renomme : jamais de fichier à moitié écrit."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    df.to_parquet(tmp, index=False)
    tmp.replace(path)
