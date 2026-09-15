"""État persistant sur un dataset Hugging Face (gratuit, versionné par git).

GitHub Actions n'a pas de disque persistant : chaque exécution quotidienne récupère
l'état (données brutes, modèle champion, sorties publiées), travaille, puis repousse.
Le dashboard lit les sorties au même endroit. L'entrepôt DuckDB, lui, est reconstruit
par dbt à chaque exécution : ce n'est qu'un dérivé.
"""

from __future__ import annotations

import argparse
import logging
import os

from huggingface_hub import HfApi, snapshot_download

from wattcast.config import ROOT

log = logging.getLogger(__name__)

REPO_ID = os.environ.get("WATTCAST_DATA_REPO", "louey9999/wattcast-data")
# Ce qui est synchronisé. Les instantanés météo live restent : ils prouvent ce que le modèle voyait.
PATTERNS = ["raw/**", "models/**", "outputs/**", "README.md"]


def pull() -> None:
    snapshot_download(REPO_ID, repo_type="dataset", local_dir=ROOT / "data", allow_patterns=PATTERNS)
    log.info("état récupéré depuis %s", REPO_ID)


def push(message: str) -> None:
    api = HfApi()
    api.create_repo(REPO_ID, repo_type="dataset", exist_ok=True)
    api.upload_folder(
        repo_id=REPO_ID,
        repo_type="dataset",
        folder_path=ROOT / "data",
        allow_patterns=PATTERNS,
        commit_message=message,
    )
    log.info("état poussé vers %s : %s", REPO_ID, message)


def main() -> None:
    parser = argparse.ArgumentParser(prog="wattcast-storage")
    parser.add_argument("action", choices=["pull", "push"])
    parser.add_argument("-m", "--message", default="Mise à jour quotidienne")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s")
    pull() if args.action == "pull" else push(args.message)


if __name__ == "__main__":
    main()
