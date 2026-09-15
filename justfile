set shell := ["bash", "-uc"]
set dotenv-load

export PYTHONUTF8 := "1"
country := "france"

# Liste des commandes
default:
    @just --list

# Installe l'environnement et les hooks
setup:
    uv sync
    uv run pre-commit install

# Télécharge conso RTE, météo et calendrier (incrémental)
ingest *args:
    uv run wattcast --country {{country}} ingest {{args}}

# Construit l'entrepôt DuckDB et lance les tests de qualité dbt
dbt:
    cd dbt && uv run dbt build --profiles-dir . --vars '{raw_path: ../data/raw/{{country}}}'

# Rejoue l'historique mois par mois et compare à RTE
backtest:
    uv run wattcast --country {{country}} backtest

# Entraîne un challenger et le promeut s'il bat le champion
train *args:
    uv run wattcast --country {{country}} train {{args}}

# Prévision du lendemain (à lancer la veille vers midi)
predict:
    uv run wattcast --country {{country}} predict

# Note les prévisions live et calcule le drift
monitor:
    uv run wattcast --country {{country}} score
    uv run wattcast --country {{country}} drift

# Pipeline quotidien complet, identique à la CI
daily: (ingest) dbt monitor predict

# Tests unitaires + lint
test:
    uv run ruff check .
    uv run pytest -q

# API sur http://localhost:8000/docs
api:
    uv run uvicorn wattcast.api.main:app --reload

# Dashboard sur http://localhost:8501
dashboard:
    uv run streamlit run dashboard/app.py

# Interface MLflow sur http://localhost:5000
mlflow:
    uv run mlflow ui --backend-store-uri sqlite:///data/mlflow.db

# Récupère / pousse l'état sur le dataset Hugging Face
pull:
    uv run python -m wattcast.storage pull
push message="Mise à jour manuelle":
    uv run python -m wattcast.storage push -m "{{message}}"
