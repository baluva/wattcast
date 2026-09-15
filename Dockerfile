# API WattCast : image légère, uniquement ce qui sert à lire les sorties publiées.
FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.11 /uv /usr/local/bin/uv
WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy PYTHONUTF8=1

COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project

COPY configs ./configs
COPY src ./src
RUN uv sync --frozen --no-dev

# Les sorties sont montées en volume (-v ./data:/app/data) ou récupérées au démarrage.
ENV WATTCAST_ROOT=/app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"
CMD ["uv", "run", "--no-sync", "uvicorn", "wattcast.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
