"""Point d'entrée : `uv run wattcast <commande> --country france`."""

from __future__ import annotations

import argparse
import logging

from wattcast.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(prog="wattcast")
    parser.add_argument("--country", default="france")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("ingest", help="Télécharge conso, météo et calendrier")
    p.add_argument("--force", action="store_true", help="Retélécharge même ce qui est à jour")
    p.add_argument("--only", choices=["consumption", "observed", "forecast_d1", "calendar"])

    sub.add_parser("backtest", help="Walk-forward sur l'historique, comparé à RTE")
    sub.add_parser("rescore", help="Recalcule les scores publiés du dernier backtest, sans réentraîner")
    p = sub.add_parser("train", help="Entraîne le modèle candidat et le promeut s'il bat le champion")
    p.add_argument("--promote", action="store_true", help="Force la promotion (premier modèle)")
    sub.add_parser("predict", help="Prévision live du lendemain (à lancer la veille à midi)")
    sub.add_parser("score", help="Note les prévisions live dont la conso réalisée est connue")
    sub.add_parser("drift", help="Rapport de drift Evidently")

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    cfg = load_config(args.country)

    if args.cmd == "ingest":
        from wattcast.ingest import calendar, rte, weather

        steps = {
            "consumption": lambda: rte.ingest_consumption(cfg, force=args.force),
            "observed": lambda: weather.ingest_observed(cfg, force=args.force),
            "forecast_d1": lambda: weather.ingest_forecast_d1(cfg, force=args.force),
            "calendar": lambda: calendar.build_calendar(cfg),
        }
        for name, step in steps.items():
            if args.only in (None, name):
                step()
    elif args.cmd == "backtest":
        from wattcast.evaluate.backtest import run_backtest

        run_backtest(cfg)
    elif args.cmd == "rescore":
        from wattcast.evaluate.backtest import publish_scores

        publish_scores(cfg)
    elif args.cmd == "train":
        from wattcast.models.registry import train_and_maybe_promote

        train_and_maybe_promote(cfg, force_promote=args.promote)
    elif args.cmd == "predict":
        from wattcast.live import predict_tomorrow

        predict_tomorrow(cfg)
    elif args.cmd == "score":
        from wattcast.live import score_live

        score_live(cfg)
    elif args.cmd == "drift":
        from wattcast.evaluate.drift import drift_report

        drift_report(cfg)


if __name__ == "__main__":
    main()
