"""WattCast — le duel quotidien contre la prévision RTE.

Autonome : lit les sorties publiées (local `data/outputs/<pays>` ou dataset Hugging Face),
sans importer le package, pour se déployer tel quel sur Streamlit Community Cloud.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

COUNTRY = "france"
REPO_ID = os.environ.get("WATTCAST_DATA_REPO", "louey9999/wattcast-data")
LOCAL = Path(__file__).resolve().parents[1] / "data" / "outputs" / COUNTRY

INK, INK_2, MUTED, GRID, PAPER = "#1b1a17", "#55534d", "#8a877f", "#e6e3da", "#fbfaf6"
MODEL, RTE = "#2a78d6", "#eb6834"
MODEL_BAND = "rgba(42,120,214,0.14)"
TZ = "Europe/Paris"

st.set_page_config(page_title="WattCast · France", page_icon="⚡", layout="wide")

st.markdown(
    f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Newsreader:opsz,wght@6..72,400;6..72,600&display=swap');
.stApp {{ background: {PAPER}; color: {INK}; }}
.block-container {{ max-width: 1180px; padding-top: 2.2rem; }}
h1, h2, h3, .serif {{ font-family: 'Newsreader', Georgia, serif !important; color: {INK}; letter-spacing: -0.01em; }}
h1 {{ font-size: 3.1rem !important; font-weight: 600 !important; line-height: 1.05 !important; margin-bottom: 0 !important; }}
h2 {{ font-size: 1.75rem !important; font-weight: 600 !important; border-top: 1px solid {INK}; padding-top: 0.9rem !important; margin-top: 2.4rem !important; }}
.kicker {{ font-size: 0.78rem; letter-spacing: 0.14em; text-transform: uppercase; color: {INK_2}; font-weight: 600; }}
.lede {{ font-family: 'Newsreader', Georgia, serif; font-size: 1.3rem; color: {INK_2}; max-width: 46rem; line-height: 1.45; }}
.stat {{ border-top: 3px solid {INK}; padding-top: 0.6rem; }}
.stat .v {{ font-size: 2.5rem; font-weight: 650; line-height: 1.1; color: {INK}; }}
.stat .l {{ font-size: 0.86rem; color: {INK_2}; margin-top: 0.2rem; line-height: 1.35; }}
.chip {{ display: inline-block; width: 0.8rem; height: 0.8rem; border-radius: 2px; vertical-align: -0.05rem; margin-right: 0.3rem; }}
.note {{ font-size: 0.86rem; color: {INK_2}; line-height: 1.5; }}
.small-caps {{ font-size: 0.74rem; letter-spacing: 0.1em; text-transform: uppercase; color: {MUTED}; }}
[data-testid="stExpander"] {{ border-color: {GRID}; background: transparent; }}
</style>
""",
    unsafe_allow_html=True,
)


# --- Données ------------------------------------------------------------------------------------
@st.cache_data(ttl=1800, show_spinner="Chargement des sorties publiées…")
def load() -> dict:
    base = LOCAL
    if not (base / "backtest_metrics.json").exists():
        from huggingface_hub import snapshot_download

        root = snapshot_download(REPO_ID, repo_type="dataset", allow_patterns=[f"outputs/{COUNTRY}/*"])
        base = Path(root) / "outputs" / COUNTRY

    def pq(name):
        p = base / name
        return pd.read_parquet(p) if p.exists() else None

    def js(name):
        p = base / name
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None

    return {
        "bt": pq("backtest_scored.parquet"),
        "bt_metrics": js("backtest_metrics.json"),
        "live": pq("live_predictions.parquet"),
        "live_scores": pq("live_scores.parquet"),
        "live_metrics": js("live_metrics.json"),
        "drift": js("drift.json"),
    }


data = load()
bt, btm = data["bt"], data["bt_metrics"]
if bt is None or btm is None:
    st.error("Aucune sortie publiée pour l'instant.")
    st.stop()
o = btm["overall"]


def fr(x: float, d: int = 1) -> str:
    return f"{x:,.{d}f}".replace(",", " ").replace(".", ",")


JOURS = ["lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi", "dimanche"]
MOIS = [
    "janvier",
    "février",
    "mars",
    "avril",
    "mai",
    "juin",
    "juillet",
    "août",
    "septembre",
    "octobre",
    "novembre",
    "décembre",
]


def date_fr(d: pd.Timestamp) -> str:
    """Pas de dépendance à la locale du serveur (un Space tourne en anglais)."""
    return f"{JOURS[d.dayofweek].capitalize()} {d.day} {MOIS[d.month - 1]} {d.year}"


def layout(fig: go.Figure, height: int = 360, y_title: str = "") -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=8, r=8, t=16, b=8),
        paper_bgcolor=PAPER,
        plot_bgcolor=PAPER,
        font=dict(family="system-ui, -apple-system, Segoe UI, sans-serif", size=13, color=INK_2),
        hovermode="x unified",
        hoverlabel=dict(bgcolor="white", bordercolor=GRID, font=dict(color=INK, size=12)),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, font=dict(color=INK_2), bgcolor=PAPER),
    )
    fig.update_xaxes(
        showgrid=False, linecolor="#c9c6bb", ticks="outside", tickcolor="#c9c6bb", tickfont=dict(color=MUTED)
    )
    fig.update_yaxes(
        gridcolor=GRID,
        zeroline=False,
        tickfont=dict(color=MUTED),
        title=dict(text=y_title, font=dict(color=MUTED, size=12)),
    )
    return fig


# --- En-tête --------------------------------------------------------------------------------------
live = data["live"]
last_issue = None if live is None else pd.to_datetime(live["issued_at"]).max().tz_convert(TZ)

st.markdown('<div class="kicker">Réseau électrique français · prévision à J+1</div>', unsafe_allow_html=True)
st.markdown("# Mon modèle contre RTE,<br>une demi-heure après l'autre", unsafe_allow_html=True)
st.markdown(
    """<p class="lede">Chaque jour avant midi, WattCast prévoit la consommation électrique du lendemain.
Il est jugé sur les mêmes demi-heures que la prévision officielle J-1 de RTE, avec la météo
telle qu'elle était <em>prévue</em> la veille, jamais celle qui a vraiment eu lieu.</p>""",
    unsafe_allow_html=True,
)

win_pct = o["win_rate_vs_rte"] * 100
c1, c2, c3, c4 = st.columns(4)
stats = [
    (
        fr(o["mape_model_cal"], 2) + " %",
        f"erreur moyenne du modèle (MAPE), backtest {min(btm['by_year'])}–{max(btm['by_year'])}",
    ),
    (
        fr(o["mape_rte_cal"], 2) + " %",
        "erreur moyenne de la prévision J-1 de RTE recalibrée, mêmes demi-heures",
    ),
    (
        f"{fr(win_pct, 0)} %",
        f"des jours où le modèle fait mieux que RTE ({o['days_model_beats_rte']} sur {o['days']})",
    ),
    (f"{fr(o['interval_coverage_80'] * 100, 0)} %", "des valeurs réelles dans l'intervalle annoncé à 80 %"),
]
for col, (v, lab) in zip((c1, c2, c3, c4), stats, strict=True):
    col.markdown(
        f'<div class="stat"><div class="v">{v}</div><div class="l">{lab}</div></div>', unsafe_allow_html=True
    )

if last_issue is not None:
    st.markdown(
        f'<p class="small-caps" style="margin-top:1.2rem">Dernière prévision émise le {last_issue:%d/%m/%Y à %H:%M}</p>',
        unsafe_allow_html=True,
    )

# --- Prévision la plus récente -------------------------------------------------------------------
st.markdown("## La dernière prévision")
if live is None or live.empty:
    st.markdown(
        '<p class="note">Le suivi live démarre avec la première exécution du pipeline quotidien.</p>',
        unsafe_allow_html=True,
    )
else:
    target = live["target_day"].max()
    day = live[live["target_day"] == target].sort_values("ts_utc").copy()
    day["t"] = pd.to_datetime(day["ts_utc"], utc=True).dt.tz_convert(TZ).dt.tz_localize(None)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=day["t"], y=day["hi"], line=dict(width=0), hoverinfo="skip", showlegend=False))
    fig.add_trace(
        go.Scatter(
            x=day["t"],
            y=day["lo"],
            fill="tonexty",
            fillcolor=MODEL_BAND,
            line=dict(width=0),
            name="Intervalle à 80 %",
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=day["t"],
            y=day["pred"],
            name="WattCast",
            line=dict(color=MODEL, width=2),
            hovertemplate="%{y:,.0f} MW",
        )
    )
    if day["rte_d1"].notna().any():
        fig.add_trace(
            go.Scatter(
                x=day["t"],
                y=day["rte_d1"],
                name="RTE J-1",
                line=dict(color=RTE, width=2),
                hovertemplate="%{y:,.0f} MW",
            )
        )
    if day["actual"].notna().any():
        fig.add_trace(
            go.Scatter(
                x=day["t"],
                y=day["actual"],
                name="Réalisé",
                line=dict(color=INK, width=2, dash="dot"),
                hovertemplate="%{y:,.0f} MW",
            )
        )
    fig.update_xaxes(tickformat="%H:%M")
    st.markdown(
        f'<p class="serif" style="font-size:1.2rem;margin:0">{date_fr(target)}</p>', unsafe_allow_html=True
    )
    st.plotly_chart(layout(fig, 380, "MW"), use_container_width=True, config={"displayModeBar": False})
    peak = day.loc[day["pred"].idxmax()]
    st.markdown(
        f'<p class="note">Pointe prévue : <b>{fr(peak["pred"], 0)} MW</b> à {peak["t"]:%H:%M} · modèle <code>{peak["model_version"]}</code>'
        f" · émise le {pd.Timestamp(peak['issued_at']).tz_convert(TZ):%d/%m à %H:%M}, jamais modifiée depuis.</p>",
        unsafe_allow_html=True,
    )

# --- Le duel dans le temps -----------------------------------------------------------------------
st.markdown("## Le duel, mois après mois")
daily = bt.groupby("day").apply(
    lambda g: pd.Series(
        {
            "model": ((g["model_cal"] - g["actual"]).abs() / g["actual"]).mean() * 100,
            "rte": ((g["rte_cal"] - g["actual"]).abs() / g["actual"]).mean() * 100,
        }
    ),
    include_groups=False,
)
monthly = daily.resample("MS").mean()
fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=monthly.index,
        y=monthly["rte"],
        name="RTE J-1",
        line=dict(color=RTE, width=2),
        hovertemplate="%{y:.2f} %",
    )
)
fig.add_trace(
    go.Scatter(
        x=monthly.index,
        y=monthly["model"],
        name="WattCast",
        line=dict(color=MODEL, width=2),
        hovertemplate="%{y:.2f} %",
    )
)
fig.update_xaxes(tickformat="%b %Y", hoverformat="%B %Y")
fig.update_yaxes(rangemode="tozero")
left, right = st.columns([2.3, 1])
with left:
    st.plotly_chart(
        layout(fig, 340, "MAPE journalier moyen (%)"),
        use_container_width=True,
        config={"displayModeBar": False},
    )
with right:
    rows = "".join(
        f"<tr><td>{y}</td><td style='text-align:right'>{fr(v['mape_model'], 2)}</td>"
        f"<td style='text-align:right'>{fr(v['mape_rte'], 2)}</td>"
        f"<td style='text-align:right'>{fr(v['win_rate_vs_rte'] * 100, 0)} %</td></tr>"
        for y, v in btm["by_year"].items()
    )
    st.markdown(
        f"""<table style="width:100%;font-variant-numeric:tabular-nums;border-collapse:collapse;font-size:0.92rem">
<thead><tr style="border-bottom:1px solid {INK};text-align:right"><th style="text-align:left">Année</th>
<th><span class="chip" style="background:{MODEL}"></span>Modèle</th><th><span class="chip" style="background:{RTE}"></span>RTE</th><th>Jours gagnés</th></tr></thead>
<tbody>{rows}</tbody></table>
<p class="note" style="margin-top:0.8rem">MAPE en %, sur les demi-heures où les deux prévisions existent.
Chaque mois est prédit par un modèle entraîné uniquement sur le passé (walk-forward).</p>""",
        unsafe_allow_html=True,
    )

# --- Où ça se joue ------------------------------------------------------------------------------
st.markdown("## Où se fait la différence")
bt_local = bt.assign(hour=pd.to_datetime(bt["ts_utc"], utc=True).dt.tz_convert(TZ).dt.hour)
by_hour = bt_local.groupby("hour").apply(
    lambda g: pd.Series(
        {
            "model": ((g["model_cal"] - g["actual"]).abs() / g["actual"]).mean() * 100,
            "rte": ((g["rte_cal"] - g["actual"]).abs() / g["actual"]).mean() * 100,
        }
    ),
    include_groups=False,
)
bt_local["kind"] = bt_local["day"].dt.dayofweek.map(lambda d: "Week-end" if d >= 5 else "Semaine")
col_a, col_b = st.columns(2)
with col_a:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=by_hour.index,
            y=by_hour["rte"],
            name="RTE J-1",
            line=dict(color=RTE, width=2),
            hovertemplate="%{y:.2f} %",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=by_hour.index,
            y=by_hour["model"],
            name="WattCast",
            line=dict(color=MODEL, width=2),
            hovertemplate="%{y:.2f} %",
        )
    )
    fig.update_xaxes(tickvals=list(range(0, 24, 3)), ticktext=[f"{h} h" for h in range(0, 24, 3)])
    fig.update_yaxes(rangemode="tozero")
    st.markdown(
        '<p class="serif" style="font-size:1.15rem;margin:0">Erreur selon l\'heure de la journée</p>',
        unsafe_allow_html=True,
    )
    st.plotly_chart(layout(fig, 300, "MAPE (%)"), use_container_width=True, config={"displayModeBar": False})
with col_b:
    imp = pd.Series(btm["feature_importance"]).head(10).iloc[::-1]
    labels = {
        "slot": "Heure de la journée",
        "lag_d7_r": "Conso J-7, même heure",
        "lag_d2_r": "Conso J-2, même heure",
        "lag_d14_r": "Conso J-14, même heure",
        "morning_d1_r": "Matinée de la veille",
        "temp": "Température prévue",
        "temp_daymean": "Température moyenne du jour",
        "hdd": "Besoin de chauffage",
        "cdd": "Besoin de clim",
        "dow": "Jour de la semaine",
        "is_holiday": "Jour férié",
        "daymean_d2_r": "Moyenne de J-2",
        "doy_sin": "Saison (sin)",
        "doy_cos": "Saison (cos)",
        "temp_d1_mean": "Température de la veille",
        "temp_d2_mean": "Température de J-2",
        "temp_minus_d7": "Écart de température vs J-7",
        "temp_minus_level": "Écart de température vs semaine de réf.",
        "morning_trend": "Tendance de la matinée",
        "radiation": "Ensoleillement",
        "cloud": "Nébulosité",
        "temp_daymin": "Température min",
        "temp_daymax": "Température max",
        "is_weekend": "Week-end",
        "month": "Mois",
        "school_zones_off": "Vacances scolaires",
        "is_bridge": "Pont",
        "holiday_eve": "Veille de férié",
        "holiday_after": "Lendemain de férié",
        "xmas_period": "Fêtes de fin d'année",
        "wind": "Vent",
    }
    fig = go.Figure(
        go.Bar(
            x=imp.values * 100,
            y=[labels.get(i, i) for i in imp.index],
            orientation="h",
            marker=dict(color=MODEL, cornerradius=4),
            hovertemplate="%{x:.1f} % du gain<extra></extra>",
        )
    )
    fig.update_xaxes(showgrid=True, gridcolor=GRID, ticksuffix=" %")
    fig.update_yaxes(gridcolor=PAPER, tickfont=dict(color=INK_2))
    st.markdown(
        '<p class="serif" style="font-size:1.15rem;margin:0">Ce qui pèse dans la décision du modèle</p>',
        unsafe_allow_html=True,
    )
    f = layout(fig, 300)
    f.update_layout(showlegend=False, hovermode="closest", bargap=0.35)
    st.plotly_chart(f, use_container_width=True, config={"displayModeBar": False})

# --- Suivi live & santé ------------------------------------------------------------------------
st.markdown("## En production")
lm, drift = data["live_metrics"], data["drift"]
k1, k2, k3 = st.columns(3)
if lm:
    k1.markdown(
        f'<div class="stat"><div class="v">{lm["days"]} j</div><div class="l">notés en live depuis le {lm["first_day"]}</div></div>',
        unsafe_allow_html=True,
    )
    k2.markdown(
        f'<div class="stat"><div class="v">{fr(lm["mape_model"], 2)} % <span style="font-size:1.1rem;color:{INK_2}">vs {fr(lm["mape_rte"], 2)} %</span></div><div class="l">MAPE live, modèle vs RTE ({fr(lm["win_rate_vs_rte"] * 100, 0)} % de jours gagnés)</div></div>',
        unsafe_allow_html=True,
    )
elif live is not None and not live.empty:
    first = live["target_day"].min()
    k1.markdown(
        f'<div class="stat"><div class="v">0 j</div><div class="l">notés en live pour l\'instant : la prévision du {date_fr(first).lower()} sera notée le lendemain, une fois la conso publiée</div></div>',
        unsafe_allow_html=True,
    )
if live is not None and not live.empty:
    k2.markdown(
        f'<div class="stat"><div class="v">{live["target_day"].nunique()}</div><div class="l">prévision(s) émise(s) depuis la mise en production, modèle <code>{live["model_version"].iloc[-1]}</code></div></div>',
        unsafe_allow_html=True,
    )
if drift:
    state = "Réentraînement recommandé" if drift["retrain_recommended"] else "Rien à signaler"
    k3.markdown(
        f'<div class="stat"><div class="v" style="font-size:1.6rem">{"⚠ " if drift["retrain_recommended"] else "✓ "}{state}</div>'
        f'<div class="l">drift : {fr(drift["drift_share"] * 100, 0)} % des variables surveillées ont bougé par rapport à la même période l\'an dernier</div></div>',
        unsafe_allow_html=True,
    )

with st.expander("Méthode, en bref"):
    st.markdown(
        f"""
- **Protocole** : la veille à midi, on prédit les 48 demi-heures du lendemain. Seules les données publiées à ce moment-là servent : conso jusqu'à J-2 en entier et matinée de J-1, météo **prévue** pour J-1 et J.
- **Données** : eCO2mix (RTE, via ODRÉ), Open-Meteo (8 aires urbaines pondérées par leur population), jours fériés, ponts, vacances scolaires.
- **Modèle** : LightGBM sur le ratio conso / niveau des 7 derniers jours connus, pour suivre les baisses de consommation (sobriété 2022-2023) qu'un modèle à arbres n'extrapole pas.
- **Références** : naïf J-7 ({fr(o["mape_naive_d7"], 2)} %), moyenne J-7/J-14 ({fr(o["mape_seasonal_d7_d14"], 2)} %). Avec la météo *réelle* au lieu de la météo prévue, le modèle ferait {fr(o["mape_model_perfect_weather"], 2)} % : c'est le coût des erreurs de prévision météo.
- **Recalibrage, le piège évité** : RTE prévoit la conso *au sens temps réel* ; la conso consolidée, publiée des mois plus tard, est révisée d'environ {fr(-o["bias_rte_raw_pct"], 1)} %. Comparée brute, la prévision RTE paraît fausse de {fr(o["mape_rte_d1"], 2)} % et le modèle « gagne » {fr(o["win_rate_vs_rte_raw"] * 100, 0)} % des jours : c'est un artefact de définition. Les deux concurrents sont donc corrigés de leur biais médian des 4 dernières semaines, avec les seules erreurs connues la veille.
- **Intervalle** : conformal en ligne, quantile 80 % des erreurs des 8 dernières semaines, calculé sans regarder le jour prédit.
- **Honnêteté** : les prévisions live sont journalisées une fois et jamais réécrites. RTE dispose de données internes que ce projet n'a pas : le but n'est pas de « battre RTE » à tout prix, mais de mesurer l'écart proprement.
"""
    )
with st.expander("Voir les chiffres en tableau"):
    st.dataframe(
        monthly.rename(columns={"model": "MAPE modèle (%)", "rte": "MAPE RTE (%)"}).round(3),
        use_container_width=True,
    )

st.markdown(
    f'<p class="small-caps" style="margin-top:3rem;border-top:1px solid {GRID};padding-top:0.8rem">'
    f'WattCast · Louey Barbirou · code sur <a href="https://github.com/baluva/wattcast" style="color:{INK_2}">github.com/baluva/wattcast</a>'
    f" · données RTE eCO2mix & Open-Meteo</p>",
    unsafe_allow_html=True,
)
