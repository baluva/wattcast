-- Table de modélisation : une ligne par demi-heure, sur une grille complète (les trous
-- restent visibles en NULL au lieu de disparaître). Pas de variable décalée ici : les lags
-- dépendent de l'heure de coupure, ils sont calculés en Python avec le protocole de prévision.
with grid as (
    select unnest(generate_series(
        timestamptz '{{ var("history_start") }} 00:00:00+00',
        -- Jusqu'au bout d'après-demain au moins : le live a besoin des lignes du jour à prédire,
        -- même si RTE n'a pas encore publié sa prévision.
        greatest(
            (select max(ts_utc) from {{ ref('stg_consumption') }}),
            date_trunc('day', now()) + interval 3 day - interval 30 minute
        ),
        interval 30 minute
    )) as ts_utc
),

weather as (
    select
        ts_utc,
        max(temperature_c) filter (where view = 'observed') as temp_obs_c,
        max(cloud_cover_pct) filter (where view = 'observed') as cloud_obs_pct,
        max(radiation_wm2) filter (where view = 'observed') as radiation_obs_wm2,
        max(wind_kmh) filter (where view = 'observed') as wind_obs_kmh,
        max(temperature_c) filter (where view = 'forecast_d1') as temp_fc_c,
        max(cloud_cover_pct) filter (where view = 'forecast_d1') as cloud_fc_pct,
        max(radiation_wm2) filter (where view = 'forecast_d1') as radiation_fc_wm2,
        max(wind_kmh) filter (where view = 'forecast_d1') as wind_fc_kmh,
        bool_or(filled_from_historical) filter (where view = 'forecast_d1') as temp_fc_filled
    from {{ ref('fct_weather_national') }}
    group by ts_utc
),

-- Météo horaire → demi-heure : à :30, moyenne de l'heure pleine et de la suivante.
weather_halfhour as (
    select
        g.ts_utc,
        {% for col in ['temp_obs_c', 'cloud_obs_pct', 'radiation_obs_wm2', 'wind_obs_kmh',
                       'temp_fc_c', 'cloud_fc_pct', 'radiation_fc_wm2', 'wind_fc_kmh'] %}
        case when minute(g.ts_utc) = 0 then h0.{{ col }}
             else (h0.{{ col }} + h1.{{ col }}) / 2 end as {{ col }},
        {% endfor %}
        -- Vrai si la température prévue vient d'une prévision plus fraîche qu'une vraie J-1.
        coalesce(h0.temp_fc_filled, false)
            or (minute(g.ts_utc) = 30 and coalesce(h1.temp_fc_filled, false)) as temp_fc_filled
    from grid g
    left join weather h0 on h0.ts_utc = date_trunc('hour', g.ts_utc)
    left join weather h1 on h1.ts_utc = date_trunc('hour', g.ts_utc) + interval 1 hour
),

local_time as (
    select
        ts_utc,
        timezone('{{ var("timezone") }}', ts_utc) as ts_local
    from grid
)

select
    g.ts_utc,
    lt.ts_local,
    lt.ts_local::date as day_local,
    c.consumption_mw,
    c.rte_forecast_d1_mw,
    c.rte_forecast_d0_mw,
    c.source as consumption_source,
    w.* exclude (ts_utc),
    coalesce(cal.is_holiday, false) as is_holiday,
    cal.holiday_name,
    coalesce(cal.is_bridge, false) as is_bridge,
    coalesce(cal.school_zones_off, 0) as school_zones_off
from grid g
join local_time lt using (ts_utc)
left join {{ ref('stg_consumption') }} c using (ts_utc)
left join weather_halfhour w using (ts_utc)
left join {{ ref('stg_calendar') }} cal on cal.day = lt.ts_local::date
order by g.ts_utc
