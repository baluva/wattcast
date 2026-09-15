-- Consolidé + temps réel, ramenés à la demi-heure (seul pas où la conso consolidée existe).
-- Priorité : une valeur consolidée remplace toujours la valeur temps réel du même instant.
with unioned as (
    select date_heure as ts_utc, consommation, prevision_j1, prevision_j, 1 as priority, 'consolidated' as source
    from {{ raw('consumption/consolidated.parquet') }}
    union all
    select date_heure as ts_utc, consommation, prevision_j1, prevision_j, 2 as priority, 'realtime' as source
    from {{ raw('consumption/realtime.parquet') }}
),

half_hours as (
    select *
    from unioned
    where minute(ts_utc) in (0, 30)
      and ts_utc >= timestamptz '{{ var("history_start") }} 00:00:00+00'
)

select
    ts_utc,
    consommation::double as consumption_mw,
    prevision_j1::double as rte_forecast_d1_mw,
    prevision_j::double as rte_forecast_d0_mw,
    source
from half_hours
-- Pour chaque instant : la ligne consolidée si elle porte une conso, sinon la plus prioritaire.
qualify row_number() over (
    partition by ts_utc
    order by (consommation is null), priority
) = 1
