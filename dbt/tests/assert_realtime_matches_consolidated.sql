-- Cohérence entre sources : sur la zone où les deux jeux se recouvrent (s'il y en a une),
-- l'écart médian entre temps réel et consolidé doit rester sous 2 %.
with rt as (
    select date_heure as ts_utc, consommation as rt from {{ raw('consumption/realtime.parquet') }}
    where consommation is not null
),
cons as (
    select date_heure as ts_utc, consommation as cons from {{ raw('consumption/consolidated.parquet') }}
    where consommation is not null
)
select median(abs(rt - cons) / cons) as median_rel_gap, count(*) as n
from rt join cons using (ts_utc)
having count(*) > 0 and median(abs(rt - cons) / cons) > 0.02
