-- Trous temporels : sur l'historique déjà réalisé, moins de 0,1 % de demi-heures sans conso.
-- Au-delà, le backtest serait faussé en silence.
with bounds as (
    select max(ts_utc) as last_known
    from {{ ref('stg_consumption') }}
    where consumption_mw is not null
),
history as (
    select m.consumption_mw
    from {{ ref('mart_halfhourly') }} m, bounds b
    where m.ts_utc <= b.last_known
)
select count(*) filter (where consumption_mw is null) as missing, count(*) as total
from history
having count(*) filter (where consumption_mw is null) > 0.001 * count(*)
