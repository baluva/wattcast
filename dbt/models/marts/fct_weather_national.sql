-- Météo « nationale » horaire : moyenne des villes pondérée par leur population.
--
-- Vue observée : on n'agrège une heure que si toutes les villes sont présentes.
-- Vue prévue la veille : une ville sans archive J-1 (ex. Lille en 2022-2023) est retirée de la
-- moyenne plutôt que comblée par une prévision plus fraîche. Si les villes restantes pèsent
-- moins de 80 % de la population, on se rabat sur les valeurs comblées et on le signale.
with points as (
    select name as point, weight, sum(weight) over () as total_weight
    from {{ raw('weather/points.parquet') }}
),

joined as (
    select w.*, p.weight, p.total_weight
    from {{ ref('stg_weather') }} w
    join points p using (point)
    where w.temperature_2m is not null
),

aggregated as (
    select
        view,
        ts_utc,
        -- Toutes les villes, valeurs comblées comprises
        sum(temperature_2m * weight) / sum(weight) as temp_all,
        -- Uniquement les vraies valeurs (observées, ou prévues avant la coupure)
        sum(temperature_2m * weight) filter (where not filled_from_historical)
            / nullif(sum(weight) filter (where not filled_from_historical), 0) as temp_clean,
        coalesce(sum(weight) filter (where not filled_from_historical), 0) / max(total_weight) as clean_share,
        sum(cloud_cover * weight) / sum(weight) as cloud_cover_pct,
        sum(shortwave_radiation * weight) / sum(weight) as radiation_wm2,
        sum(wind_speed_10m * weight) / sum(weight) as wind_kmh,
        count(*) as n_points
    from joined
    group by all
)

select
    view,
    ts_utc,
    case when clean_share >= 0.8 then temp_clean else temp_all end as temperature_c,
    cloud_cover_pct,
    radiation_wm2,
    wind_kmh,
    round(clean_share, 3) as clean_weight_share,
    clean_share < 0.8 as filled_from_historical
from aggregated
where view = 'forecast_d1' or n_points = (select count(*) from points)
