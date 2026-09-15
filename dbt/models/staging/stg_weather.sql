-- Météo horaire par ville, les deux vues empilées. `view` sépare observé et prévu la veille.
select 'observed' as view, time as ts_utc, point, temperature_2m, cloud_cover, shortwave_radiation,
       wind_speed_10m, false as filled_from_historical
from {{ raw('weather/observed/*.parquet') }}
union all
select 'forecast_d1' as view, time as ts_utc, point, temperature_2m, cloud_cover, shortwave_radiation,
       wind_speed_10m, filled_from_historical
from {{ raw('weather/forecast_d1/*.parquet') }}
