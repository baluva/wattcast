select
    day::date as day,
    holiday_name,
    is_holiday,
    is_bridge,
    school_zones_off::integer as school_zones_off
from {{ raw('calendar.parquet') }}
