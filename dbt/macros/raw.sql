{# Lit un Parquet (ou un motif glob) de la couche brute du pays courant. #}
{% macro raw(relative) -%}
read_parquet('{{ var("raw_path") }}/{{ relative }}', union_by_name = true)
{%- endmacro %}
