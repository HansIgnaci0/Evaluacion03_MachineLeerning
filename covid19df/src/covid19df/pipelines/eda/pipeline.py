"""
Pipeline 'eda' para el proyecto COVID-19 con Kedro.
Contiene nodos para exploración, limpieza, verificación de datos y generación de gráficos.
Cada nodo está documentado y los outputs son únicos.
"""

from kedro.pipeline import Pipeline, node
from .nodes import (
    explore_dataset,
    plot_cases,
    plot_cases_by_continent,
    clean_province_state,
    check_multiple_datasets,
    clean_usa_county,
    clean_worldometer_data,
)


def create_pipeline(**kwargs) -> Pipeline:
    """
    Crea el pipeline 'eda' completo.

    Returns:
        Pipeline: Pipeline de Kedro con nodos para explorar, limpiar, verificar y graficar datasets.
    """
    return Pipeline([

        # ==========================
        # Exploración de datasets
        # ==========================
        node(
            func=explore_dataset,
            inputs=dict(df="country_wise_latest", name="params:dataset1_name"),
            outputs=None,
            name="explore_country_wise",
        ),
        node(
            func=explore_dataset,
            inputs=dict(df="covid_19_clean_complete", name="params:dataset2_name"),
            outputs=None,
            name="explore_clean_complete",
        ),
        node(
            func=explore_dataset,
            inputs=dict(df="day_wise", name="params:dataset3_name"),
            outputs=None,
            name="explore_day_wise",
        ),
        node(
            func=explore_dataset,
            inputs=dict(df="full_grouped", name="params:dataset4_name"),
            outputs=None,
            name="explore_full_grouped",
        ),
        node(
            func=explore_dataset,
            inputs=dict(df="usa_county_wise", name="params:dataset5_name"),
            outputs=None,
            name="explore_usa_county",
        ),
        node(
            func=explore_dataset,
            inputs=dict(df="worldometer_data", name="params:dataset6_name"),
            outputs=None,
            name="explore_worldometer",
        ),

        # ==========================
        # Limpieza de datasets
        # ==========================
        node(
            func=clean_province_state,
            inputs="covid_19_clean_complete",
            outputs="covid_19_clean_complete_final",
            name="clean_province_state_node"
        ),
        node(
            func=clean_usa_county,
            inputs="usa_county_wise",
            outputs="usa_county_wise_clean",
            name="clean_usa_county_node"
        ),
        node(
            func=clean_worldometer_data,
            inputs="worldometer_data",
            outputs="worldometer_data_clean",
            name="clean_worldometer_data_node"
        ),

        # ==========================
        # Revisión de nulos y duplicados
        # Genera CSVs en data/02_reporting
        # ==========================
        node(
            func=check_multiple_datasets,
            inputs=[
                "covid_19_clean_complete",
                "day_wise",
                "full_grouped",
                "usa_county_wise",
                "worldometer_data",
                "country_wise_latest"
            ],
            outputs=None,
            name="check_all_datasets"
        ),

        # ==========================
        # Generación de gráficos
        # ==========================
        node(
            func=plot_cases,
            inputs="covid_19_clean_complete_final",
            outputs="covid_cases_plot",
            name="plot_cases_node"
        ),
        node(
            func=plot_cases_by_continent,
            inputs="worldometer_data_clean",
            outputs="cases_by_continent_plot",
            name="plot_cases_by_continent_node"
        ),
    ])
