"""
Nodes del pipeline 'eda' para el proyecto COVID-19.
Cada función representa un nodo del pipeline Kedro y contiene docstrings para documentación.
Generado con Kedro 1.0.0
"""

import pandas as pd
import matplotlib.pyplot as plt


def explore_dataset(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """
    Nodo para explorar un dataset: imprime estadísticas básicas y la forma del dataset.

    Args:
        df (pd.DataFrame): Dataset a explorar.
        name (str): Nombre del dataset para imprimir.

    Returns:
        pd.DataFrame: Devuelve el mismo dataset sin modificaciones.
    """
    print(f"\n===== Explorando dataset: {name} =====")
    print("Shape:", df.shape)
    print("Columnas:", df.columns.tolist())
    print(df.head(3))
    print(df.describe(include='all').transpose().head(5))
    return df


def plot_cases(covid_19_clean_complete: pd.DataFrame) -> plt.Figure:
    """
    Nodo para crear un gráfico de barras de los 10 países con más casos confirmados.

    Args:
        covid_19_clean_complete (pd.DataFrame): Dataset limpio de COVID-19.

    Returns:
        plt.Figure: Objeto de la figura del gráfico.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    covid_19_clean_complete.groupby("Country/Region")["Confirmed"].max() \
        .sort_values(ascending=False)[:10].plot(kind="bar", ax=ax)
    ax.set_title("Top 10 países con más casos confirmados")
    ax.set_ylabel("Casos confirmados")
    return fig


def plot_cases_by_continent(worldometer_data_clean: pd.DataFrame) -> str:
    """
    Nodo para generar un gráfico de pastel de casos totales por continente
    y guardarlo en data/04_models.

    Args:
        worldometer_data_clean (pd.DataFrame): Dataset limpio de Worldometer.

    Returns:
        str: Ruta del archivo PNG generado.
    """
    df = worldometer_data_clean.groupby("Continent")["TotalCases"].sum()

    plt.figure(figsize=(8, 8))
    plt.pie(df, labels=df.index, autopct='%1.1f%%', startangle=140, colors=plt.cm.tab20.colors)
    plt.title("Distribución de casos COVID-19 por continente")

    output_path = "data/04_models/cases_by_continent.png"
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()

    return output_path


def check_multiple_datasets(
    covid_19_clean_complete: pd.DataFrame,
    day_wise: pd.DataFrame,
    full_grouped: pd.DataFrame,
    usa_county_wise: pd.DataFrame,
    worldometer_data: pd.DataFrame,
    country_wise_latest: pd.DataFrame,
) -> None:
    """
    Nodo para generar reportes de nulos y duplicados para múltiples datasets.
    Guarda un CSV por cada dataset en data/02_reporting.

    Args:
        covid_19_clean_complete, day_wise, full_grouped, usa_county_wise,
        worldometer_data, country_wise_latest (pd.DataFrame): Datasets a analizar.

    Returns:
        None
    """
    datasets = {
        "covid_19_clean_complete": covid_19_clean_complete,
        "day_wise": day_wise,
        "full_grouped": full_grouped,
        "usa_county_wise": usa_county_wise,
        "worldometer_data": worldometer_data,
        "country_wise_latest": country_wise_latest,
    }

    for name, df in datasets.items():
        report = pd.DataFrame({
            "column": df.columns,
            "missing_values": df.isnull().sum(),
            "duplicated_rows": [df.duplicated().sum()] * len(df.columns)
        })
        report.to_csv(f"data/02_reporting/missing_duplicates_report_{name}.csv", index=False)


def clean_province_state(df: pd.DataFrame) -> pd.DataFrame:
    """
    Nodo para limpiar los nulos de la columna 'Province/State'
    reemplazándolos por 'Unknown' y guardar el dataset limpio.

    Args:
        df (pd.DataFrame): Dataset con posibles nulos en 'Province/State'.

    Returns:
        pd.DataFrame: Dataset limpio guardado en data/03_intermediate.
    """
    df["Province/State"] = df["Province/State"].fillna("Unknown")
    df.to_csv("data/03_intermediate/covid_19_clean_complete_CLEAN.csv", index=False)
    return df


def clean_usa_county(df: pd.DataFrame) -> pd.DataFrame:
    """
    Nodo para limpiar los nulos del dataset usa_county_wise:
    - Columnas FIPS y Admin2 rellenadas con 'Unknown'.

    Args:
        df (pd.DataFrame): Dataset original con posibles nulos.

    Returns:
        pd.DataFrame: Dataset limpio guardado en data/03_intermediate.
    """
    df["FIPS"] = df["FIPS"].fillna("Unknown")
    df["Admin2"] = df["Admin2"].fillna("Unknown")
    df.to_csv("data/03_intermediate/usa_county_wise_CLEAN.csv", index=False)
    return df


def clean_worldometer_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Nodo para limpiar los nulos del dataset worldometer_data:
    - Columnas categóricas → 'Unknown'
    - Columnas numéricas → 0

    Args:
        df (pd.DataFrame): Dataset original con nulos.

    Returns:
        pd.DataFrame: Dataset limpio guardado en data/03_intermediate.
    """
    # Columnas categóricas
    categorical_cols = ["Continent", "WHO Region"]
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown")

    # Columnas numéricas
    numeric_cols = [
        "Population", "NewCases", "TotalDeaths", "NewDeaths", "TotalRecovered",
        "NewRecovered", "ActiveCases", "Serious,Critical", "Tot Cases/1M pop",
        "Deaths/1M pop", "TotalTests", "Tests/1M pop"
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    df.to_csv("data/03_intermediate/worldometer_data_CLEAN.csv", index=False)
    return df
