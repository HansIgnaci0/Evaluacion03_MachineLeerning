import pandas as pd
import numpy as np
from typing import Tuple, Dict
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
# Imports pesados de plotting/jerárquicos se hacen dentro de las funciones


def load_latest_by_country(csv_path: str) -> pd.DataFrame:
    """Carga el CSV con la serie temporal y devuelve una fila por `Country/Region`
    tomando la última fecha disponible por país.
    """
    df = pd.read_csv(csv_path, parse_dates=["Date"]) if csv_path else pd.DataFrame()
    if df.empty:
        return df
    # ordenar por fecha y tomar la última entrada por país
    df = df.sort_values("Date").groupby("Country/Region", as_index=False).last()
    return df


def preprocess_for_clustering(df: pd.DataFrame, features=None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Selecciona columnas numéricas relevantes y devuelve matriz escalada + df original.

    Por defecto usa ['Confirmed','Deaths','Recovered','Active','Lat','Long'].
    """
    if df is None or df.empty:
        return pd.DataFrame(), pd.DataFrame()
    if features is None:
        features = [c for c in ["Confirmed", "Deaths", "Recovered", "Active", "Lat", "Long"] if c in df.columns]
    X = df[features].fillna(0).astype(float)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=features, index=df.index)
    return X_scaled_df, df


def run_kmeans(X: pd.DataFrame, n_clusters: int = 4, random_state: int = 42) -> Tuple[np.ndarray, KMeans]:
    model = KMeans(n_clusters=n_clusters, random_state=random_state)
    labels = model.fit_predict(X)
    return labels, model


def run_dbscan(X: pd.DataFrame, eps: float = 0.5, min_samples: int = 5) -> Tuple[np.ndarray, DBSCAN]:
    model = DBSCAN(eps=eps, min_samples=min_samples)
    labels = model.fit_predict(X)
    return labels, model


def run_hierarchical(X: pd.DataFrame, n_clusters: int = 4, linkage: str = "ward") -> Tuple[np.ndarray, AgglomerativeClustering]:
    model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
    labels = model.fit_predict(X)
    return labels, model


def compute_clustering_metrics(X: pd.DataFrame, labels: np.ndarray) -> Dict[str, float]:
    metrics = {}
    try:
        n_labels = len(set(labels)) - (1 if -1 in labels else 0)
    except Exception:
        n_labels = 0

    if n_labels > 1 and n_labels < len(X):
        try:
            metrics["silhouette"] = float(silhouette_score(X, labels))
        except Exception:
            metrics["silhouette"] = float("nan")
        try:
            metrics["davies_bouldin"] = float(davies_bouldin_score(X, labels))
        except Exception:
            metrics["davies_bouldin"] = float("nan")
        try:
            metrics["calinski_harabasz"] = float(calinski_harabasz_score(X, labels))
        except Exception:
            metrics["calinski_harabasz"] = float("nan")
    else:
        metrics["silhouette"] = float("nan")
        metrics["davies_bouldin"] = float("nan")
        metrics["calinski_harabasz"] = float("nan")

    return metrics


def save_cluster_results(df_original: pd.DataFrame, labels: np.ndarray, out_csv: str) -> str:
    out = df_original.copy()
    out["cluster"] = labels
    out.to_csv(out_csv, index=False)
    return out_csv


def elbow_method(X: pd.DataFrame, k_min: int = 2, k_max: int = 10, out_csv: str = None, out_plot: str = None) -> Dict:
    """Calcula inercia para KMeans en el rango [k_min, k_max] y guarda CSV/plot si se solicitan.

    Retorna dict con listas 'k' e 'inertia'.
    """
    results = {"k": [], "inertia": []}
    if X is None or X.shape[0] == 0:
        return results
    for k in range(max(1, k_min), max(k_min, k_max) + 1):
        try:
            km = KMeans(n_clusters=k, random_state=42)
            km.fit(X)
            inertia = float(km.inertia_)
        except Exception:
            inertia = float("nan")
        results["k"].append(k)
        results["inertia"].append(inertia)

    if out_csv:
        pd.DataFrame(results).to_csv(out_csv, index=False)

    if out_plot:
        try:
            import matplotlib.pyplot as plt
        except Exception:
            plt = None

        if plt is not None:
            plt.figure(figsize=(6, 4))
            plt.plot(results["k"], results["inertia"], marker="o")
            plt.xlabel("k")
            plt.ylabel("Inertia")
            plt.title("Elbow Method")
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(out_plot)
            plt.close()

    return results


def plot_dendrogram(X: pd.DataFrame, method: str = "ward", out_plot: str = None) -> str:
    """Calcula linkage y dibuja dendrograma; guarda plot si se proporciona ruta."""
    if X is None or X.shape[0] == 0:
        return ""
    try:
        # import localmente para evitar error en import-time si scipy no está instalado
        from scipy.cluster.hierarchy import linkage, dendrogram
    except Exception:
        return ""

    try:
        Z = linkage(X, method=method)
    except Exception:
        try:
            Z = linkage(X, method="average")
        except Exception:
            return ""

    if out_plot:
        try:
            import matplotlib.pyplot as plt
        except Exception:
            plt = None

        if plt is not None:
            plt.figure(figsize=(8, 6))
            dendrogram(Z, no_labels=True)
            plt.title("Dendrograma ({})".format(method))
            plt.ylabel("Distancia")
            plt.tight_layout()
            plt.savefig(out_plot)
            plt.close()
            return out_plot

    return ""


def save_metrics_csv(metrics: Dict, out_csv: str) -> str:
    """Guarda un diccionario de métricas en un CSV aplanando si es necesario."""
    if not metrics:
        pd.DataFrame([{}]).to_csv(out_csv, index=False)
        return out_csv
    # convertir valores simples
    df = pd.DataFrame([metrics])
    df.to_csv(out_csv, index=False)
    return out_csv
