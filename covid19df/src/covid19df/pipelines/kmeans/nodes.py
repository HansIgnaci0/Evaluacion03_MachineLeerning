import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import logging

logger = logging.getLogger(__name__)


def load_features(df: pd.DataFrame) -> pd.DataFrame:
    X = df.select_dtypes(include=[np.number]).copy().fillna(0)
    try:
        logger.info(
            "[kmeans] Features: %s filas, %s columnas numéricas",
            X.shape[0], X.shape[1]
        )
        if X.shape[1] > 0:
            logger.info("[kmeans] Columnas ejemplo: %s", list(X.columns[:5]))
            logger.info("[kmeans] Primeras 3 filas:\n%s", X.head(3).to_string(index=False))
    except Exception as e:
        logger.warning("[kmeans] No se pudo loguear resumen de features: %s", e)
    return X


def run_kmeans(X: pd.DataFrame, params: dict) -> pd.DataFrame:
    n_clusters = int(params.get("kmeans_n_clusters", 5))
    random_state = int(params.get("random_state", 42))
    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=random_state)
    labels = km.fit_predict(X.values)
    # Construir un DataFrame limpio con columnas clave + cluster
    desired_cols = ["Lat", "Long", "Confirmed", "Deaths", "Recovered"]
    present_cols = [c for c in desired_cols if c in X.columns]
    if not present_cols:
        # Si no están esas columnas, incluir las primeras 5 numéricas como fallback
        present_cols = list(X.columns[:5])
    out_df = X[present_cols].copy()
    out_df["cluster"] = labels
    try:
        counts = pd.Series(labels).value_counts().sort_index().to_dict()
        logger.info("[kmeans] n_clusters=%s, inertia=%.4f", n_clusters, getattr(km, "inertia_", float("nan")))
        logger.info("[kmeans] Distribución por cluster: %s", counts)
        logger.info("[kmeans] Columnas en salida: %s", present_cols + ["cluster"])
        logger.info("[kmeans] Salida (primeras 5 filas):\n%s", out_df.head(5).to_string(index=False))
    except Exception as e:
        logger.warning("[kmeans] No se pudo loguear resumen de clustering: %s", e)
    return out_df


def plot_kmeans(X: pd.DataFrame, labels_df: pd.DataFrame) -> plt.Figure:
    # Alinear índices por si hubiera submuestreo u orden distinto
    common_idx = X.index.intersection(labels_df.index)
    Xp = X.loc[common_idx]
    yp = labels_df.loc[common_idx, "cluster"].values

    # Estándar + PCA(2) solo para visualización
    scaler = StandardScaler(copy=False)
    X_scaled = scaler.fit_transform(Xp.values)
    pca = PCA(n_components=2, random_state=42)
    emb = pca.fit_transform(X_scaled)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(emb[:, 0], emb[:, 1], c=yp, cmap="tab10", s=20, alpha=0.8)
    ax.set_title("K-Means clusters (proyección PCA 2D)")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    try:
        logger.info("[kmeans] Plot PCA generado con %s puntos", len(common_idx))
    except Exception:
        pass
    return fig

def kmeans_metrics(X: pd.DataFrame, labels_df: pd.DataFrame) -> pd.DataFrame:
    # Alinear y escalar como en el plot
    common_idx = X.index.intersection(labels_df.index)
    Xp = X.loc[common_idx]
    yp = labels_df.loc[common_idx, "cluster"].values
    scaler = StandardScaler(copy=False)
    X_scaled = scaler.fit_transform(Xp.values)

    # Evitar métricas si sólo hay 1 cluster
    if len(np.unique(yp)) < 2:
        return pd.DataFrame({
            "silhouette": [np.nan],
            "davies_bouldin": [np.nan],
            "calinski_harabasz": [np.nan],
        })

    sil = silhouette_score(X_scaled, yp)
    dbi = davies_bouldin_score(X_scaled, yp)
    chi = calinski_harabasz_score(X_scaled, yp)
    return pd.DataFrame({
        "silhouette": [sil],
        "davies_bouldin": [dbi],
        "calinski_harabasz": [chi],
    })

def kmeans_elbow(X: pd.DataFrame, params: dict) -> plt.Figure:
    # Escalar y calcular inertia para k en rango
    k_min = int(params.get("kmeans_elbow_min_k", 2))
    k_max = int(params.get("kmeans_elbow_max_k", 10))
    scaler = StandardScaler(copy=False)
    X_scaled = scaler.fit_transform(X.values)

    ks = list(range(k_min, k_max + 1))
    inertias = []
    for k in ks:
        km = KMeans(n_clusters=k, n_init=10, random_state=int(params.get("random_state", 42)))
        km.fit(X_scaled)
        inertias.append(km.inertia_)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(ks, inertias, marker="o")
    ax.set_xlabel("k")
    ax.set_ylabel("Inertia")
    ax.set_title("K-Means Elbow")
    return fig
