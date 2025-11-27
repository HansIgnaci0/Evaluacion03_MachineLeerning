import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from typing import Dict
import logging
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from scipy.cluster.hierarchy import linkage, dendrogram

logger = logging.getLogger(__name__)

def load_features(df: pd.DataFrame) -> pd.DataFrame:
    X = df.select_dtypes(include=[np.number]).copy().fillna(0)
    try:
        logger.info(
            "[jerarquico] Features: %s filas, %s columnas numéricas", X.shape[0], X.shape[1]
        )
        if X.shape[1] > 0:
            logger.info("[jerarquico] Columnas ejemplo: %s", list(X.columns[:5]))
            logger.info("[jerarquico] Primeras 3 filas:\n%s", X.head(3).to_string(index=False))
    except Exception as e:
        logger.warning("[jerarquico] No se pudo loguear resumen de features: %s", e)
    return X

def run_hierarchical(X: pd.DataFrame, params: Dict) -> pd.DataFrame:
    n_clusters = int(params.get("jerarquico_n_clusters", 5))
    linkage = str(params.get("jerarquico_linkage", "ward"))
    metric = str(params.get("jerarquico_metric", "euclidean"))
    max_samples = int(params.get("jerarquico_max_samples", 3000))
    random_state = int(params.get("jerarquico_random_state", 42))

    # Submuestreo para evitar explosión de memoria (O(n^2) en jerárquico)
    if X.shape[0] > max_samples:
        X_use = X.sample(n=max_samples, random_state=random_state)
        logger.info(
            "[jerarquico] Submuestreo: %s -> %s filas para clustering (max_samples=%s)",
            X.shape[0], len(X_use), max_samples,
        )
    else:
        X_use = X

    # Ajustar parámetros según API sklearn: para ward NO pasar metric=None (usa euclidean).
    if linkage == "ward":
        model = AgglomerativeClustering(
            n_clusters=n_clusters,
            linkage=linkage,
        )
        effective_metric = "euclidean"
    else:
        model = AgglomerativeClustering(
            n_clusters=n_clusters,
            linkage=linkage,
            metric=metric,
        )
        effective_metric = metric
    labels = model.fit_predict(X_use.values)

    desired_cols = ["Lat", "Long", "Confirmed", "Deaths", "Recovered"]
    present_cols = [c for c in desired_cols if c in X_use.columns]
    if not present_cols:
        present_cols = list(X_use.columns[:5])
    out_df = X_use[present_cols].copy()
    out_df["cluster"] = labels

    try:
        counts = out_df["cluster"].value_counts().sort_index().to_dict()
        logger.info(
            "[jerarquico] n_clusters=%s, linkage=%s, metric=%s", n_clusters, linkage, effective_metric
        )
        logger.info("[jerarquico] Distribución por cluster: %s", counts)
        logger.info(
            "[jerarquico] Salida (primeras 5 filas):\n%s", out_df.head(5).to_string(index=False)
        )
    except Exception as e:
        logger.warning("[jerarquico] No se pudo loguear resumen clustering: %s", e)
    return out_df

def plot_hierarchical(X: pd.DataFrame, labels_df: pd.DataFrame) -> plt.Figure:
    # Alinear con los índices del subconjunto usado en clustering
    common_idx = X.index.intersection(labels_df.index)
    Xp = X.loc[common_idx]
    yp = labels_df.loc[common_idx, "cluster"].values

    # Escalado + PCA 2D para visualización comparable
    scaler = StandardScaler(copy=False)
    X_scaled = scaler.fit_transform(Xp.values)
    pca = PCA(n_components=2, random_state=42)
    emb = pca.fit_transform(X_scaled)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(emb[:, 0], emb[:, 1], c=yp, cmap="tab10", s=20, alpha=0.8)
    ax.set_title("Hierarchical Clustering (proyección PCA 2D)")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    try:
        logger.info("[jerarquico] Plot PCA generado con %s puntos", len(common_idx))
    except Exception:
        pass
    return fig

def jerarquico_metrics(X: pd.DataFrame, labels_df: pd.DataFrame) -> pd.DataFrame:
    common_idx = X.index.intersection(labels_df.index)
    Xp = X.loc[common_idx]
    yp = labels_df.loc[common_idx, "cluster"].values
    scaler = StandardScaler(copy=False)
    X_scaled = scaler.fit_transform(Xp.values)
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

def jerarquico_dendrogram(X: pd.DataFrame, params: Dict) -> plt.Figure:
    # Submuestreo como en run_hierarchical para evitar OOM
    max_samples = int(params.get("jerarquico_max_samples", 3000))
    random_state = int(params.get("jerarquico_random_state", 42))
    linkage_method = str(params.get("jerarquico_linkage", "ward"))
    metric = str(params.get("jerarquico_metric", "euclidean"))
    X_use = X.sample(n=min(len(X), max_samples), random_state=random_state)
    scaler = StandardScaler(copy=False)
    X_scaled = scaler.fit_transform(X_use.values)
    # Para ward, scipy requiere euclidean y ignora metric distinto
    Z = linkage(X_scaled, method=linkage_method if linkage_method != "single" else "single", metric=metric)
    fig, ax = plt.subplots(figsize=(10, 6))
    dendrogram(Z, truncate_mode="level", p=5, ax=ax)
    ax.set_title("Dendrograma (truncado)")
    ax.set_xlabel("Observaciones (agrupadas)")
    ax.set_ylabel("Distancia")
    return fig
