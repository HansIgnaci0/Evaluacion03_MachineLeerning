import logging
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

logger = logging.getLogger(__name__)


def load_features(df: pd.DataFrame) -> pd.DataFrame:
    X = df.select_dtypes(include=[np.number]).copy().fillna(0)
    try:
        logger.info("[gmm] Features: %s filas, %s columnas numéricas", X.shape[0], X.shape[1])
        if X.shape[1] > 0:
            logger.info("[gmm] Columnas ejemplo: %s", list(X.columns[:5]))
            logger.info("[gmm] Primeras 3 filas:\n%s", X.head(3).to_string(index=False))
    except Exception as e:
        logger.warning("[gmm] No se pudo loguear resumen de features: %s", e)
    return X


def run_gmm(X: pd.DataFrame, params: Dict) -> pd.DataFrame:
    """Ajusta GaussianMixture sobre una submuestra (configurable) y devuelve CSV limpio.

    Devuelve un DataFrame con columnas clave (Lat, Long, Confirmed, Deaths, Recovered cuando existan)
    más columnas `cluster` y `prob` (probabilidad posterior para la mezcla asignada).
    """
    n_components = int(params.get("gmm_n_components", 5))
    covariance_type = str(params.get("gmm_covariance_type", "full"))
    random_state = int(params.get("gmm_random_state", 42))
    max_samples = int(params.get("gmm_max_samples", 5000))
    reg_covar = float(params.get("gmm_reg_covar", 1e-6))

    # Submuestreo para rendimiento
    if X.shape[0] > max_samples:
        X_use = X.sample(n=max_samples, random_state=random_state)
        logger.info("[gmm] Submuestreo: %s -> %s filas (max_samples=%s)", X.shape[0], len(X_use), max_samples)
    else:
        X_use = X

    # Escalamos características para estabilidad numérica
    scaler = StandardScaler(copy=False)
    X_scaled = scaler.fit_transform(X_use.values)

    gm = GaussianMixture(
        n_components=n_components,
        covariance_type=covariance_type,
        random_state=random_state,
        reg_covar=reg_covar,
        init_params="kmeans",
    )
    try:
        gm.fit(X_scaled)
    except ValueError as e:
        logger.warning(
            "[gmm] Falló el ajuste inicial (%s). Reintentando con mayor reg_covar y/o 'diag'",
            e,
        )
        fallback_reg = max(reg_covar, 1e-3)
        fallback_cov = "diag" if covariance_type == "full" else covariance_type
        gm = GaussianMixture(
            n_components=n_components,
            covariance_type=fallback_cov,
            random_state=random_state,
            reg_covar=fallback_reg,
            init_params="kmeans",
        )
        gm.fit(X_scaled)

    probs = gm.predict_proba(X_scaled)
    labels = gm.predict(X_scaled)
    assigned_prob = probs.max(axis=1)

    desired_cols = ["Lat", "Long", "Confirmed", "Deaths", "Recovered"]
    present_cols = [c for c in desired_cols if c in X_use.columns]
    if not present_cols:
        present_cols = list(X_use.columns[:5])

    out_df = X_use[present_cols].copy()
    out_df["cluster"] = labels
    out_df["prob"] = assigned_prob

    # Log some useful info for professor verification
    try:
        logger.info(
            "[gmm] n_components=%s, covariance_type=%s, reg_covar=%s",
            n_components,
            covariance_type,
            reg_covar,
        )
        logger.info("[gmm] Weights: %s", np.round(gm.weights_, 4).tolist())
        logger.info("[gmm] Medias (primeras 2 componentes): %s", np.round(gm.means_[:2].tolist(), 4))
        logger.info("[gmm] BIC=%.2f, AIC=%.2f", gm.bic(X_scaled), gm.aic(X_scaled))
        logger.info("[gmm] Distribución por cluster: %s", pd.Series(labels).value_counts().sort_index().to_dict())
        logger.info("[gmm] Salida (primeras 5 filas):\n%s", out_df.head(5).to_string(index=False))
    except Exception as e:
        logger.warning("[gmm] No se pudo loguear resumen de GMM: %s", e)

    return out_df


def plot_gmm(X: pd.DataFrame, labels_df: pd.DataFrame) -> plt.Figure:
    # Alinear por índices: labels_df contiene la submuestra usada
    common_idx = X.index.intersection(labels_df.index)
    Xp = X.loc[common_idx]
    yp = labels_df.loc[common_idx, "cluster"].values

    # Visualización en PCA(2) del X estándar
    scaler = StandardScaler(copy=False)
    X_scaled = scaler.fit_transform(Xp.values)
    pca = PCA(n_components=2, random_state=42)
    emb = pca.fit_transform(X_scaled)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(emb[:, 0], emb[:, 1], c=yp, cmap="viridis", s=20, alpha=0.8)
    ax.set_title("GMM Clusters (proyección PCA 2D)")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    try:
        logger.info("[gmm] Plot PCA generado con %s puntos", len(common_idx))
    except Exception:
        pass
    return fig

def gmm_metrics(X: pd.DataFrame, labels_df: pd.DataFrame) -> pd.DataFrame:
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
