"""Nodes para el pipeline de reducción de dimensionalidad (PCA y t-SNE).

Cada nodo devuelve objetos simples que Kedro almacenará según el `catalog.yml`.
"""
from typing import Tuple
import logging

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

logger = logging.getLogger(__name__)


def load_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extrae columnas numéricas y rellena NA con 0.

    Args:
        df: dataframe limpio (por ejemplo `df1` del catálogo).

    Returns:
        DataFrame con solo columnas numéricas y mismos índices.
    """
    X = df.select_dtypes(include=[np.number]).copy()
    X = X.fillna(0)
    try:
        logger.info(
            "[reduccion_dimensional] Features: %s filas, %s columnas numéricas",
            X.shape[0],
            X.shape[1],
        )
        if X.shape[1] > 0:
            preview_cols = list(X.columns[:5])
            logger.info(
                "[reduccion_dimensional] Columnas (ejemplo): %s%s",
                preview_cols,
                "..." if X.shape[1] > 5 else "",
            )
            logger.info(
                "[reduccion_dimensional] Primeras 3 filas:\n%s",
                X.head(3).to_string(index=False),
            )
    except Exception as e:
        logger.warning("[reduccion_dimensional] No se pudo loguear resumen de features: %s", e)
    return X


def run_pca(X: pd.DataFrame, params: dict) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Ejecuta PCA y devuelve embeddings, loadings y varianza explicada como DataFrame.

    Args:
        X: DataFrame de características numéricas.
        params: diccionario de parámetros con claves:
          - n_pca_components (int)
          - random_state (int)

    Returns:
        pca_embeddings (DataFrame), pca_loadings (DataFrame), explained_variance_df (DataFrame)
    """
    requested_components = int(params.get("n_pca_components", 10))
    random_state = int(params.get("random_state", 42))

    # Limitar componentes a lo que permite el dataset
    max_possible = min(X.shape[0], X.shape[1])
    n_components = min(requested_components, max_possible)
    if n_components < requested_components:
        print(
            f"[run_pca] Aviso: n_pca_components solicitado={requested_components} reducido a {n_components} "
            f"(min(n_samples, n_features)={max_possible})."
        )

    pca = PCA(n_components=n_components, random_state=random_state)
    X_pca = pca.fit_transform(X.values)

    emb_cols = [f"pca_{i+1}" for i in range(X_pca.shape[1])]
    pca_embeddings = pd.DataFrame(X_pca, index=X.index, columns=emb_cols)

    loadings = pd.DataFrame(pca.components_.T, index=X.columns, columns=[f"pc_{i+1}" for i in range(pca.n_components_)])

    explained = pca.explained_variance_ratio_.tolist()
    explained_df = pd.DataFrame(
        {
            "component": [f"PC{i+1}" for i in range(len(explained))],
            "explained_variance_ratio": explained,
        }
    )

    try:
        cum = np.cumsum(explained_df["explained_variance_ratio"].values)
        top = min(5, len(explained_df))
        logger.info(
            "[reduccion_dimensional][PCA] n_components usados: %s",
            pca.n_components_,
        )
        logger.info(
            "[reduccion_dimensional][PCA] Varianza explicada (primeras %s): %s",
            top,
            np.round(explained_df["explained_variance_ratio"].values[:top], 4).tolist(),
        )
        logger.info(
            "[reduccion_dimensional][PCA] Varianza acumulada (primeras %s): %s",
            top,
            np.round(cum[:top], 4).tolist(),
        )
        logger.info(
            "[reduccion_dimensional][PCA] Embeddings shape: %s, Loadings shape: %s",
            pca_embeddings.shape,
            loadings.shape,
        )
        logger.info(
            "[reduccion_dimensional][PCA] Embeddings (3 filas):\n%s",
            pca_embeddings.head(3).to_string(index=False),
        )
    except Exception as e:
        logger.warning("[reduccion_dimensional][PCA] No se pudo loguear resumen: %s", e)

    return pca_embeddings, loadings, explained_df


def plot_biplot(pca_embeddings: pd.DataFrame, pca_loadings: pd.DataFrame) -> plt.Figure:
    """Genera un biplot con las dos primeras componentes principales.

    Args:
        pca_embeddings: DataFrame con columnas `pca_1`, `pca_2`, ...
        pca_loadings: DataFrame de loadings (features x pcs)

    Returns:
        Matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    x = pca_embeddings.iloc[:, 0]
    y = pca_embeddings.iloc[:, 1] if pca_embeddings.shape[1] > 1 else np.zeros(len(x))

    ax.scatter(x, y, alpha=0.6, s=25)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("PCA Biplot (PC1 vs PC2)")

    # Mostrar flechas para las variables más importantes (top 10 por magnitud)
    if "pc_1" in pca_loadings.columns and "pc_2" in pca_loadings.columns:
        loadings = pca_loadings[["pc_1", "pc_2"]]
        # seleccionar top features por norma
        norms = np.linalg.norm(loadings.values, axis=1)
        top_idx = np.argsort(-norms)[:10]
        for i in top_idx:
            feat = loadings.index[i]
            lx, ly = loadings.values[i, 0], loadings.values[i, 1]
            ax.arrow(0, 0, lx * max(x), ly * max(y), color="r", alpha=0.6, head_width=0.02)
            ax.text(lx * max(x) * 1.15, ly * max(y) * 1.15, feat, color="r", fontsize=8)

    return fig


def run_tsne(X: pd.DataFrame, params: dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Ejecuta t-SNE en 2D y 3D. Devuelve dos DataFrames con embeddings.

    Args:
        X: DataFrame de features numéricas.
        params: diccionario con claves opcionales `perplexity`, `n_iter`, `random_state`.

    Returns:
        tsne_2d (DataFrame), tsne_3d (DataFrame)
    """
    perplexity = float(params.get("tsne_perplexity", 30.0))
    n_iter = int(params.get("tsne_n_iter", 1000))
    random_state = int(params.get("random_state", 42))
    max2d = int(params.get("tsne_max_samples_2d", 2000))
    max3d = int(params.get("tsne_max_samples_3d", 1000))
    enable_3d = bool(params.get("tsne_enable_3d", True))

    # Por rendimiento, inicializamos con PCA a 50 dimensiones si hay muchas características
    init_components = min(50, X.shape[1])
    pca_init = PCA(n_components=init_components, random_state=random_state)

    # Muestreo para acelerar t-SNE 2D
    if X.shape[0] > max2d:
        idx2 = (
            X.sample(n=max2d, random_state=random_state).index
        )
        X2 = X.loc[idx2]
        print(f"[run_tsne] Submuestreando 2D: {X.shape[0]} -> {len(X2)} filas")
    else:
        idx2 = X.index
        X2 = X

    X2_init = pca_init.fit_transform(X2.values)

    # Algunas versiones pueden usar 'n_iter' y otras 'max_iter'; también usar verbose si existe
    import inspect
    tsne_sig = inspect.signature(TSNE.__init__)
    tsne_kwargs_base = {
        "n_components": 2,
        "perplexity": perplexity,
        "init": "pca",
        "random_state": random_state,
    }
    if "n_iter" in tsne_sig.parameters:
        tsne_kwargs_base["n_iter"] = n_iter
    elif "max_iter" in tsne_sig.parameters:
        tsne_kwargs_base["max_iter"] = n_iter
    if "verbose" in tsne_sig.parameters:
        tsne_kwargs_base["verbose"] = 1

    tsne2 = TSNE(**tsne_kwargs_base)
    emb2 = tsne2.fit_transform(X2_init)
    df2 = pd.DataFrame(emb2, index=idx2, columns=["tsne_1", "tsne_2"])
    try:
        iter_key = "n_iter" if "n_iter" in tsne_sig.parameters else ("max_iter" if "max_iter" in tsne_sig.parameters else "n_iter")
        logger.info(
            "[reduccion_dimensional][t-SNE 2D] muestras=%s, perplexity=%s, %s=%s",
            len(idx2), perplexity, iter_key, n_iter
        )
        logger.info(
            "[reduccion_dimensional][t-SNE 2D] Embeddings shape: %s",
            df2.shape,
        )
        logger.info(
            "[reduccion_dimensional][t-SNE 2D] Primeras 3 filas:\n%s",
            df2.head(3).to_string(index=False),
        )
    except Exception as e:
        logger.warning("[reduccion_dimensional][t-SNE 2D] No se pudo loguear resumen: %s", e)

    # 3D: puede ser muy costoso (usa método 'exact'); submuestrear más o deshabilitar
    if enable_3d:
        if X.shape[0] > max3d:
            idx3 = X.sample(n=max3d, random_state=random_state).index
            X3 = X.loc[idx3]
            print(f"[run_tsne] Submuestreando 3D: {X.shape[0]} -> {len(X3)} filas")
        else:
            idx3 = X.index
            X3 = X
        X3_init = pca_init.fit_transform(X3.values)

        tsne_kwargs_3d = tsne_kwargs_base.copy()
        tsne_kwargs_3d["n_components"] = 3
        tsne3 = TSNE(**tsne_kwargs_3d)
        emb3 = tsne3.fit_transform(X3_init)
        df3 = pd.DataFrame(emb3, index=idx3, columns=["tsne_1", "tsne_2", "tsne_3"])
        try:
            iter_key = "n_iter" if "n_iter" in tsne_sig.parameters else ("max_iter" if "max_iter" in tsne_sig.parameters else "n_iter")
            logger.info(
                "[reduccion_dimensional][t-SNE 3D] muestras=%s, perplexity=%s, %s=%s",
                len(idx3), perplexity, iter_key, n_iter
            )
            logger.info(
                "[reduccion_dimensional][t-SNE 3D] Embeddings shape: %s",
                df3.shape,
            )
            logger.info(
                "[reduccion_dimensional][t-SNE 3D] Primeras 3 filas:\n%s",
                df3.head(3).to_string(index=False),
            )
        except Exception as e:
            logger.warning("[reduccion_dimensional][t-SNE 3D] No se pudo loguear resumen: %s", e)
    else:
        print("[run_tsne] 3D deshabilitado por parámetro tsne_enable_3d=false")
        df3 = pd.DataFrame(columns=["tsne_1", "tsne_2", "tsne_3"])  # vacío

    return df2, df3


def plot_tsne_2d(tsne_embeddings_2d: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(tsne_embeddings_2d["tsne_1"], tsne_embeddings_2d["tsne_2"], s=20, alpha=0.7)
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.set_title("t-SNE 2D")
    return fig


def plot_tsne_3d(tsne_embeddings_3d: pd.DataFrame) -> plt.Figure:
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 - requerido para 3D

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(tsne_embeddings_3d["tsne_1"], tsne_embeddings_3d["tsne_2"], tsne_embeddings_3d["tsne_3"], s=20, alpha=0.7)
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.set_zlabel("t-SNE 3")
    ax.set_title("t-SNE 3D")
    return fig
