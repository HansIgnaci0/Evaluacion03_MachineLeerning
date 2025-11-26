import pandas as pd
from typing import Tuple, Dict


def run_pca(X, n_components: int = 2):
    """Ejecuta PCA sobre X y devuelve embeddings, explained_variance_ratio y components."""
    from sklearn.decomposition import PCA

    pca = PCA(n_components=n_components)
    embeddings = pca.fit_transform(X)
    return {
        "embeddings": embeddings,
        "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "components": pca.components_.tolist(),
    }


def run_tsne(X, n_components: int = 2, perplexity: float = 30.0, random_state: int = 42):
    """Ejecuta t-SNE y devuelve embeddings."""
    from sklearn.manifold import TSNE

    tsne = TSNE(n_components=n_components, perplexity=perplexity, random_state=random_state)
    embeddings = tsne.fit_transform(X)
    return {"embeddings": embeddings}


def save_embeddings(df_indexed: pd.DataFrame, embeddings, out_csv: str) -> str:
    """Guarda embeddings en CSV junto con el índice de `df_indexed` (p. ej. Country/Region)."""
    cols = [f"dim_{i+1}" for i in range(embeddings.shape[1])]
    out = pd.DataFrame(embeddings, columns=cols, index=df_indexed.index)
    # si df_indexed tiene una columna identificadora, la preservamos como primera columna
    try:
        if "Country/Region" in df_indexed.columns:
            out.insert(0, "Country/Region", df_indexed["Country/Region"].values)
    except Exception:
        pass
    out.to_csv(out_csv, index=False)
    return out_csv
