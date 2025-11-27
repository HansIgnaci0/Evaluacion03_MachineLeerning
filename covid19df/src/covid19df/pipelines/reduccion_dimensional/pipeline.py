from kedro.pipeline import Pipeline, node
from .nodes import (
    load_features,
    run_pca,
    plot_biplot,
    run_tsne,
    plot_tsne_2d,
    plot_tsne_3d,
)


def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                func=load_features,
                inputs="df1",
                outputs="X_features",
                name="load_features_node",
            ),
            node(
                func=run_pca,
                inputs=["X_features", "params:reduccion_dimensional"],
                outputs=["pca_embeddings", "pca_loadings", "pca_explained_variance"],
                name="run_pca_node",
            ),
            node(
                func=plot_biplot,
                inputs=["pca_embeddings", "pca_loadings"],
                outputs="pca_biplot",
                name="plot_biplot_node",
            ),
            node(
                func=run_tsne,
                inputs=["X_features", "params:reduccion_dimensional"],
                outputs=["tsne_embeddings_2d", "tsne_embeddings_3d"],
                name="run_tsne_node",
            ),
            node(
                func=plot_tsne_2d,
                inputs="tsne_embeddings_2d",
                outputs="tsne_plot_2d",
                name="plot_tsne_2d_node",
            ),
            node(
                func=plot_tsne_3d,
                inputs="tsne_embeddings_3d",
                outputs="tsne_plot_3d",
                name="plot_tsne_3d_node",
            ),
        ]
    )
