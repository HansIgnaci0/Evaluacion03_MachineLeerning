from kedro.pipeline import Pipeline, node
from covid19df.pipelines.unsupervised.clustering.nodes import (
    load_latest_by_country,
    preprocess_for_clustering,
)
from .nodes import run_pca, run_tsne, save_embeddings


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline([
        node(
            func=load_latest_by_country,
            inputs="params:clustering_input_csv",
            outputs="raw_country_df",
            name="reduction_load_latest_by_country",
        ),
        node(
            func=preprocess_for_clustering,
            inputs="raw_country_df",
            outputs=["X_scaled_df", "raw_country_df"],
            name="reduction_preprocess",
        ),
        node(
            func=run_pca,
            inputs=["X_scaled_df", "params:reduction_pca_n_components"],
            outputs="pca_results",
            name="run_pca_node",
        ),
        node(
            func=save_embeddings,
            inputs=["raw_country_df", "pca_results:embeddings", "params:reduction_pca_output"],
            outputs=None,
            name="save_pca_embeddings",
        ),
        node(
            func=run_tsne,
            inputs=["X_scaled_df", "params:reduction_tsne_n_components", "params:reduction_tsne_perplexity", "params:reduction_random_state"],
            outputs="tsne_results",
            name="run_tsne_node",
        ),
        node(
            func=save_embeddings,
            inputs=["raw_country_df", "tsne_results:embeddings", "params:reduction_tsne_output"],
            outputs=None,
            name="save_tsne_embeddings",
        ),
    ])
