from kedro.pipeline import Pipeline, node
from .nodes import (
    load_latest_by_country,
    preprocess_for_clustering,
    run_kmeans,
    run_dbscan,
    run_hierarchical,
    compute_clustering_metrics,
    save_cluster_results,
    elbow_method,
    plot_dendrogram,
    save_metrics_csv,
)


def create_pipeline(**kwargs) -> Pipeline:
    """Crea el pipeline de clustering con nodos básicos.

    Entradas esperadas en `catalog` o `params`:
    - `params:clustering_input_csv` : ruta al CSV de entrada (string)
    - `params:clustering_kmeans_output` : ruta CSV para guardar resultados KMeans
    """
    return Pipeline([
        node(
            func=load_latest_by_country,
            inputs="params:clustering_input_csv",
            outputs="raw_country_df",
            name="load_latest_by_country_node",
        ),
        node(
            func=preprocess_for_clustering,
            inputs="raw_country_df",
            outputs=["X_scaled_df", "raw_country_df"],
            name="preprocess_for_clustering_node",
        ),
        node(
            func=run_kmeans,
            inputs=["X_scaled_df", "params:clustering_n_clusters"],
            outputs=["kmeans_labels", "kmeans_model"],
            name="run_kmeans_node",
        ),
        node(
            func=run_dbscan,
            inputs=["X_scaled_df", "params:clustering_dbscan_eps", "params:clustering_dbscan_min_samples"],
            outputs=["dbscan_labels", "dbscan_model"],
            name="run_dbscan_node",
        ),
        node(
            func=run_hierarchical,
            inputs=["X_scaled_df", "params:clustering_n_clusters"],
            outputs=["hier_labels", "hier_model"],
            name="run_hierarchical_node",
        ),
        node(
            func=compute_clustering_metrics,
            inputs=["X_scaled_df", "kmeans_labels"],
            outputs="kmeans_metrics",
            name="kmeans_metrics_node",
        ),
        node(
            func=compute_clustering_metrics,
            inputs=["X_scaled_df", "dbscan_labels"],
            outputs="dbscan_metrics",
            name="dbscan_metrics_node",
        ),
        node(
            func=compute_clustering_metrics,
            inputs=["X_scaled_df", "hier_labels"],
            outputs="hier_metrics",
            name="hier_metrics_node",
        ),
        node(
            func=elbow_method,
            inputs=["X_scaled_df", "params:clustering_k_min", "params:clustering_k_max", "params:clustering_elbow_csv", "params:clustering_elbow_plot"],
            outputs="elbow_results",
            name="elbow_method_node",
        ),
        node(
            func=plot_dendrogram,
            inputs=["X_scaled_df", "params:clustering_hier_method", "params:clustering_dendrogram_plot"],
            outputs="dendrogram_plot",
            name="dendrogram_node",
        ),
        node(
            func=save_metrics_csv,
            inputs=["kmeans_metrics", "params:clustering_kmeans_metrics_csv"],
            outputs=None,
            name="save_kmeans_metrics_node",
        ),
        node(
            func=save_metrics_csv,
            inputs=["dbscan_metrics", "params:clustering_dbscan_metrics_csv"],
            outputs=None,
            name="save_dbscan_metrics_node",
        ),
        node(
            func=save_metrics_csv,
            inputs=["hier_metrics", "params:clustering_hier_metrics_csv"],
            outputs=None,
            name="save_hier_metrics_node",
        ),
        node(
            func=save_cluster_results,
            inputs=["raw_country_df", "kmeans_labels", "params:clustering_kmeans_output"],
            outputs=None,
            name="save_kmeans_results_node",
        ),
    ])
